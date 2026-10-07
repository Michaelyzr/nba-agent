"""Coach agent: before a user commits a pick, decide what to check, which lesson to teach and whether to nudge.

    from agents.coach_agent import advise
    a = advise(s, "AWY", history)       # s = coach.snapshot(...) at the decision time; history = earlier picks
    a["steps"]    # checks run, in order, each with why it was chosen and what it found
    a["lesson"]   # one concept card picked for this user and this pick
    a["nudge"]    # "pass", "caution" or "none": the Coach only ever suggests passing or caution
    a["message"]  # plain-language advice ending with coach.NOTICE

The planner is a loop: given what it has found so far, it picks the next as-of check
(break-even after fees, move since the 24 h anchor, long-shot price, news freshness,
M4 news shift, the user's own habits) or stops. Every input is the snapshot (as-of
view) and decision-time facts about earlier picks (side price, prior move, whether
a trade was made); results of earlier picks are never read, because their games may
not have finished at this decision time.

Offline mode (default) is rule-based and reproducible. With llm=True and a key, Gemini
rewrites the message from the tool outputs only; the text must pass the banned-words,
encouragement and number checks or the rule-based message is used. Responses are
cached in runs/llm_cache/ (git-ignored).
"""
import hashlib
import json
import re

import pandas as pd

from agents.coach import CHASE_MOVE, LONG_SHOT, NOTICE, banned_words, lessons
from agents.graph import DEFAULT_MIN_EDGE
from data_sources import ROOT

CACHE = ROOT / "runs" / "llm_cache"
PERCENT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%")
NUDGES = ("pass", "caution", "none")
STALE_NEWS_MIN = 30.0            # news older than this at decision time has usually been priced in
OVERTRADE_RATE = 0.8             # trading at least this share of decisions counts as overtrading
HABIT_SHARE = 0.5                # share of trades that makes chasing or long shots a habit
MIN_HISTORY = 3                  # decisions needed before habits are judged
ENCOURAGE = re.compile(r"bet more|stake more|increase (your )?stake|bigger (bet|stake)|double down|all[- ]in|"
                       r"go for it|you should (buy|back|bet)|must (buy|back|bet)|can'?t miss|easy money", re.I)
LESSON_FOR = {"break_even": "fees_and_spread", "priced_in": "priced_in", "long_shot": "long_shot",
              "news_freshness": "closing_line", "m4_shift": "injury_shift", "habits": "passing"}
FALLBACK_CARDS = {
    "fees_and_spread": ("The spread and the fee move your break-even",
                        "You buy at the ask and pay a fee of 7% x price x (1 - price) per contract. Small edges "
                        "disappear into these costs."),
    "priced_in": ("Is the news already in the price?",
                  "Markets react within minutes. Buying after a big move usually means paying for news everyone "
                  "already knows."),
    "long_shot": ("Long shots look cheap",
                  "A cheap contract pays a lot when it wins, but it loses most of the time, and an over-confident "
                  "model sees fake value here most often."),
    "closing_line": ("The closing line is the scoreboard",
                     "Judge a trade by whether you bought below the tip-off price, not by whether it won."),
    "injury_shift": ("How injury news moves a win probability",
                     "Losing a starter matters far more than losing a bench player; check which way the news "
                     "moves your side."),
    "passing": ("Passing is a position",
                "Most decisions offer no edge after fees. Trading every game is the fastest way to lose to the "
                "spread and fees."),
}


def habits(history: list) -> dict:
    """The user's habits from decision-time facts only (never P&L, CLV or results of earlier picks)."""
    made = list(history)
    trades = [h for h in made if h.get("team") is not None and "pass" not in (h.get("choice"), h.get("why"))]
    n, k = len(made), len(trades)
    chased = sum(bool(h.get("chased", (h.get("move") or 0) >= CHASE_MOVE)) for h in trades)
    long_shots = sum(bool(h.get("long_shot", (h.get("price") or 1) <= LONG_SHOT)) for h in trades)
    neg = sum((h.get("gap") or 0) < 0 for h in trades)
    out = {"decisions": n, "trades": k, "trade_rate": k / n if n else 0.0,
           "chase_share": chased / k if k else 0.0, "long_shot_share": long_shots / k if k else 0.0,
           "negative_edge_share": neg / k if k else 0.0}
    enough = n >= MIN_HISTORY
    out["overtrading"] = enough and out["trade_rate"] >= OVERTRADE_RATE
    out["chasing"] = enough and k > 0 and out["chase_share"] >= HABIT_SHARE
    out["long_shot_habit"] = enough and k > 0 and out["long_shot_share"] >= HABIT_SHARE
    return out


# --- as-of tools: each reads only the snapshot (and habits) and returns numbers plus a flag -------------------

def _side_shift(s, team):
    return (s["shift"] if team == s["home"] else -s["shift"]) + 0.0


def check_break_even(s, side, h):
    return {"ask": side["ask"], "fee": side["fee"], "breakeven": side["breakeven"], "estimate": side["p"],
            "gap": side["gap"], "flag": side["gap"] < DEFAULT_MIN_EDGE, "severe": side["gap"] < 0}


def check_priced_in(s, side, h):
    shift = _side_shift(s, side["team"])
    moved = side["move"] >= CHASE_MOVE
    return {"anchor": side["anchor"], "mid": side["mid"], "move": side["move"], "news_shift": shift,
            "flag": moved, "severe": moved and side["move"] >= shift}


def check_long_shot(s, side, h):
    cheap = side["ask"] <= LONG_SHOT
    return {"ask": side["ask"], "payout_multiple": 1 / side["ask"], "loses_if_price_right": 1 - side["ask"],
            "flag": cheap, "severe": cheap and side["gap"] < DEFAULT_MIN_EDGE}


def check_news_freshness(s, side, h):
    now = pd.Timestamp(s["as_of"])
    ages = [(now - pd.Timestamp(n["published_at"])).total_seconds() / 60 for n in s["news"]]
    age = min(ages) if ages else None
    stale = age is not None and age > STALE_NEWS_MIN
    return {"latest_news_minutes_ago": age, "items": len(ages), "flag": stale and abs(side["move"]) >= CHASE_MOVE,
            "severe": False}


def check_m4_shift(s, side, h):
    shift = _side_shift(s, side["team"])
    return {"p_home_before": s["p_home_before"], "p_home_after": s["p_home_after"], "news_shift": shift,
            "out": list(s["out"]), "flag": shift < 0, "severe": shift <= -0.02}


def check_habits(s, side, h):
    flag = h["overtrading"] or h["chasing"] or h["long_shot_habit"]
    return {**{k: h[k] for k in ("decisions", "trades", "trade_rate", "chase_share", "long_shot_share")},
            "overtrading": h["overtrading"], "chasing": h["chasing"], "long_shot_habit": h["long_shot_habit"],
            "flag": flag, "severe": False}


TOOLS = {"habits": check_habits, "break_even": check_break_even, "priced_in": check_priced_in,
         "long_shot": check_long_shot, "news_freshness": check_news_freshness, "m4_shift": check_m4_shift}


def next_check(s, side, h, done: dict):
    """The planner: the next check worth running given the results so far, with the reason; None to stop."""
    if "habits" not in done and h["decisions"] >= MIN_HISTORY:
        return "habits", f"you have made {h['decisions']} calls; look for habits before judging this one"
    if "break_even" not in done:
        return "break_even", "every trade starts with the price plus fee versus our estimate"
    if "priced_in" not in done and (abs(side["move"]) >= 0.005 or h["chasing"]):
        why = (f"the {side['team']} price moved {side['move'] * 100:+.1f} points since 24 hours before tip"
               if abs(side["move"]) >= 0.005 else "you often buy after the price has moved")
        return "priced_in", why
    if "long_shot" not in done and (side["ask"] <= 0.5 or h["long_shot_habit"]):
        why = (f"{side['team']} is the underdog at {side['ask']:.0%}" if side["ask"] <= 0.5
               else "you often buy cheap contracts")
        return "long_shot", why
    if "news_freshness" not in done and s["news"]:
        return "news_freshness", "there is injury news; how old is it relative to the price move?"
    if "m4_shift" not in done and abs(s["shift"]) >= 0.005:
        return "m4_shift", "the model moved after the news; does it move toward or away from your side?"
    return None


def _choose_lesson(results: dict, h: dict) -> str:
    flagged = [c for c, r in results.items() if r["flag"]]
    habit_match = [c for c in flagged if (c == "priced_in" and h["chasing"]) or (c == "long_shot" and h["long_shot_habit"])]
    if habit_match:
        return LESSON_FOR[habit_match[0]]
    if h["overtrading"] and results.get("break_even", {}).get("flag"):
        return "passing"
    for c in ("priced_in", "long_shot", "m4_shift", "break_even", "news_freshness", "habits"):
        if c in flagged:
            return LESSON_FOR[c]
    return "closing_line"


def _nudge(results: dict) -> str:
    severe = [c for c, r in results.items() if r["severe"]]
    flagged = [c for c, r in results.items() if r["flag"]]
    if "break_even" in severe or len(severe) >= 2:
        return "pass"
    return "caution" if flagged else "none"


def _reasons(team, results, h) -> list:
    out = []
    b = results.get("break_even")
    if b and b["severe"]:
        out.append(f"{team} costs {b['ask']:.0%} plus a {b['fee'] * 100:.1f}-cent fee, so it breaks even at "
                   f"{b['breakeven']:.1%}, above our estimate of {b['estimate']:.1%}.")
    elif b and b["flag"]:
        out.append(f"Our estimate for {team} is only {b['gap'] * 100:+.1f} points above the break-even after fees, "
                   f"below our {DEFAULT_MIN_EDGE:.0%} threshold.")
    p = results.get("priced_in")
    if p and p["flag"]:
        out.append(f"The {team} price already moved {p['move'] * 100:+.1f} points since 24 hours before tip, while "
                   f"the model's news shift for {team} is {p['news_shift'] * 100:+.1f} points: the news is probably "
                   f"priced in.")
    ls = results.get("long_shot")
    if ls and ls["flag"]:
        out.append(f"At {ls['ask']:.0%}, {team} is a long shot: it loses about {ls['loses_if_price_right']:.0%} of "
                   f"the time if the price is right.")
    n = results.get("news_freshness")
    if n and n["flag"]:
        out.append(f"The latest news is {n['latest_news_minutes_ago']:.0f} minutes old and the price has moved since.")
    m = results.get("m4_shift")
    if m and m["flag"]:
        out.append(f"The injury news moves our model {m['news_shift'] * 100:+.1f} points against {team}.")
    if h["overtrading"]:
        out.append(f"You traded {h['trade_rate']:.0%} of your decisions so far; most decisions have no edge after fees.")
    if h["chasing"] and not (p and p["flag"]):
        out.append(f"{h['chase_share']:.0%} of your trades came after the price had moved your way.")
    if h["long_shot_habit"] and not (ls and ls["flag"]):
        out.append(f"{h['long_shot_share']:.0%} of your trades were long shots.")
    return out


def _message(team, nudge, results, h, card) -> str:
    head = {"pass": f"Consider passing on {team}.", "caution": f"Be careful with {team}.",
            "none": f"No warning signs on {team} from these checks; it is still a paper trade and one result is "
                    f"mostly luck."}[nudge]
    title = card["title"] if card["title"].endswith("?") else card["title"] + "."
    return " ".join([head, *_reasons(team, results, h), f"Lesson: {title}", NOTICE])


def _card(s, lesson_id) -> dict:
    card = next((c for c in lessons(s) if c["id"] == lesson_id), None)
    if card is None:
        title, body = FALLBACK_CARDS[lesson_id]
        card = {"id": lesson_id, "title": title, "body": body}
    return card


def _numbers(results: dict) -> set:
    """Every percentage a writer may quote: tool outputs as whole percents and as points (x100)."""
    vals = {f"{DEFAULT_MIN_EDGE * 100:.0f}", "7", f"{LONG_SHOT * 100:.0f}", "100"}
    for r in results.values():
        for v in r.values():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                continue
            for x in (v, abs(v), 1 - v):
                vals |= {f"{x * 100:.0f}", f"{x * 100:.1f}"}
    return vals


def validate(text: str, results: dict) -> list:
    """Why an LLM message is unsafe: banned wording, encouragement, or a percentage not in the tool outputs."""
    problems = [f"banned:{w}" for w in banned_words([text])]
    problems += [f"encourages:{m.group(0)}" for m in [ENCOURAGE.search(text)] if m]
    allowed = _numbers(results)
    problems += [f"number:{p}%" for p in PERCENT.findall(text) if p not in allowed]
    return problems


def _cached_ask(prompt: str, ask) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"coach_{hashlib.sha256(prompt.encode()).hexdigest()[:20]}.json"
    if path.exists():
        return json.loads(path.read_text())["text"]
    text = ask(prompt)
    path.write_text(json.dumps({"prompt": prompt, "text": text}))
    return text


def _llm_text(team, nudge, results, card, ask):
    facts = {c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for c, r in results.items()}
    prompt = ("You are a cautious tutor in an educational paper-money simulation of NBA prediction markets. "
              f"A user wants to back {team}. Using ONLY the numbers in these check results, explain in 2-4 short "
              f"sentences what the checks found. The decision is '{nudge}' (pass = suggest passing, caution = "
              f"suggest care, none = no warning). Never encourage betting, never suggest a bigger stake, never "
              f"promise outcomes. End by naming the lesson '{card['title']}'.\n"
              f"Check results (prices and probabilities are fractions of 1): {json.dumps(facts, default=str)}")
    return _cached_ask(prompt, ask)


def advise(s: dict, team: str | None, history: list | None = None, llm: bool = False, ask=None) -> dict:
    """Plan and run the checks for a pending pick of `team` (None = pass) and return steps, lesson, nudge, message."""
    h = habits(history or [])
    if team is None:
        card = _card(s, "passing")
        return {"team": None, "steps": [], "results": {}, "habits": h, "lesson": card, "nudge": "none",
                "mode": "rules", "message": f"Passing costs nothing and pays no fee. {NOTICE}", "skipped": []}
    side = s["sides"].get(team)
    if side is None:
        card = _card(s, "passing")
        return {"team": team, "steps": [], "results": {}, "habits": h, "lesson": card, "nudge": "pass",
                "mode": "rules", "skipped": [],
                "message": f"Consider passing: there is no fresh price for {team} at this time. {NOTICE}"}
    steps, results = [], {}
    while (nxt := next_check(s, side, h, results)) is not None:
        check, why = nxt
        results[check] = r = TOOLS[check](s, side, h)
        steps.append({"step": len(steps) + 1, "check": check, "why": why, "flag": r["flag"], "severe": r["severe"],
                      "result": {k: v for k, v in r.items() if k not in ("flag", "severe")}})
    nudge = _nudge(results)
    card = _card(s, _choose_lesson(results, h))
    message, mode = _message(team, nudge, results, h, card), "rules"
    if llm:
        if ask is None:
            import llm as llm_mod
            ask = llm_mod.ask if llm_mod.available() else None
        if ask is not None:
            try:
                text = _llm_text(team, nudge, results, card, ask).strip()
                problems = validate(text, results)
            except Exception as exc:                        # network or quota: keep the rule-based message
                text, problems = "", [f"error:{type(exc).__name__}"]
            if text and not problems:
                message, mode = f"{text} {NOTICE}", "llm"
            else:
                mode = "rules (llm rejected: " + ", ".join(problems[:3]) + ")"
    skipped = [c for c in TOOLS if c not in results]
    return {"team": team, "steps": steps, "results": results, "habits": h, "lesson": card, "nudge": nudge,
            "mode": mode, "message": message, "skipped": skipped}


def trace_table(a: dict) -> pd.DataFrame:
    """The step trace as a small table for the app."""
    rows = [{"step": st["step"], "check": st["check"].replace("_", " "), "why": st["why"],
             "finding": "warning" if st["severe"] else "note" if st["flag"] else "ok"} for st in a["steps"]]
    return pd.DataFrame(rows, columns=["step", "check", "why", "finding"])
