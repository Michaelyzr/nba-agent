"""Every node the graph runs. Each takes the state dict and returns the fields it changes.

Chat-model steps (clarify, plan, write code, write note) use Gemini when a key is set and
fall back to offline rules otherwise, so the demo runs with no network.
"""
import json
import re
import traceback

import numpy as np
import pandas as pd
from langgraph.types import interrupt

from nba_agent.agents import llm
from nba_agent.data.tables import MAX_TRIES, SEASON, SKILLS, all_tables, load, player_name, team_name
from nba_agent.forecast import models
from nba_agent.policy import checks

SKILL_FILES = {"trade_split": "trade.md", "rate": "rate.md", "team_window": "team.md",
               "signing": "signing.md", "context": "context.md"}
METRICS = list(SKILL_FILES)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
OUT_OF_SCOPE = re.compile(r"\b(mvp|award|all-star|odds|bet|betting|spread|over/under)\b|"
                          r"will .*(get hurt|be injured|miss games)|predict .*injur", re.I)
RULES = [
    ("signing", re.compile(r"salary|sign\b|signing|contract|annual pay|offer", re.I)),
    ("context", re.compile(r"injury report|reports? say|said|news|quote|what did .* say", re.I)),
    ("team_window", re.compile(r"net rating", re.I)),
    ("rate", re.compile(r"per 100|per game|scoring rate|points per", re.I)),
    ("trade_split", re.compile(r"trade|traded|changed teams", re.I)),
]
NEEDS_TEAM = {"team_window"}
FAULTS = {
    "playoffs": "trade_split rate team_window", "wrong_team": "trade_split", "no_count": "trade_split rate team_window",
    "rate_mix": "rate", "invented_quote": "context", "wrong_paragraph": "context", "no_error": "signing",
    "note_number": "trade_split rate team_window signing context",
}


def log(step, detail):
    return [{"step": step, "detail": detail}]


def faulty(state, kind, tries_key="tries"):
    return state.get("plant") == kind and state.get(tries_key, 0) == 0 and state["form"]["metric"] in FAULTS[kind]


# ---------------- clarify ----------------

def find_subject(text):
    low = text.lower()
    players, teams = load("players"), load("teams")
    hits = [(len(n), "player", i, n) for i, n in zip(players.PLAYER_ID, players.PLAYER_NAME) if n.lower() in low]
    for i, n in zip(teams.TEAM_ID, teams.TEAM_NAME):
        nick = n.split(" ", 1)[1] if n.split()[0] not in ("LA", "Los", "New", "San", "Golden", "Oklahoma", "Portland") else n.split()[-1]
        if n.lower() in low or re.search(rf"\b{re.escape(nick.lower())}\b", low):
            hits.append((len(n), "team", i, n))
    if not hits:
        last = {n.split()[-1].lower(): (i, n) for i, n in zip(players.PLAYER_ID, players.PLAYER_NAME)}
        counts = players.PLAYER_NAME.str.split().str[-1].str.lower().value_counts()
        for tok in re.findall(r"[a-z]+", low):
            if tok in last and counts.get(tok) == 1:
                i, n = last[tok]
                return {"kind": "player", "id": int(i), "who": n}
        return None
    _, kind, i, n = max(hits, key=lambda h: (h[1] == "player", h[0]))
    return {"kind": kind, "id": int(i), "who": n}


def parse_season(text):
    if re.search(r"2025\s*[-–/]\s*(20)?26|this season|this year|current season", text, re.I):
        return SEASON
    other = re.search(r"(20\d\d)\s*[-–/]\s*(20)?(\d\d)", text)
    return f"{other.group(1)}-{other.group(3)}" if other else None


def parse_dates(text):
    low = text.lower()
    found = []
    for d, m, y in re.findall(r"(\d{1,2}) (" + "|".join(MONTHS) + r"),? (\d{4})", low):
        found.append(pd.Timestamp(int(y), MONTHS.index(m) + 1, int(d)))
    for m, d, y in re.findall(r"(" + "|".join(MONTHS) + r") (\d{1,2}),? (\d{4})", low):
        found.append(pd.Timestamp(int(y), MONTHS.index(m) + 1, int(d)))
    for iso in re.findall(r"\d{4}-\d{2}-\d{2}", low):
        found.append(pd.Timestamp(iso))
    out = {"split_date": None, "window_start": None, "window_end": None}
    if found:
        out["split_date"] = str(found[0].date())
        if "week of" in low:
            out["window_start"], out["window_end"] = str(found[0].date()), str((found[0] + pd.Timedelta(days=6)).date())
    else:
        month = re.search(r"\bin (" + "|".join(MONTHS) + r")\b", low)
        if month:
            m = MONTHS.index(month.group(1)) + 1
            start = pd.Timestamp(2025 if m >= 9 else 2026, m, 1)
            out["window_start"], out["window_end"] = str(start.date()), str((start + pd.offsets.MonthEnd(1)).date())
    return out


def parse_offline(question):
    if OUT_OF_SCOPE.search(question):
        metric = "out_of_scope"
    else:
        metric = next((name for name, rx in RULES if rx.search(question)), None)
    subject = find_subject(question)
    return {"who": subject["who"] if subject else None, "season": parse_season(question), "metric": metric,
            **parse_dates(question)}


def parse_with_llm(question):
    prompt = (
        "Fill this form from an NBA question. Return JSON with keys who, season, metric, split_date, "
        "window_start, window_end. who: the player or team name as written, or null. season: like "
        f"'2025-26'; 'this season' means {SEASON}; null if not stated. metric: one of {METRICS}, or "
        "'out_of_scope' if it asks to predict awards, injuries, or bets, or null if unclear. Dates as "
        "YYYY-MM-DD or null. Do not guess missing fields.\n\nQuestion: " + question
    )
    raw = llm.ask(prompt, want_json=True)
    subject = find_subject(raw.get("who") or "") if raw.get("who") else None
    return {**raw, "who": subject["who"] if subject else None, "season": parse_season(str(raw.get("season") or ""))}


FORM_KEYS = ("who", "season", "metric", "split_date", "window_start", "window_end")


def clarify(state):
    answer = state.get("answer")
    text = answer or state["question"]
    mode = "gemini" if llm.available() else "offline rules"
    try:
        form = parse_with_llm(text) if mode == "gemini" else parse_offline(text)
    except Exception as exc:
        form, mode = parse_offline(text), f"offline rules (Gemini failed: {exc.__class__.__name__})"
    if answer:
        prior = state.get("form") or {}
        for key in FORM_KEYS:
            keep_prior = prior.get(key) and prior.get(key) != "out_of_scope"
            if form.get(key) in (None, "") and keep_prior:
                form[key] = prior[key]
    subject = find_subject(form["who"]) if form.get("who") else None
    if subject:
        form.update(subject)

    missing, reasons = [], []
    if form.get("metric") == "out_of_scope":
        missing.append("metric")
        reasons.append("This agent does not predict awards, injuries, or betting lines.")
    elif form.get("metric") not in METRICS:
        missing.append("metric")
    if not subject:
        missing.append("who")
    elif form.get("metric") in NEEDS_TEAM and subject["kind"] != "team":
        missing.append("who")
        reasons.append("A net-rating window needs a team.")
    elif form.get("metric") in METRICS and form["metric"] not in NEEDS_TEAM and subject["kind"] != "player":
        missing.append("who")
        reasons.append("That question type needs a player.")
    if form.get("season") != SEASON:
        missing.append("season")
        if form.get("season"):
            reasons.append(f"Only {SEASON} is saved.")
    if form.get("metric") == "team_window" and not form.get("split_date"):
        missing.append("date")
    if form.get("metric") == "trade_split" and subject and subject["kind"] == "player":
        if subject["id"] not in set(load("trades").PLAYER_ID):
            missing.append("metric")
            reasons.append(f"{subject['who']} has no saved trade in {SEASON}.")

    followup = None
    if missing:
        asks = {"who": "which player or team", "season": f"which season (only {SEASON} is saved)",
                "metric": "what you want: a trade split, a per-game vs per-100 rate, a team net-rating window, "
                          "a signing-pay estimate, or what a saved report said",
                "date": "the date to split the team's season on"}
        followup = " ".join(reasons + ["Please tell me " + "; and ".join(asks[m] for m in dict.fromkeys(missing)) + "."])
    detail = f"[{mode}] form = {json.dumps({k: form.get(k) for k in FORM_KEYS})}"
    if missing:
        detail += f" | missing: {', '.join(dict.fromkeys(missing))}"
    return {"form": form, "missing": list(dict.fromkeys(missing)), "followup": followup, "mode": mode,
            "answer": None, "trace": log("clarify", detail)}


def wait(state):
    answer = interrupt({"followup": state["followup"]})
    return {"question": f"{state['question']} {answer}", "answer": answer,
            "trace": log("wait", f"user answered: {answer}")}


# ---------------- plan ----------------

def plan(state):
    form = state["form"]
    skill = SKILL_FILES[form["metric"]]
    skill_text = (SKILLS / skill).read_text()
    params = {"player_id" if form["kind"] == "player" else "team_id": form["id"]}
    if form["metric"] == "trade_split":
        t = load("trades").set_index("PLAYER_ID").loc[form["id"]]
        params.update(trade_date=str(t.TRADE_DATE.date()), from_team_id=int(t.FROM_TEAM_ID), to_team_id=int(t.TO_TEAM_ID))
    elif form["metric"] == "team_window":
        params["split_date"] = form["split_date"]
    elif form["metric"] in ("rate", "context"):
        params.update(start=form.get("window_start"), end=form.get("window_end"))

    if state.get("mode") == "gemini":
        try:
            text = llm.ask(f"{skill_text}\n\nForm: {json.dumps(form, default=str)}\nParameters: {json.dumps(params)}\n"
                           "Write a numbered plan of 3-5 short steps for this question. No code.")
        except Exception as exc:
            text = f"(Gemini failed: {exc.__class__.__name__}) Follow {skill} with {params}."
    else:
        steps = [line for line in skill_text.splitlines() if re.match(r"\d\.", line)]
        text = "\n".join(steps)
    return {"plan": {"skill": skill, "params": params, "text": text},
            "trace": log("plan", f"opened skills/{skill} with {json.dumps(params)}")}


# ---------------- stat branch ----------------

def rank_recaps(state):
    form, params = state["form"], state["plan"]["params"]
    recaps = load("recaps")
    if form["kind"] == "player":
        pool = recaps[recaps.PLAYER_ID == form["id"]]
    else:
        pool = recaps[recaps.TEAM_ID == form["id"]]
        if params.get("split_date"):
            d = pd.Timestamp(params["split_date"])
            pool = pool[(pool.GAME_DATE >= d - pd.Timedelta(days=7)) & (pool.GAME_DATE <= d + pd.Timedelta(days=7))]
    texts = pool.TEXT.tolist()
    idx, scores, source = models.rank(state["question"], texts)
    ranked = [{"doc_id": pool.RECAP_ID.iloc[i], "text": texts[i], "score": round(s, 3)} for i, s in zip(idx, scores)]
    return {"ranked": ranked, "trace": log("rank_recaps", f"{source}: kept {len(ranked)} of {len(texts)} recap sentences")}


TEMPLATES = {
    "trade_split": '''
reg = tables["player_regular"]
pid, cut = {player_id}, pd.Timestamp("{trade_date}")
rows = reg[reg.PLAYER_ID == pid]
def ts(f):
    return float(f.PTS.sum() / (2 * (f.FGA.sum() + 0.44 * f.FTA.sum())))
windows = []
for label, team, part in (("before", {from_team_id}, rows[rows.GAME_DATE < cut]),
                          ("after", {to_team_id}, rows[rows.GAME_DATE >= cut])):
    part = part[part.TEAM_ID == team]
    windows.append({{"label": label, "team_id": team, "game_ids": part.GAME_ID.tolist(), "games": len(part),
                    "start": str(part.GAME_DATE.min().date()), "end": str(part.GAME_DATE.max().date()),
                    "values": {{"ts_pct": ts(part)}}}})
result = {{"metric": "trade_split", "season_type": "Regular Season", "subject": {{"kind": "player", "id": pid}},
          "windows": windows, "units": {{"ts_pct": "true shooting (ratio of season totals)"}}}}
''',
    "rate": '''
reg, adv = tables["player_regular"], tables["advanced_regular"]
pid = {player_id}
rows = reg[reg.PLAYER_ID == pid]
if {start!r}:
    rows = rows[(rows.GAME_DATE >= pd.Timestamp({start!r})) & (rows.GAME_DATE <= pd.Timestamp({end!r}))]
poss = adv[(adv.PLAYER_ID == pid) & adv.GAME_ID.isin(rows.GAME_ID)].POSS.sum()
window = {{"label": "window", "game_ids": rows.GAME_ID.tolist(), "games": len(rows),
          "start": str(rows.GAME_DATE.min().date()), "end": str(rows.GAME_DATE.max().date()),
          "values": {{"pts_per_game": float(rows.PTS.sum() / len(rows)), "pts_per_100": float(rows.PTS.sum() / poss * 100)}}}}
result = {{"metric": "rate", "season_type": "Regular Season", "subject": {{"kind": "player", "id": pid}},
          "windows": [window], "units": {{"pts_per_game": "points per game", "pts_per_100": "points per 100 possessions"}}}}
''',
    "team_window": '''
tr = tables["team_regular"]
tid, cut = {team_id}, pd.Timestamp("{split_date}")
rows = tr[tr.TEAM_ID == tid]
windows = []
for label, part in (("before", rows[rows.GAME_DATE < cut]), ("after", rows[rows.GAME_DATE >= cut])):
    net = float((part.PTS.sum() - part.OPP_PTS.sum()) / part.POSS.sum() * 100)
    windows.append({{"label": label, "game_ids": part.GAME_ID.tolist(), "games": len(part),
                    "start": str(part.GAME_DATE.min().date()), "end": str(part.GAME_DATE.max().date()),
                    "values": {{"net_rating": net}}}})
result = {{"metric": "team_window", "season_type": "Regular Season", "subject": {{"kind": "team", "id": tid}},
          "windows": windows, "units": {{"net_rating": "net points per 100 possessions"}}}}
''',
    "context": '''
p, reg = tables["paragraphs"], tables["player_regular"]
pid = {player_id}
sel = p[p.PLAYER_ID == pid]
games = reg[reg.PLAYER_ID == pid]
if {start!r}:
    lo, hi = pd.Timestamp({start!r}), pd.Timestamp({end!r})
    sel = sel[(sel.DATE >= lo) & (sel.DATE <= hi)]
    games = games[(games.GAME_DATE >= lo) & (games.GAME_DATE <= hi)]
result = {{"metric": "context", "subject": {{"kind": "player", "id": pid}},
          "window": {{"start": {start!r}, "end": {end!r}}},
          "candidates": sel.to_dict("records"), "box_facts": {{"games_in_window": int(games.GAME_ID.nunique())}}}}
''',
}

SCHEMA = """Tables (pandas DataFrames in a dict called `tables`):
player_regular: GAME_ID, GAME_DATE (datetime), PLAYER_ID, TEAM_ID, MIN, FGM, FGA, FG3M, FG3A, FTM, FTA, PTS
advanced_regular: GAME_ID, PLAYER_ID, TEAM_ID, TS_PCT, POSS, USG_PCT
team_regular: GAME_ID, GAME_DATE, TEAM_ID, PTS, OPP_PTS, POSS
paragraphs: PARA_ID, DATE, PLAYER_ID, KIND, SOURCE, URL, TEXT
trades: PLAYER_ID, TRADE_DATE, FROM_TEAM_ID, TO_TEAM_ID
player_playoffs, team_playoffs: playoff games, never for a regular-season question."""


def write_code(state):
    metric, params = state["form"]["metric"], state["plan"]["params"]
    template = TEMPLATES[metric].format(**{"start": None, "end": None, **params}).strip()
    step = "write_filter" if metric == "context" else "write_pandas"
    if state.get("mode") != "gemini":
        note = f"template for {metric}" + (" (retry after: " + ", ".join(f['check'] for f in state["failures"]) + ")" if state.get("failures") else "")
        return {"code": template, "trace": log(step, note)}
    feedback = "\n".join(f"- {f['check']}: {f['detail']}" for f in state.get("failures", []))
    prompt = (f"{SCHEMA}\n\nSkill:\n{(SKILLS / state['plan']['skill']).read_text()}\nPlan:\n{state['plan']['text']}\n"
              f"Parameters: {json.dumps(params)}\n\nWrite Python that assigns a dict to `result` with exactly the "
              f"structure of this reference (same keys):\n{template}\n"
              + (f"\nThe previous attempt was rejected by checks.py:\n{feedback}\nFix those problems.\n" if feedback else "")
              + "Use only `pd`, `np`, and `tables`. Return only code.")
    try:
        code = llm.strip_fences(llm.ask(prompt))
        return {"code": code, "trace": log(step, "Gemini wrote the code" + (" after feedback" if feedback else ""))}
    except Exception as exc:
        return {"code": template, "trace": log(step, f"Gemini failed ({exc.__class__.__name__}); used template")}


def inject(state, result):
    notes = []
    windows = result.get("windows", [])
    if faulty(state, "playoffs") and windows:
        pid = result["subject"]["id"]
        if result["subject"]["kind"] == "player":
            ids = load("player_playoffs").loc[lambda f: f.PLAYER_ID == pid, "GAME_ID"].unique().tolist()[:3]
        else:
            ids = load("team_playoffs").loc[lambda f: f.TEAM_ID == pid, "GAME_ID"].unique().tolist()[:3]
        ids = ids or load("team_playoffs").GAME_ID.unique().tolist()[:3]
        windows[-1]["game_ids"] += ids
        windows[-1]["games"] += len(ids)
        notes.append(f"added {len(ids)} playoff games")
    if faulty(state, "wrong_team") and len(windows) == 2:
        windows[0]["team_id"] = windows[1]["team_id"]
        notes.append("charged the before-trade games to the new team")
    if faulty(state, "no_count"):
        for w in windows:
            w.pop("games", None)
        notes.append("dropped the game counts")
    if faulty(state, "rate_mix"):
        result["comparison"] = {"a": "pts_per_game", "b": "pts_per_100"}
        notes.append("compared per-game with per-100 directly")
    return notes


def run_code(state):
    metric = state["form"]["metric"]
    step = "run_filter" if metric == "context" else "run_pandas"
    scope = {"pd": pd, "np": np, "tables": all_tables()}
    try:
        exec(state["code"], scope)
        result = scope.get("result")
        if not isinstance(result, dict):
            result = {"error": "code did not assign a dict to `result`"}
    except Exception:
        result = {"error": traceback.format_exc().strip().splitlines()[-1]}
    notes = inject(state, result) if "error" not in result else []
    detail = result["error"] if "error" in result else "ran on the saved season"
    trace = log(step, detail)
    if notes:
        trace += log("fault", "injected for the demo: " + "; ".join(notes))
    return {"result": result, "trace": trace}


# ---------------- context branch ----------------

def rank_paragraphs(state):
    result = dict(state["result"])
    cands = result.pop("candidates", [])
    texts = [c["TEXT"] for c in cands]
    idx, scores, source = models.rank(state["question"], texts)
    kept = []
    for i, s in zip(idx, scores):
        c = cands[i]
        kept.append({"para_id": c["PARA_ID"], "date": str(pd.Timestamp(c["DATE"]).date()), "source": c["SOURCE"],
                     "url": c["URL"], "text": c["TEXT"], "score": round(s, 3)})
    quotes = [{"para_id": k["para_id"], "text": re.split(r"(?<=\.)\s", k["text"])[0]} for k in kept]
    result.update(paragraphs=kept, quotes=quotes)
    trace = log("rank_paragraphs", f"{source}: kept {len(kept)} of {len(texts)} saved paragraphs")
    if faulty(state, "invented_quote") and quotes:
        quotes[0]["text"] += " He expects to be back next week."
        trace += log("fault", "injected for the demo: added an invented sentence to a quote")
    if faulty(state, "wrong_paragraph"):
        other = load("paragraphs").loc[lambda f: f.PLAYER_ID != result["subject"]["id"]].iloc[0]
        kept.append({"para_id": other.PARA_ID, "date": str(other.DATE.date()), "source": other.SOURCE,
                     "url": other.URL, "text": other.TEXT, "score": 0.0})
        trace += log("fault", "injected for the demo: kept a paragraph about another player")
    return {"result": result, "ranked": kept, "trace": trace}


# ---------------- signing branch ----------------

def build_features(state):
    pid = state["form"]["id"]
    reg, adv = load("player_regular"), load("advanced_regular")
    rows = reg[reg.PLAYER_ID == pid]
    if rows.empty:
        return {"features": None, "trace": log("build_features", "no regular-season games")}
    contracts = load("contracts")
    prior = contracts.loc[contracts.PLAYER_ID == pid, "PRIOR_PAY"]
    prior = float(prior.iloc[0]) if len(prior) and pd.notna(prior.iloc[0]) else None
    feats = {
        "GP": int(rows.GAME_ID.nunique()), "MPG": float(rows.MIN.mean()), "PPG": float(rows.PTS.mean()),
        "TS": float(rows.PTS.sum() / (2 * (rows.FGA.sum() + 0.44 * rows.FTA.sum()))),
        "USG": float(adv[adv.PLAYER_ID == pid].USG_PCT.mean()),
        "AGE": int(load("players").set_index("PLAYER_ID").loc[pid, "AGE"]),
        "PRIOR_PAY": prior or 0.0, "HAS_PRIOR": int(prior is not None),
    }
    return {"features": feats, "trace": log("build_features", json.dumps({k: round(v, 3) for k, v in feats.items()}))}


def predict_pay(state):
    feats = state.get("features")
    base = {"metric": "signing", "subject": {"kind": "player", "id": state["form"]["id"]}, "features": feats}
    if not feats:
        return {"result": {**base, "prediction": None}, "trace": log("predict_pay", "skipped: no features")}
    net = models.signing_net()
    result = {**base, "prediction": net.predict(feats), "label": "prediction", "heldout_mae": net.heldout_mae,
              "model": "one-hidden-layer network (models/signing)"}
    trace = log("predict_pay", f"predicted ${result['prediction'] / 1e6:.2f}M, held-out MAE ${net.heldout_mae / 1e6:.2f}M")
    if faulty(state, "no_error"):
        result.pop("heldout_mae")
        trace += log("fault", "injected for the demo: dropped the held-out error")
    return {"result": result, "trace": trace}


# ---------------- check ----------------

def check(state):
    failures = checks.check_result(state["form"], state["result"])
    if not failures:
        return {"failures": [], "trace": log("checks.py", "passed")}
    tries = state.get("tries", 0) + 1
    detail = f"rejected (try {tries} of {MAX_TRIES}): " + "; ".join(f"{f['check']}: {f['detail']}" for f in failures)
    return {"failures": failures, "tries": tries, "trace": log("checks.py", detail)}


# ---------------- note ----------------

def fmt_money(v):
    return f"${v / 1e6:.2f}M"


def template_note(form, result):
    name = form["who"]
    m = form["metric"]
    if m == "trade_split":
        b, a = result["windows"]
        return (f"{name}'s true shooting was {b['values']['ts_pct']:.3f} over {b['games']} games with the "
                f"{team_name(b['team_id'])} before the trade [W:before]. It was {a['values']['ts_pct']:.3f} over "
                f"{a['games']} games with the {team_name(a['team_id'])} after it [W:after]. Both figures are "
                f"regular-season totals, and the split date comes from the saved transactions file.")
    if m == "rate":
        w = result["windows"][0]
        v = w["values"]
        return (f"Over {w['games']} regular-season games from {w['start']} to {w['end']}, {name} scored "
                f"{v['pts_per_game']:.1f} points per game [W:window]. Over the same games he scored "
                f"{v['pts_per_100']:.1f} points per 100 possessions [W:window]. These are two different "
                f"quantities and should not be compared directly.")
    if m == "team_window":
        b, a = result["windows"]
        return (f"The {name} had a net rating of {b['values']['net_rating']:+.1f} over {b['games']} regular-season "
                f"games before {form['split_date']} [W:before]. Their net rating was {a['values']['net_rating']:+.1f} "
                f"over {a['games']} games from that date on [W:after].")
    if m == "signing":
        f = result["features"]
        return (f"Predicted annual pay to sign {name}: {fmt_money(result['prediction'])} [M:signing]. This is a "
                f"model prediction, with a mean absolute error of {fmt_money(result['heldout_mae'])} on contracts "
                f"held out of training [M:signing]. Inputs: {f['GP']} games, {f['MPG']:.1f} minutes and "
                f"{f['PPG']:.1f} points per game, true shooting {f['TS']:.3f}, age {f['AGE']} [M:signing]. "
                f"It is not a statement about the team's cap position.")
    if m == "context":
        if not result["quotes"]:
            return f"No saved report about {name} matches that window."
        lines = [f"Saved reports about {name}:"]
        for q in result["quotes"][:3]:
            p = next(p for p in result["paragraphs"] if p["para_id"] == q["para_id"])
            lines.append(f"On {pd.Timestamp(p['date']).strftime('%-d %B %Y')}, {p['source']} wrote: \"{q['text']}\" [P:{p['para_id']}].")
        games = result.get("box_facts", {}).get("games_in_window")
        if games is not None and result.get("window", {}).get("start"):
            lines.append(f"He played {games} regular-season games in that window, per the box score [W:box].")
        return " ".join(lines)
    raise KeyError(m)


def sources_block(form, result, ranked):
    lines = ["", "Sources"]
    for w in result.get("windows", []):
        ids = w["game_ids"]
        shown = ", ".join(ids[:6]) + (f", ... ({len(ids)} games)" if len(ids) > 6 else "")
        lines.append(f"[W:{w['label']}] {w.get('start')} to {w.get('end')}: {shown}")
    for p in result.get("paragraphs", []):
        lines.append(f"[P:{p['para_id']}] {p['source']}, {p['date']}, {p['url']}")
    if form["metric"] == "signing":
        lines.append(f"[M:signing] {result['model']}; held-out MAE {fmt_money(result['heldout_mae'])}")
    if ranked and form["metric"] != "context":
        lines.append("Related recaps (ranker top 5):")
        lines += [f"  - {r['text']}" for r in ranked]
    lines.append("Data: frozen 2025-26 season files (NBA.com via nba_api in the real build; synthetic sample here).")
    return "\n".join(lines)


def compact(result):
    r = json.loads(json.dumps(result, default=str))
    for w in r.get("windows", []):
        w["game_ids"] = f"{len(w['game_ids'])} ids"
    r.pop("candidates", None)
    return r


def write_note(state):
    form, result = state["form"], state["result"]
    if form["metric"] == "context" and result.get("window", {}).get("start"):
        result = {**result, "windows": [{"label": "box", "game_ids": [], "games": result["box_facts"]["games_in_window"],
                                         "start": result["window"]["start"], "end": result["window"]["end"], "values": {}}]}
    body, how = None, "template"
    if state.get("mode") == "gemini":
        feedback = "\n".join(f"- {f['check']}: {f['detail']}" for f in state.get("note_failures", []))
        prompt = (
            f"Write a short note (2-5 sentences) answering: {state['question']}\nUse ONLY this checked result:\n"
            f"{json.dumps(compact(result), indent=1)}\nRules: every sentence with a number ends with a citation tag "
            "[W:<window label>] for stat windows, [P:<para_id>] for paragraphs, [M:signing] for signing. Numbers must "
            "come from the result (ratios to 3 decimals, rates to 1 decimal, money as $X.XXM). Quote paragraphs only "
            "using the exact `quotes` texts inside double quotes. A signing figure must be called a prediction and must "
            "state heldout_mae. Never mention cap room or recommend a move. Never present per-game and per-100 as the "
            "same quantity." + (f"\nYour previous note was rejected:\n{feedback}" if feedback else "")
        )
        try:
            body, how = llm.ask(prompt), "Gemini"
        except Exception as exc:
            how = f"template (Gemini failed: {exc.__class__.__name__})"
    if body is None:
        body = template_note(form, result)
    trace = log("write_note", how)
    if faulty(state, "note_number", "note_tries"):
        nums = re.findall(r"\d+\.\d+", body)
        if nums:
            body = body.replace(nums[0], f"{float(nums[0]) * 1.2:.{len(nums[0].split('.')[1])}f}", 1)
            trace += log("fault", f"injected for the demo: changed {nums[0]} in the note")
    note = body.strip() + "\n" + sources_block(form, result, state.get("ranked"))
    return {"note": note, "note_result": result, "trace": trace}


def evidence_for(result):
    return models.evidence_text(checks.allowed_numbers(result))


def support_check(state):
    form, result, note = state["form"], state["note_result"], state["note"]
    failures = checks.check_note(form, result, note)
    body = note.split("\nSources")[0]
    evidence = evidence_for(result)
    source = None
    for s in re.split(r"(?<=[.!?])\s+", body):
        claim = checks.strip_for_numbers(s)
        if not checks.NUMBER.search(claim):
            continue
        ok, score, source = models.support(claim, evidence)
        if not ok:
            failures.append({"check": "unsupported_sentence", "detail": f"support model rejected: '{s.strip()[:70]}'"})
    if not failures:
        return {"note_failures": [], "status": "answered",
                "trace": log("support_check", f"{source or 'no numeric sentences'}: every sentence supported")}
    tries = state.get("note_tries", 0) + 1
    detail = f"rejected (try {tries} of {MAX_TRIES}): " + "; ".join(f"{f['check']}: {f['detail']}" for f in failures)
    return {"note_failures": failures, "note_tries": tries, "trace": log("support_check", detail)}


def unresolved(state):
    if state.get("note_tries", 0) >= MAX_TRIES:
        step, fails = "the note", state.get("note_failures", [])
    else:
        step, fails = "the calculation", state.get("failures", [])
    reasons = "; ".join(f"{f['check']}: {f['detail']}" for f in fails)
    note = f"Unresolved: {step} failed the checks {MAX_TRIES} times, so no figure is given. Last failures: {reasons}"
    return {"note": note, "status": "unresolved", "trace": log("unresolved", step)}
