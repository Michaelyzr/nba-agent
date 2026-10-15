"""Truncation check on the original approve/reject sceptic: how many of its vetoes were fail-closed, not judgements.

The original sceptic (agents/tool_agent.py _sceptic) turns an API error, an empty or unparseable reply, a verdict
other than approve/reject, or banned wording into {"verdict": "reject", "invalid": True}. deepseek-reasoner counts
reasoning against max_tokens (REASONER_MAX_TOKENS), so a reply that spends the whole budget thinking comes back
empty and is vetoed. Every review in both walk-forward windows (Dec 2025 - Jan 2026, runs/llm_agent/prev-tool_sceptic;
Feb - Apr 2026 test, the finished full-sample runs used by llm_sceptic_full) is classified from the trace and the
cached reply in runs/llm_cache/ (errors are never cached, so an invalid review without a cache entry was an API
error or timeout):

    approve / reject                      parsed JSON verdict, not invalid
    fail-closed: truncated                cached, tokens_out >= TRUNC_SHARE x max_tokens, no valid verdict
    fail-closed: API error or timeout     invalid and no cache entry
    fail-closed: parse error              cached, below the cap, no valid verdict
    fail-closed: banned wording           reason says so

Counterfactual fill, CLV and P&L per proposal come from Replay._fill / Replay._settle at the decision time
($20 stake, fill at the ask, CLV = mid at tip minus the price paid), as in evaluation/sceptic_balanced.py.
No API calls.

    python -m evaluation.sceptic_truncation
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.llm_agent_eval import OUT, RESULTS, WINDOWS, load_frozen, load_traces
from evaluation.llm_sceptic_full import from_runs
from evaluation.stats import REPS, SEED

STEM = "sceptic_truncation"
STAKE = 20.0
MIN_EDGE = 0.04
REASONER = "deepseek-reasoner"
TRUNC_SHARE = 0.98
REASK_MIN = 10
WINDOW_NAMES = {"prev": "Dec 2025 - Jan 2026", "test": "Feb - Apr 2026 (test)", "pooled": "Pooled"}
HEAD = re.compile(r"Game: (\w+) \(away\) at (\w+) \(home\), game date (\S+)\. Tip-off in (\d+) minutes")
CATS = ["approve", "reject", "fail: truncated", "fail: API error/timeout", "fail: parse error", "fail: banned wording"]


def reviewed_traces(games: pd.DataFrame) -> list:
    prev = load_traces(OUT / "prev-tool_sceptic" / "tool_sceptic")
    dates = games.date.astype(str)
    days = sorted(dates[(dates >= WINDOWS["test"][0]) & (dates <= WINDOWS["test"][1])].unique())
    test = [t for p in from_runs(games, days) for t in p["traces"]]
    return [("prev", t) for t in prev if "sceptic" in t] + [("test", t) for t in test if "sceptic" in t]


def cached_sceptic_entries() -> dict:
    """(date, away, home, minutes to tip, side, price, estimate_p) -> cached reasoner sceptic entries."""
    from agents.llm_client import CACHE_DIR
    index = {}
    for p in CACHE_DIR.glob("*/*.json"):
        d = json.loads(p.read_text())
        if d.get("role") != "sceptic" or d.get("model") != REASONER:
            continue
        m = HEAD.search(d["prompt"])
        lines = d["prompt"].split("\n")
        e = json.loads(lines[lines.index("EVIDENCE") + 1])["code_computed_edge"]
        if abs(e.get("min_edge", 0) - MIN_EDGE) > 1e-9:
            continue
        key = (m.group(3), m.group(1), m.group(2), int(m.group(4)), e["side"], round(e["price"], 4),
               round(float(e["estimate_p"]), 4))
        index.setdefault(key, []).append(d)
    return index


def classify(sceptic: dict, hit: dict | None) -> str:
    from agents.llm_client import REASONER_MAX_TOKENS, parse_json
    if not sceptic.get("invalid"):
        return sceptic["verdict"]
    if "banned" in sceptic.get("reason", ""):
        return "fail: banned wording"
    if hit is None:
        return "fail: API error/timeout"
    verdict = str((parse_json(hit["text"]) or {}).get("verdict", "")).lower()
    assert verdict not in ("approve", "reject"), "invalid trace but a valid cached verdict"
    return "fail: truncated" if hit["tokens_out"] >= TRUNC_SHARE * REASONER_MAX_TOKENS else "fail: parse error"


def review_table(tables: dict) -> pd.DataFrame:
    from replay import Order, Replay
    games = tables["games"]
    g = games.set_index("game_id")
    index = cached_sceptic_entries()
    rp = Replay(tables, lambda *a: [])
    rows = []
    for window, t in reviewed_traces(games):
        gid, now = t["game_id"], pd.Timestamp(t["as_of"])
        tip = g.tip_time[gid]
        cand = t.get("candidate") or t.get("order")
        est = round(float(t["proposal"]["estimate_p"]), 4)
        key = (str(g.date[gid]), g.away_team[gid], g.home_team[gid], int(round((tip - now).total_seconds() / 60)),
               cand["side"], round(cand["price"], 4), est)
        hits = index.get(key, [])
        if len({h["prompt"] for h in hits}) > 1:
            raise SystemExit(f"{len(hits)} distinct cached sceptic prompts for {key}")
        hit = hits[0] if hits else None
        fill, why = rp._fill(Order(cand["ticker"], cand["side"], 0.0, STAKE, "counterfactual"), now, tip)
        if fill is None:
            raise SystemExit(f"no counterfactual fill for {key}: {why}")
        s = rp._settle({**fill, "market_ticker": cand["ticker"], "side": cand["side"], "game_id": gid}, tip)
        rows.append({"window": window, "date": str(g.date[gid]), "game_id": gid, "as_of": str(now),
                     "game": f"{g.away_team[gid]} @ {g.home_team[gid]}", "ticker": cand["ticker"],
                     "side": cand["side"], "category": classify(t["sceptic"], hit),
                     "trace_reason": str(t["sceptic"].get("reason", ""))[:160],
                     "cached": hit is not None, "tokens_out": hit["tokens_out"] if hit else np.nan,
                     "reasoning_tokens": hit.get("reasoning_tokens", np.nan) if hit else np.nan,
                     "answer_chars": len(hit["text"]) if hit else np.nan,
                     "price": s["price"], "contracts": s["contracts"], "fee": s["fee"], "clv": s["clv"],
                     "clv_usd": s["clv"] * s["contracts"], "pnl": s["pnl"]})
    return pd.DataFrame(rows)


def counts(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for w, sub in [("prev", df[df.window == "prev"]), ("test", df[df.window == "test"]), ("pooled", df)]:
        c = sub.category.value_counts()
        r = {"window": WINDOW_NAMES[w], "reviewed": len(sub), **{k: int(c.get(k, 0)) for k in CATS}}
        r["vetoed"] = len(sub) - r["approve"]
        r["fail_closed"] = r["vetoed"] - r["reject"]
        out.append(r)
    return pd.DataFrame(out)


def mean_ci(sub: pd.DataFrame, reps=REPS, seed=SEED) -> tuple:
    """Mean CLV per contract (cents) over proposals, 95% CI from a day-clustered bootstrap."""
    if len(sub) < 2:
        return 100 * sub.clv.mean() if len(sub) else np.nan, np.nan, np.nan
    days = sub.date.to_numpy()
    uniq = np.unique(days)
    by_day = {d: sub.clv.to_numpy()[days == d] for d in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws.append(np.concatenate([by_day[d] for d in pick]).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return 100 * sub.clv.mean(), 100 * lo, 100 * hi


def clv_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = {"approve": df.category == "approve", "reject (genuine)": df.category == "reject",
              "fail-closed (all)": df.category.str.startswith("fail"),
              "all vetoes (genuine + fail-closed)": df.category != "approve"}
    out = []
    for w in ("prev", "test", "pooled"):
        wmask = df.window == w if w != "pooled" else pd.Series(True, index=df.index)
        for name, m in groups.items():
            sub = df[wmask & m]
            mean, lo, hi = mean_ci(sub)
            out.append({"window": WINDOW_NAMES[w], "group": name, "n": len(sub), "mean_clv_c": mean,
                        "ci_lo": lo, "ci_hi": hi, "clv_usd": sub.clv_usd.sum(), "pnl": sub.pnl.sum()})
    return pd.DataFrame(out)


def write_report(df: pd.DataFrame, cnt: pd.DataFrame, clv: pd.DataFrame) -> str:
    from agents.llm_client import REASONER_MAX_TOKENS
    f = lambda v: "–" if pd.isna(v) else f"{v:+.2f}"
    md = ["# Truncation check on the original sceptic", "",
          "Every review by the original approve/reject deepseek-reasoner sceptic (fail closed) in both walk-forward "
          f"windows, classified from the trace and the cached reply. max_tokens = {REASONER_MAX_TOKENS} "
          f"(REASONER_MAX_TOKENS); truncated = no valid verdict with tokens_out >= {TRUNC_SHARE:.0%} of the cap. "
          "API errors are never cached, so an invalid review without a cache entry is counted as an API error "
          "or timeout. No API calls.", "",
          "## Classification", "",
          "| Window | Reviewed | Approve | Reject (genuine) | Fail: truncated | Fail: API error/timeout "
          "| Fail: parse error | Fail: banned wording | Vetoed | of which fail-closed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in cnt.to_dict("records"):
        md.append(f"| {d['window']} | {d['reviewed']} | " + " | ".join(str(d[k]) for k in CATS)
                  + f" | {d['vetoed']} | {d['fail_closed']} |")
    md += ["", "Corrected veto statements:", ""]
    for d in cnt.to_dict("records"):
        md.append(f"- {d['window']}: of {d['vetoed']} vetoes, {d['reject']} were genuine rejects and "
                  f"{d['fail_closed']} were fail-closed ({d['fail: truncated']} truncated, "
                  f"{d['fail: API error/timeout']} API error/timeout, {d['fail: parse error']} parse error, "
                  f"{d['fail: banned wording']} banned wording).")
    fails = df[df.category.str.startswith("fail")]
    if len(fails):
        md += ["", "Fail-closed reviews:", "", "| Window | Date | Game | Side | Category | tokens_out | answer chars "
               "| CLV (c) | P&L $ |", "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {r.window} | {r.date} | {r.game} | {r.ticker.split('-')[-1]} {r.side} | {r.category} | "
               f"{'' if pd.isna(r.tokens_out) else int(r.tokens_out)} | "
               f"{'' if pd.isna(r.answer_chars) else int(r.answer_chars)} | {100 * r.clv:+.1f} | {r.pnl:+.2f} |"
               for r in fails.itertuples()]
    md += ["", "## Mean CLV per contract by category", "",
           "Counterfactual $20 fills at the decision time; mean over proposals in cents per contract with a "
           f"day-clustered bootstrap 95% CI ({REPS} draws); CLV $ and settled P&L are sums.", "",
           "| Window | Group | n | Mean CLV (c) [95% CI] | CLV $ | P&L $ |", "|---|---|---|---|---|---|"]
    md += [f"| {r.window} | {r.group} | {r.n} | {f(r.mean_clv_c)} [{f(r.ci_lo)}, {f(r.ci_hi)}] | "
           f"{f(r.clv_usd)} | {f(r.pnl)} |" for r in clv.itertuples()]
    bal = RESULTS / "sceptic_balanced_proposals.csv"
    if bal.exists():
        b = pd.read_csv(bal, dtype={"game_id": str}).merge(df.astype({"game_id": str}),
                                                           on=["game_id", "as_of", "ticker"])
        both = int((b.truncated_at_8k & (b.category == "fail: truncated")).sum())
        md += ["", f"The {int(b.truncated_at_8k.sum())} truncations reported by Approach A (sceptic_balanced) were "
               f"replies to the balanced prompt; the original review was also truncated for {both} of them. The "
               "original prompt rarely exhausted the budget."]
    y = int(df.category.str.startswith("fail").sum())
    md += ["", "## Re-ask of truncated reviews", "",
           (f"Not run: {y} fail-closed reviews in total, below the threshold of {REASK_MIN}; the fail-closed "
            "share is too small to change the veto statistics." if y < REASK_MIN else
            f"{y} fail-closed reviews: eligible for a re-ask with a larger token cap.")]
    return "\n".join(md) + "\n"


def main():
    df = review_table(load_frozen())
    cnt = counts(df)
    clv = clv_table(df)
    cnt.to_csv(RESULTS / f"{STEM}_counts.csv", index=False)
    df.to_csv(RESULTS / f"{STEM}.csv", index=False)
    clv.to_csv(RESULTS / f"{STEM}_clv.csv", index=False)
    md = write_report(df, cnt, clv)
    (RESULTS / f"{STEM}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
