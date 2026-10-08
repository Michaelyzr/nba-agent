"""Arm D (DeepSeek reasoner tool agent + sceptic, 4¢ min edge) vs arm A (anchor) on the full test period.

D is replayed day by day from runs/llm_cache/ only: a cache miss aborts that day, so no API call is made and
only game-days the live run has fully answered are scored. A comes from the full-sample anchor run
(runs/llm_agent/test-full-anchor/anchor) restricted to the same days. Same replay rules as llm_agent_eval
(stake $20, caps, fills at the ask plus the Kalshi fee, no learning).

    python -m evaluation.llm_sceptic_full [--workers 8]
"""
import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.llm_agent_eval import OUT, RESULTS, WINDOWS, _shared, close_clv, load_frozen, load_traces
from evaluation.stats import REPS, SEED, bootstrap, day_table, paired

MODEL = "deepseek-reasoner"
MIN_EDGE = 0.04
ANCHOR = OUT / "test-full-anchor" / "anchor"
PARTS = OUT / "sceptic-full-cacheonly"
FULL_RUNS = ("test-reasoner-full", "test-reasoner-full-b")
LONG_SHOT = 0.20
BANKROLL = 1000.0
LAUNCH = pd.Timestamp("2026-10-08 21:19", tz="Asia/Hong_Kong").timestamp()


class CacheMiss(BaseException):
    """Not an Exception, so CachedLLM's error handling does not swallow it."""


class CacheOnly:
    model = MODEL

    def generate(self, system, prompt, meta=None):
        raise CacheMiss()


def run_day(day: str) -> dict:
    path = PARTS / f"{day}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from agents.llm_client import CachedLLM
    from agents.tool_agent import ToolAgent
    from replay import Replay
    s = _shared()
    agent = ToolAgent(CachedLLM(CacheOnly()), sceptic=True, forecaster=s["forecaster"], impact=s["impact"],
                      players=s["players"], min_edge=MIN_EDGE)
    try:
        decisions, fills = Replay(s["tables"], agent.policy).run(day, day)
    except CacheMiss:
        return {"day": day, "complete": False}
    out = {"day": day, "complete": True, "fills": fills, "traces": agent.traces}
    PARTS.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def from_runs(games: pd.DataFrame, days: list) -> list:
    """Per-day parts from finished full-sample D runs (fills.parquet written only when a run completes)."""
    day_of = games.set_index("game_id").date.astype(str)
    parts = {}
    for run in FULL_RUNS:
        folder = OUT / run / "tool_sceptic"
        if not (folder / "meta.json").exists():
            continue
        meta = json.loads((folder / "meta.json").read_text())
        fills = pd.read_parquet(folder / "fills.parquet") if (folder / "fills.parquet").exists() else pd.DataFrame()
        traces = load_traces(folder)
        for d in days:
            if meta["start"] <= d <= meta["end"] and d not in parts:
                parts[d] = {"day": d, "complete": True,
                            "fills": fills[fills.game_id.map(day_of) == d] if len(fills) else fills,
                            "traces": [t for t in traces if day_of.get(t["game_id"]) == d]}
    return list(parts.values())


def spend() -> tuple:
    from agents.llm_client import CACHE_DIR, cost_usd
    job = total = 0.0
    for p in CACHE_DIR.glob("*/*.json"):
        d = json.loads(p.read_text())
        if d.get("model") != MODEL:
            continue
        c = cost_usd(MODEL, int(d["tokens_in"]), int(d["tokens_out"]))
        total += c
        job += c if float(d.get("created", 0)) >= LAUNCH else 0.0
    return job, total


def summary(t: pd.DataFrame, f: pd.DataFrame) -> dict:
    b = bootstrap(t)
    staked = float((f.price * f.contracts + f.fee).sum()) if len(f) else 0.0
    ci = lambda m: (b.loc[m, "estimate"], b.loc[m, "ci_low"], b.loc[m, "ci_high"])
    pnl, clv = ci("pnl"), ci("clv_dollars")
    ls = f[f.price > LONG_SHOT] if len(f) else f
    return {"trades": int(t.trades.sum()), "staked": staked,
            "pnl": pnl[0], "pnl_lo": pnl[1], "pnl_hi": pnl[2],
            "ret_staked": pnl[0] / staked if staked else np.nan, "ret_1000": pnl[0] / BANKROLL,
            "clv_dollars": clv[0], "clv_lo": clv[1], "clv_hi": clv[2],
            "win_rate": float(f.outcome.eq(1).mean()) if len(f) else np.nan,
            "pnl_ex_longshots": float(ls.pnl.sum()) if len(ls) else 0.0,
            "longshot_trades": int((f.price <= LONG_SHOT).sum()) if len(f) else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--live", type=Path, default=OUT / "test-reasoner-full" / "tool_sceptic",
                    help="finished live D run over the whole period; the cache-only replay is used if absent")
    ap.add_argument("--no-replay", action="store_true", help="score only days covered by finished runs")
    args = ap.parse_args()
    from evaluation.walkforward import market_days
    tables = load_frozen()
    games = tables["games"]
    start, end = WINDOWS["test"]
    days = market_days(tables, start, end)
    t0 = time.time()
    parts = from_runs(games, days)
    todo = [] if args.no_replay else [d for d in days if d not in {p["day"] for p in parts}]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for p in pool.map(run_day, todo):
            parts.append(p)
            print(p["day"], "complete" if p["complete"] else "incomplete", flush=True)
    done = [p for p in parts if p["complete"]]
    dset = sorted(p["day"] for p in done)
    print(f"complete days {len(dset)}/{len(days)} in {(time.time() - t0) / 60:.1f} min")

    day_of = games.set_index("game_id").date.astype(str)
    a_tr = [t for t in load_traces(ANCHOR) if day_of.get(t["game_id"]) in dset]
    a_all = load_traces(ANCHOR)
    a_f = pd.read_parquet(ANCHOR / "fills.parquet")
    a_f = a_f[a_f.game_id.map(day_of).isin(dset)]
    d_tr = [t for p in done for t in p["traces"]]
    fl = [p["fills"] for p in done if len(p["fills"])]
    d_f = pd.concat(fl, ignore_index=True) if fl else pd.DataFrame(columns=a_f.columns)
    ta, td = day_table(a_f, games, dset), day_table(d_f, games, dset)
    rows = {"A": summary(ta, a_f), "D": summary(td, d_f)}
    diff = paired(td, ta)

    prices = tables["prices"].sort_values("ts")
    pidx = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}
    live = [t for t in d_tr if t["status"] not in ("held", "idle")]
    sc = [t for t in live if "sceptic" in t]
    vetoed = [t for t in sc if t["sceptic"]["verdict"] != "approve"]
    veto_clv = [close_clv(tables, pidx, t["candidate"]["ticker"], t["candidate"]["side"], t["candidate"]["price"],
                          t["game_id"]) for t in vetoed if "candidate" in t]
    job_cost, total_cost = spend()

    g = games.set_index("game_id")
    trades = d_f.assign(game=lambda x: x.game_id.map(lambda i: f"{g.away_team[i]} @ {g.home_team[i]}"),
                        date=lambda x: x.game_id.map(day_of),
                        team=lambda x: x.market_ticker.str.split("-").str[-1],
                        staked=lambda x: x.price * x.contracts + x.fee,
                        clv_dollars=lambda x: x.clv * x.contracts) if len(d_f) else pd.DataFrame()
    n_pts = len(a_tr)
    out = pd.DataFrame([{"arm": k, "days": len(dset), "decision_points": n_pts, **v} for k, v in rows.items()])
    out.to_csv(RESULTS / "llm_sceptic_full.csv", index=False)
    if len(trades):
        trades[["date", "game", "team", "side", "price", "contracts", "staked", "outcome", "pnl", "clv",
                "clv_dollars"]].to_csv(RESULTS / "llm_sceptic_full_trades.csv", index=False)

    f = lambda v: f"{v:+,.0f}"
    md = ["# Arm D (reasoner tool agent + sceptic) vs arm A (anchor), full test period", "",
          f"**{'Complete' if len(dset) == len(days) else 'Partial'} result: {len(dset)} of {len(days)} game-days, "
          f"{n_pts} of {len(a_all)} decision points** (every decision point on a scored day; no subsampling). D comes "
          f"from the finished full-sample `deepseek-reasoner` runs ({', '.join(FULL_RUNS)}); days they do not cover "
          f"are replayed from the LLM cache only and count only if every call on them was cached. A is the "
          f"full-sample anchor run restricted to the same days. Staked includes fees. Replay: $20 stake, caps "
          f"$50/$100/$300, fills at the ask plus the "
          f"Kalshi fee, min edge {MIN_EDGE * 100:.0f}¢, no learning. 95% CIs from a day-clustered bootstrap "
          f"({REPS} replicates, seed {SEED}).", "",
          "| Arm | Trades | Staked | P&L [95% CI] | Return on staked | Return on $1,000 | CLV $ [95% CI] | Win rate |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in rows.items():
        md.append(f"| {k} | {r['trades']} | ${r['staked']:,.0f} | {f(r['pnl'])} [{f(r['pnl_lo'])}, {f(r['pnl_hi'])}] "
                  f"| {r['ret_staked']:+.1%} | {r['ret_1000']:+.1%} | {f(r['clv_dollars'])} [{f(r['clv_lo'])}, "
                  f"{f(r['clv_hi'])}] | {r['win_rate']:.0%} |" if r["trades"] else
                  f"| {k} | 0 | $0 | +0 | – | +0.0% | +0 | – |")
    md += ["", "Paired D − A on the same days: " + "; ".join(
        f"{m} {diff.loc[m, 'estimate']:+,.2f} [{diff.loc[m, 'ci_low']:+,.2f}, {diff.loc[m, 'ci_high']:+,.2f}]"
        for m in ("pnl", "clv_dollars")), "",
        f"Sceptic: {len(sc)} reviews, {len(sc) - len(vetoed)} approved, {len(vetoed)} vetoed. Mean CLV per contract: "
        f"vetoed {np.nanmean(veto_clv) if veto_clv else float('nan'):+.4f} vs kept "
        f"{d_f.clv.mean() if len(d_f) else float('nan'):+.4f}.", "",
        f"Long-shot dependence (price ≤ {LONG_SHOT * 100:.0f}¢): D P&L excluding long shots "
        f"{f(rows['D']['pnl_ex_longshots'])} ({rows['D']['longshot_trades']} long-shot trades); A "
        f"{f(rows['A']['pnl_ex_longshots'])} ({rows['A']['longshot_trades']}).", "",
        "Kept D trades:", "", "| Date | Game | Bought | Price | Won | P&L | CLV / contract | CLV $ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in (trades.itertuples() if len(trades) else []):
        md.append(f"| {x.date} | {x.game} | {x.team} {x.side} | {x.price:.2f} | {'won' if x.pnl > 0 else 'lost'} "
                  f"| {x.pnl:+.2f} | {x.clv:+.3f} | {x.clv_dollars:+.2f} |")
    md += ["", f"DeepSeek spend (token estimate at list price, all reasoner calls in the cache): "
           f"${job_cost:.2f} since the full run started, ${total_cost:.2f} in total.", ""]
    (RESULTS / "llm_sceptic_full.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
