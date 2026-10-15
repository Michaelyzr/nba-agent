"""M4 news-shift baseline: "empty" (published) vs "anchor" (absences already known a day before tip).

    # fixed 4c, no learning: days are independent, so run it day-chunked in parallel (about 13 min per baseline)
    for b in empty anchor; do python -m evaluation.llm_agent_eval --window test --setups anchor \
        --name ../quick_wins/shift_fixed4_$b --shift-baseline $b --report ../quick_wins/shift_fixed4_$b --stem tmp_$b; done
    python -m evaluation.shift_baseline --arms split  # split-gate learning: one sequential replay per baseline (slow)
    python -m evaluation.shift_baseline --report-only

The anchored agent prices p = anchor mid (24 h before tip) + M4(out now) - M4(out at baseline). With the
"empty" baseline every absence on tonight's inactive list moves p, including players who were already out in
the team's previous game and so were in the anchor price. "anchor" subtracts M4 with those players out.
Same settings as evaluation/adaptive_edge.py arms fixed4 and split (flat $20, split gate, empty notebook
on 1 Feb). Runs go to runs/quick_wins/shift_<arm>_<baseline>/; the summary is printed and saved as JSON.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from evaluation.adaptive_edge import TEST, load_frozen
from evaluation.stats import bootstrap, day_table, paired
from evaluation.walkforward import market_days
from replay import RUNS

OUT = RUNS / "quick_wins"
ARMS = {"fixed4": False, "split": True}           # arm -> learn
BASELINES = ("empty", "anchor")
PUBLISHED = {"fixed4": RUNS / "adaptive_edge" / "fixed4", "split": RUNS / "adaptive_edge" / "split"}


def run_folder(arm: str, base: str):
    folder = OUT / f"shift_{arm}_{base}"
    return folder / "anchor" if (folder / "anchor" / "fills.parquet").exists() else folder


def run_one(job: dict) -> dict:
    from agents.graph import MarketAgent, save_run
    from agents.notebook import Notebook
    from forecast.api import Forecaster
    from replay import Replay

    t0 = time.time()
    tables, forecaster = load_frozen(), Forecaster.load()
    learn = ARMS[job["arm"]]
    agent = MarketAgent(notebook=Notebook(), forecaster=forecaster, learn=learn, gate="split", min_edge=0.04,
                        shift_baseline=job["baseline"])
    rp = Replay(tables, agent.policy, on_day_end=agent.on_day_end if learn else None)
    decisions, fills = rp.run(*TEST)
    save_run(f"quick_wins/shift_{job['arm']}_{job['baseline']}", decisions, fills, agent)
    return {**job, "fills": len(fills), "minutes": round((time.time() - t0) / 60, 1)}


def summary(fills: pd.DataFrame, table: pd.DataFrame) -> dict:
    b = bootstrap(table)
    staked = float((fills.price * fills.contracts).sum()) if len(fills) else 0.0
    fees = float(fills.fee.sum()) if len(fills) else 0.0
    out = {"trades": int(table.trades.sum()), "staked": round(staked, 2), "staked_with_fees": round(staked + fees, 2)}
    for m in ("pnl", "clv_dollars", "mean_clv"):
        out[m] = [float(b.loc[m, "estimate"]), float(b.loc[m, "ci_low"]), float(b.loc[m, "ci_high"])]
    return out


def brier_check(tables) -> dict:
    """Brier of the agent's forecast clip(anchor + M4 shift) on the test games, published vs anchored baseline.

    Uses the published per-game table (evaluation/results/m4_games.csv: M4 1 h / 1 min before tip, as in
    evaluation/m4_report.py) and the 24 h anchor (runs/win_nn/anchor.csv), recomputing only M4 "before"."""
    from agents.graph import anchor_out
    from evaluation.forecast_blend import RESULTS, clip, score_ci
    from forecast.api import Forecaster
    from replay import AsOf
    g = pd.read_csv(RESULTS / "m4_games.csv", dtype={"game_id": str})
    an = pd.read_csv(RUNS / "win_nn" / "anchor.csv", dtype={"game_id": str})
    t = g.merge(an, on="game_id")
    games = tables["games"].assign(game_id=lambda d: d.game_id.astype(str)).set_index("game_id")
    forecaster = Forecaster.load()
    prices = tables["prices"].sort_values("ts")
    index = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}
    before_empty, before_anchor, n_known = [], [], []
    for gid in t.game_id:           # anchored only once tonight's list is out, as in MarketAgent.forecast
        game = games.loc[gid]
        game = pd.Series({**game.to_dict(), "game_id": gid})
        view = AsOf(tables, pd.Timestamp(game.tip_time) - pd.Timedelta(minutes=60), index)
        game_t = next(games.loc[[gid]].reset_index().itertuples(index=False))
        known = anchor_out(view, game_t) if t.loc[t.game_id == gid, "players_out"].iloc[0] > 0 else []
        n_known.append(len(known))
        before_empty.append(forecaster.win(view, game_t, [])["p_home"])
        before_anchor.append(forecaster.win(view, game_t, known)["p_home"])
    t["m4_before_anchor"], t["n_known"] = before_anchor, n_known
    y = t.home_win.to_numpy(float)
    p_pub = clip(t.anchor + t.m4_after - t.m4_before).to_numpy(float)
    p_new = clip(t.anchor + t.m4_after - t.m4_before_anchor).to_numpy(float)
    d = score_ci(p_new, p_pub, y, t.date, "brier")
    t[["game_id", "date", "anchor", "m4_before", "m4_before_anchor", "m4_after", "n_known", "players_out",
       "home_win"]].to_csv(OUT / "shift_baseline_games.csv", index=False)
    return {"games": len(t), "brier_empty": float(np.mean((p_pub - y) ** 2)),
            "brier_anchor": float(np.mean((p_new - y) ** 2)), "brier_diff": list(d),
            "m4_before_recomputed_max_abs_diff": float(np.max(np.abs(np.array(before_empty) - t.m4_before))),
            "games_with_known_absences": int((t.n_known > 0).sum()),
            "mean_abs_shift_empty": float(np.mean(np.abs(t.m4_after - t.m4_before))),
            "mean_abs_shift_anchor": float(np.mean(np.abs(t.m4_after - t.m4_before_anchor)))}


def report() -> dict:
    tables = load_frozen()
    games = tables["games"]
    days = market_days(tables, *TEST)
    res = {"runs": {}, "paired": {}, "published_match": {}}
    tabs = {}
    for arm in ARMS:
        for base in BASELINES:
            folder = run_folder(arm, base)
            if not (folder / "fills.parquet").exists():
                continue
            fills = pd.read_parquet(folder / "fills.parquet")
            tabs[(arm, base)] = day_table(fills, games, days)
            res["runs"][f"{arm}_{base}"] = summary(fills, tabs[(arm, base)])
        pub = PUBLISHED[arm] / "fills.parquet"
        if (arm, "empty") in tabs and pub.exists():
            a = pd.read_parquet(run_folder(arm, "empty") / "fills.parquet")
            b = pd.read_parquet(pub)
            cols = ["game_id", "market_ticker", "side", "price", "contracts"]
            res["published_match"][arm] = bool(a[cols].sort_values(cols).reset_index(drop=True).equals(
                b[cols].sort_values(cols).reset_index(drop=True)))
        if (arm, "empty") in tabs and (arm, "anchor") in tabs:
            d = paired(tabs[(arm, "anchor")], tabs[(arm, "empty")])
            res["paired"][arm] = {m: [float(d.loc[m, "estimate"]), float(d.loc[m, "ci_low"]),
                                      float(d.loc[m, "ci_high"]), float(d.loc[m, "p_a_gt_b"])]
                                  for m in ("pnl", "clv_dollars", "mean_clv")}
    saved = OUT / "brier.json"
    res["brier"] = json.loads(saved.read_text()) if saved.exists() else brier_check(tables)
    (OUT / "shift_baseline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default="split")
    ap.add_argument("--baselines", default="empty,anchor")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        jobs = [{"arm": a, "baseline": b} for a in args.arms.split(",") for b in args.baselines.split(",")]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(run_one, jobs):
                print(r, flush=True)
    report()


if __name__ == "__main__":
    main()
