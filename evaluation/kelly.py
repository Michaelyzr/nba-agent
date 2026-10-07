"""Fractional Kelly sizing vs flat $20 stake.

    python -m evaluation.kelly
    python -m evaluation.kelly --report-only

Pre-registered in docs/preregistration_gate.md (H4). Arms: flat, ¼-Kelly, optional
calibrated ¼-Kelly (if forecast/calibrate.py is importable), never trade.
No learning, split gate unused. Writes evaluation/results/kelly.{csv,md}.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from evaluation.walkforward import REGULAR_END, market_days
from replay import RUNS

RESULTS = Path(__file__).resolve().parent / "results"
# Deadline default: test window only. Pass --start/--end for the primary Nov–Apr window.
START, END = "2026-02-01", REGULAR_END
ARMS = {
    "flat": ("flat", 0.25, False),
    "kelly": ("kelly", 0.25, False),
    "kelly_cal": ("kelly", 0.25, True),
}
LABELS = {"flat": "Flat $20", "kelly": "¼-Kelly", "kelly_cal": "¼-Kelly + Platt M4", "never": "Never trade"}


def _calibrated_forecaster():
    """Anchored agent still uses Forecaster; wrap M4 with Platt fit before 1 Feb if available."""
    from forecast.api import Forecaster
    from forecast.calibrate import CalibratedWinModel, fit_before, month_starts, out_of_sample
    from forecast.baselines import load_source
    from forecast.history import History
    from forecast.win import training_rows

    base = Forecaster.load()
    player_games, games = load_source("frozen")
    rows = training_rows(History(player_games, games), games).merge(games[["game_id", "date"]], on="game_id")
    oos = out_of_sample(rows, month_starts("2023-11-01", "2026-06-01"))
    cal = fit_before(oos, "2026-02-01", "platt")
    return Forecaster(CalibratedWinModel(base.win_model, cal), base.play_model, base.points_model)


def _job(job: dict) -> dict:
    from agents.graph import MarketAgent, save_run
    from forecast.api import Forecaster
    from replay import Replay, load_tables

    t0 = time.time()
    sizing, kf, calibrated = job["spec"]
    try:
        forecaster = _calibrated_forecaster() if calibrated else Forecaster.load()
    except Exception as exc:
        return {"name": job["name"], "error": f"{exc.__class__.__name__}: {exc}", "seconds": time.time() - t0}
    agent = MarketAgent(forecaster=forecaster, learn=False, sizing=sizing, kelly_fraction=kf, gate="legacy")
    rp = Replay(load_tables(), agent.policy, kill_switch=None)
    decisions, fills = rp.run(job["start"], job["end"])
    save_run(job["name"], decisions, fills, agent)
    return {"name": job["name"], "fills": len(fills), "seconds": time.time() - t0}


def run_all(workers: int):
    jobs = [{"name": f"kelly/{name}", "spec": spec, "start": START, "end": END} for name, spec in ARMS.items()]
    # Skip calibrated arm if calibrate module missing
    try:
        import forecast.calibrate  # noqa: F401
    except ImportError:
        jobs = [j for j in jobs if j["name"] != "kelly/kelly_cal"]
    from evaluation.kelly import _job as run_job
    print(f"running {len(jobs)} Kelly arms (workers={workers})", flush=True)
    if workers <= 1:
        for job in jobs:
            print(f"  {run_job(job)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(run_job, jobs):
                print(f"  {r}", flush=True)


def report():
    from replay import load_tables
    tables = load_tables()
    games = tables["games"]
    days = market_days(tables, START, END)
    tabs, rows = {}, []
    for name in list(ARMS) + ["never"]:
        max_dd = 0.0
        if name == "never":
            table = day_table(pd.DataFrame(), games, days)
        else:
            path = RUNS / "kelly" / name / "fills.parquet"
            if not path.exists():
                continue
            fills = pd.read_parquet(path)
            fills = fills[fills.game_id.isin(games.loc[games.date.isin(days), "game_id"])]
            table = day_table(fills, games, days)
            settled = fills.dropna(subset=["outcome"])
            if len(settled):
                max_dd = float((settled.pnl.cumsum().cummax() - settled.pnl.cumsum()).max())
        boot = bootstrap(table)
        row = {"setup": LABELS[name], "setup_id": name, "trades": int(table.trades.sum()),
               "max_drawdown": max_dd}
        for m in boot.index:
            row[m] = boot.loc[m, "estimate"]
            row[f"{m}_lo"] = boot.loc[m, "ci_low"]
            row[f"{m}_hi"] = boot.loc[m, "ci_high"]
        rows.append(row)
        tabs[name] = table
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "kelly.csv", index=False)
    pairs = []
    for a, b in (("kelly", "flat"), ("kelly", "never"), ("flat", "never"), ("kelly_cal", "kelly"),
                 ("kelly_cal", "flat")):
        if a not in tabs or b not in tabs:
            continue
        d = paired(tabs[a], tabs[b])
        for m in ("mean_clv", "clv_dollars", "pnl"):
            r = d.loc[m]
            pairs.append({"comparison": f"{LABELS[a]} − {LABELS[b]}", "metric": m, "estimate": r.estimate,
                          "ci_low": r.ci_low, "ci_high": r.ci_high, "p_one_sided": r.p_a_gt_b})
    pd.DataFrame(pairs).to_csv(RESULTS / "kelly_pairs.csv", index=False)

    def fmt(est, lo, hi, kind):
        if pd.isna(est):
            return "–"
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]" if kind == "clv" else f"{est:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"

    lines = ["# Fractional Kelly sizing", "",
             f"No-learning market-anchored agent, {START}–{END}. Stake = fraction × Kelly × $1,000 bankroll, "
             f"capped at $50, same game/day caps. Day-clustered bootstrap {REPS} reps, seed {SEED}.", "",
             "| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L [95% CI] | Max drawdown |",
             "|" + " --- |" * 6]
    for r in frame.itertuples():
        lines.append(f"| {r.setup} | {r.trades} | {fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
                     f"{fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} | "
                     f"{fmt(r.pnl, r.pnl_lo, r.pnl_hi, '$')} | {r.max_drawdown:+.0f} |")
    lines += ["", "## Paired comparisons", "",
              "| Comparison | Metric | Difference [95% CI] | p (one-sided) |", "| --- | --- | --- | --- |"]
    for r in pd.DataFrame(pairs).itertuples():
        kind = "clv" if r.metric == "mean_clv" else "$"
        p = "–" if pd.isna(r.p_one_sided) else f"{r.p_one_sided:.3f}"
        lines.append(f"| {r.comparison} | {r.metric} | {fmt(r.estimate, r.ci_low, r.ci_high, kind)} | {p} |")
    lines += ["", "## H4", "",
              "Pre-registered: ¼-Kelly does not beat flat $20 on P&L or CLV $. Report as a null unless "
              "Kelly − flat has a P&L CI above 0.", ""]
    (RESULTS / "kelly.md").write_text("\n".join(lines) + "\n")
    print((RESULTS / "kelly.md").read_text())


def main():
    global START, END
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    START, END = args.start, args.end
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_all(args.workers)
    report()


if __name__ == "__main__":
    main()
