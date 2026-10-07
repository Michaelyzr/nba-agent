"""Gate audit runs: legacy vs split gate, with and without the enforced kill switch.

    python -m evaluation.gate_audit                 # test period (default; deadline-safe)
    python -m evaluation.gate_audit --period wf     # full walk-forward season as well
    python -m evaluation.gate_audit --report-only   # re-score saved runs

Pre-registered in docs/preregistration_gate.md. Writes evaluation/results/gate_audit.{csv,md}.

Deviation (deadline): the default run uses the secondary test window 1 Feb – 12 Apr
only. With `--warm-dev` (off by default under deadline), learning arms warm up on
Nov–Jan first; otherwise they start with an empty notebook on 1 Feb. Pass
`--period wf` for the primary walk-forward window. Documented in the report and
in preregistration_gate.md §Deviations.
"""
import argparse
import copy
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from evaluation.walkforward import (HOLDOUT, REGULAR_END, SEASON_START, WINDOWS, WalkForwardForecaster,
                                    market_days, train_win_models)
from replay import RUNS

RESULTS = Path(__file__).resolve().parent / "results"
# setup name -> (gate mode, learn, kill_switch, sizing, kelly_fraction)
SETUPS = {
    "legacy_full": ("legacy", True, None, "flat", 0.25),
    "split_full": ("split", True, None, "flat", 0.25),
    "split_kill": ("split", True, 100.0, "flat", 0.25),
    "split_nolearn": ("split", False, None, "flat", 0.25),
    "legacy_nolearn": ("legacy", False, None, "flat", 0.25),
}
LABELS = {
    "legacy_full": "Legacy gate + learning",
    "split_full": "Split gate + learning",
    "split_kill": "Split gate + learning + kill switch",
    "split_nolearn": "Split gate, no learning",
    "legacy_nolearn": "Legacy gate, no learning",
    "never": "Never trade",
}
PAIRS = [("split_full", "legacy_full"), ("split_kill", "split_full"), ("split_full", "split_nolearn"),
         ("split_full", "never"), ("legacy_full", "never"), ("split_kill", "never")]
TEST = ("2026-02-01", REGULAR_END)
WF = (WINDOWS[0][1], REGULAR_END)


def _job(job: dict, tables=None, base=None, price_index=None) -> dict:
    """One setup. When `tables`/`base`/`price_index` are passed, reuse them (sequential mode)."""
    import copy
    from agents.graph import MarketAgent, save_run
    from agents.notebook import Notebook
    from forecast.api import Forecaster
    from replay import Replay, load_tables

    t0 = time.time()
    if tables is None:
        tables = load_tables()
    if base is None:
        base = Forecaster.load()
    forecaster = (WalkForwardForecaster(pickle.loads(job["models"]), base.play_model, base.points_model)
                  if job.get("models") else base)
    gate, learn, kill, sizing, kf = job["spec"]
    notebook = Notebook.load(job["notebook"]) if job.get("notebook") else Notebook()
    agent = MarketAgent(notebook=copy.deepcopy(notebook), forecaster=forecaster, learn=learn, gate=gate,
                        sizing=sizing, kelly_fraction=kf)
    hook = agent.on_day_end if learn else None
    rp = Replay(tables, agent.policy, on_day_end=hook, kill_switch=kill, price_index=price_index)
    decisions, fills = rp.run(job["start"], job["end"])
    save_run(job["name"], decisions, fills, agent, rp.kill_trips)
    # Persist situations for the placebo audit when this is a no-learning walk-forward.
    if not learn and "walkforward" in job["name"]:
        from evaluation.gate_placebo import save_situations
        save_situations(RUNS / job["name"], agent.situations)
    return {"name": job["name"], "fills": len(fills), "trips": len(rp.kill_trips),
            "seconds": time.time() - t0, "rules": len(agent.notebook.rules),
            "active": sum(r["status"] == "active" for r in agent.notebook.rules),
            "price_index": rp.price_index}


def run_all(workers: int, which: list | None = None, period: str = "test",
            warm_dev: bool = False):
    blob = None
    if period in ("wf", "all"):
        starts = [SEASON_START] + [w[1] for w in WINDOWS]
        print(f"training M4 for {len(starts)} window starts", flush=True)
        models, _ = train_win_models(starts)
        blob = pickle.dumps(models)
    # Optional Nov–Jan warm-up, then test-period jobs, then optional walk-forward.
    from evaluation.gate_audit import _job as run_job
    batches = []
    if period in ("test", "all"):
        if warm_dev:
            dev = []
            for name, spec in SETUPS.items():
                if which and name not in which:
                    continue
                if spec[1]:
                    dev.append({"name": f"gate_audit/dev-{name}", "spec": spec, "start": "2025-11-01",
                                "end": "2026-01-31", "models": None})
            batches.append(dev)
        test_jobs = []
        for name, spec in SETUPS.items():
            if which and name not in which:
                continue
            nb = None
            if spec[1] and warm_dev:
                nb = str(RUNS / "gate_audit" / f"dev-{name}" / "notebook.json")
            test_jobs.append({"name": f"gate_audit/test-{name}", "spec": spec, "start": TEST[0],
                              "end": TEST[1], "models": None, "notebook": nb})
        batches.append(test_jobs)
    if period in ("wf", "all"):
        wf = []
        for name, spec in SETUPS.items():
            if which and name not in which:
                continue
            wf.append({"name": f"gate_audit/wf-{name}", "spec": spec, "start": WF[0],
                       "end": WINDOWS[-1][2], "models": blob})
        batches.append(wf)

    def _emit(r):
        print(f"  {r['name']}: {r['fills']} fills, {r['active']}/{r['rules']} rules active, "
              f"{r['trips']} kill trips in {r['seconds'] / 60:.1f} min", flush=True)

    shared_tables = shared_base = shared_index = None
    if workers <= 1:
        from forecast.api import Forecaster
        from replay import load_tables
        print("loading shared tables + forecaster once", flush=True)
        shared_tables = load_tables()
        shared_base = Forecaster.load()
        print(f"loaded {len(shared_tables['prices']):,} price rows", flush=True)

    for batch in batches:
        if not batch:
            continue
        print(f"running {len(batch)} jobs (workers={workers}): "
              f"{[j['name'] for j in batch]}", flush=True)
        if workers <= 1:
            for job in batch:
                r = run_job(job, shared_tables, shared_base, shared_index)
                shared_index = r.pop("price_index", shared_index)
                _emit(r)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for r in pool.map(run_job, batch):
                    r.pop("price_index", None)
                    _emit(r)


def score_setup(name: str, period: str, start: str, end: str, tables: dict) -> dict:
    from evaluation.scorer import notebook_metrics, trading_metrics

    folder = RUNS / "gate_audit" / f"{period}-{name}"
    fills = pd.read_parquet(folder / "fills.parquet") if (folder / "fills.parquet").exists() else pd.DataFrame()
    decisions = (pd.read_parquet(folder / "decisions.parquet")
                 if (folder / "decisions.parquet").exists() else None)
    games = tables["games"]
    days = market_days(tables, start, end)
    if len(fills):
        fills = fills[fills.game_id.isin(games.loc[games.date.isin(days), "game_id"])]
    metrics = trading_metrics(fills, decisions, games)
    table = day_table(fills, games, days)
    boot = bootstrap(table)
    worst = float(fills.merge(games[["game_id", "date"]], on="game_id").groupby("date").pnl.sum().min()) if len(fills) else 0.0
    trips_file = folder / "kill_switch.json"
    enforced = len(json.loads(trips_file.read_text())) if trips_file.exists() else 0
    nb = notebook_metrics(json.loads((folder / "notebook.json").read_text())) if (folder / "notebook.json").exists() else {}
    row = {"period": period, "setup": LABELS[name], "setup_id": name, "trades": int(table.trades.sum()),
           "worst_day": worst, "kill_switch_trips_enforced": enforced,
           "kill_switch_trips_measured": metrics.get("kill_switch_trips", 0), **nb}
    for m in boot.index:
        row[m] = boot.loc[m, "estimate"]
        row[f"{m}_lo"] = boot.loc[m, "ci_low"]
        row[f"{m}_hi"] = boot.loc[m, "ci_high"]
    return row, table


def report():
    from replay import load_tables
    tables = load_tables()
    rows, tabs = [], {}
    for period, (start, end) in (("wf", WF), ("test", TEST)):
        for name in list(SETUPS) + ["never"]:
            if name == "never":
                days = market_days(tables, start, end)
                table = day_table(pd.DataFrame(), tables["games"], days)
                boot = bootstrap(table)
                row = {"period": period, "setup": LABELS["never"], "setup_id": "never", "trades": 0,
                       "worst_day": 0.0, "kill_switch_trips_enforced": 0, "kill_switch_trips_measured": 0}
                for m in boot.index:
                    row[m] = boot.loc[m, "estimate"]
                    row[f"{m}_lo"] = boot.loc[m, "ci_low"]
                    row[f"{m}_hi"] = boot.loc[m, "ci_high"]
                rows.append(row)
                tabs[(period, name)] = table
                continue
            folder = RUNS / "gate_audit" / f"{period}-{name}"
            if not (folder / "fills.parquet").exists():
                continue
            # Restrict walk-forward totals to Nov–12 Apr (primary window)
            row, table = score_setup(name, period, start, end, tables)
            rows.append(row)
            tabs[(period, name)] = table
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "gate_audit.csv", index=False)

    sig = []
    for period in ("wf", "test"):
        for a, b in PAIRS:
            if (period, a) not in tabs or (period, b) not in tabs:
                continue
            d = paired(tabs[(period, a)], tabs[(period, b)])
            for m in ("mean_clv", "clv_dollars", "pnl"):
                r = d.loc[m]
                sig.append({"period": period, "comparison": f"{LABELS[a]} − {LABELS[b]}", "metric": m,
                            "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high,
                            "p_one_sided": r.p_a_gt_b})
    sig = pd.DataFrame(sig)
    sig.to_csv(RESULTS / "gate_audit_pairs.csv", index=False)

    def fmt(est, lo, hi, kind):
        if pd.isna(est):
            return "–"
        if kind == "clv":
            return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        return f"{est:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"

    lines = ["# Gate audit", "",
             f"Pre-registered in `docs/preregistration_gate.md`. Day-clustered bootstrap, {REPS} replicates, "
             f"seed {SEED}. **Reported window (deadline deviation):** test {TEST[0]}–{TEST[1]} "
             f"(not clean; learning arms start empty unless `--warm-dev`). Walk-forward "
             f"{WF[0]}–{WF[1]} is optional via `--period wf`.", "",
             "## Summaries", "",
             "| Period | Setup | Trades | Rules active | Mean CLV [95% CI] | CLV $ [95% CI] | P&L [95% CI] | "
             "Worst day | Kill trips (enforced / measured) |",
             "|" + " --- |" * 9]
    for r in frame.itertuples():
        lines.append(
            f"| {r.period} | {r.setup} | {r.trades} | {getattr(r, 'rules_active', 0) or 0} | "
            f"{fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | "
            f"{fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} | "
            f"{fmt(r.pnl, r.pnl_lo, r.pnl_hi, '$')} | {r.worst_day:+.0f} | "
            f"{r.kill_switch_trips_enforced} / {r.kill_switch_trips_measured} |")
    lines += ["", "## Paired comparisons", "",
              "| Period | Comparison | Metric | Difference [95% CI] | p (one-sided) |",
              "| --- | --- | --- | --- | --- |"]
    for r in sig.itertuples():
        kind = "clv" if r.metric == "mean_clv" else "$"
        p = "–" if pd.isna(r.p_one_sided) else f"{r.p_one_sided:.3f}"
        lines.append(f"| {r.period} | {r.comparison} | {r.metric} | "
                     f"{fmt(r.estimate, r.ci_low, r.ci_high, kind)} | {p} |")
    lines += ["", "## Hypotheses", "",
              "- **H1** (held-out gate days → fewer rules pass): compare `rules_active` for split_full vs "
              "legacy_full.",
              "- **H3** (kill switch cuts worst-day loss, mean CLV unchanged): compare split_kill vs "
              "split_full on worst_day and mean_clv.", ""]
    (RESULTS / "gate_audit.md").write_text("\n".join(lines) + "\n")
    print((RESULTS / "gate_audit.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = sequential (default; safer under machine load); >1 uses a process pool")
    ap.add_argument("--only", nargs="*", default=None, help="subset of setup ids")
    ap.add_argument("--period", choices=("test", "wf", "all"), default="test",
                    help="test = Feb–Apr only (default); wf = walk-forward; all = both")
    ap.add_argument("--warm-dev", action="store_true",
                    help="learn on Nov–Jan before the test window (slow; off by default)")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_all(args.workers, args.only, args.period, args.warm_dev)
    report()


if __name__ == "__main__":
    main()
