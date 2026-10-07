"""G: evaluate the learned trade/pass policy (forecast/policy.py) against the agents and never trade.

    python -m evaluation.learned_policy extract     # as-of decision rows, Nov .. 14 Jun (walk-forward M4), ~15 min
    python -m evaluation.learned_policy select      # model and threshold chosen on 15-31 Jan, fit on 1 Nov-14 Jan
    python -m evaluation.learned_policy test        # refit on Nov-Jan, replay 1 Feb-12 Apr, report
    python -m evaluation.learned_policy holdout     # play-in and play-offs, refuses to run a second time
    python -m evaluation.learned_policy report      # re-score saved runs

Pre-registration: docs/preregistration_learned_policy.md (committed before the test step).
Arms on the same game-days: learned policy (validation-selected; may be "never trade"), learned forced to
trade (best validation config with at least 10 trades), best MLP config (the deep-learning arm), the
deterministic agent without learning and the full agent (runs/walkforward/, same walk-forward M4), and
never trade. Statistics: evaluation/stats.py day-clustered bootstrap, 2,000 replicates, seed 7606.
"""
import argparse
import json
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from forecast.policy import FEATURES, GRID, HOLDOUT, MIN_FORCED_TRADES, TEST, TRAIN, LearnedPolicy, PolicyModel, select, split

RESULTS = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "runs" / "learned_policy"
ROWS = OUT / "rows.parquet"
SPANS = [("2025-11-01", "2025-11-30"), ("2025-12-01", "2025-12-31"), ("2026-01-01", "2026-01-31"),
         ("2026-02-01", "2026-02-28"), ("2026-03-01", "2026-03-31"), ("2026-04-01", "2026-04-12"),
         ("2026-04-13", "2026-06-14")]
ARMS = {"learned": "Learned policy (validation-selected)", "forced": "Learned, forced to trade",
        "mlp": "Learned MLP (best MLP config)", "no_learning": "Agent: anchor, no learning",
        "full": "Full agent: anchor + learning", "never": "Never trade"}
PAIRS = [("learned", "never"), ("learned", "no_learning"), ("learned", "full"), ("forced", "never"),
         ("forced", "no_learning"), ("mlp", "no_learning"), ("no_learning", "never"), ("full", "never")]


# ---------------- extract ----------------

def _extract_job(job: dict) -> pd.DataFrame:
    from evaluation.walkforward import WalkForwardForecaster
    from forecast.api import Forecaster
    from forecast.policy import decision_rows
    from replay import load_tables

    base = Forecaster.load()
    forecaster = WalkForwardForecaster(pickle.loads(job["models"]), base.play_model, base.points_model)
    t0 = time.time()
    rows = decision_rows(load_tables(), forecaster, job["start"], job["end"])
    print(f"  {job['start']}..{job['end']}: {len(rows)} option rows in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def extract(workers: int):
    from evaluation.walkforward import SEASON_START, WINDOWS, train_win_models

    OUT.mkdir(parents=True, exist_ok=True)
    models, _ = train_win_models([SEASON_START] + [w[1] for w in WINDOWS])
    blob = pickle.dumps(models)
    jobs = [{"start": s, "end": e, "models": blob} for s, e in SPANS]
    from evaluation.learned_policy import _extract_job as job_fn  # picklable under "python -m"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = pd.concat(list(pool.map(job_fn, jobs)), ignore_index=True)
    rows.to_parquet(ROWS, index=False)
    print(f"{len(rows)} option rows, {rows.groupby(['game_id', 'as_of']).ngroups} decision points, "
          f"{rows.game_id.nunique()} games -> {ROWS}")


def load_rows() -> pd.DataFrame:
    return pd.read_parquet(ROWS)


# ---------------- select (training data only) ----------------

def run_select():
    rows = load_rows()
    train = split(rows, *TRAIN)
    table, best, forced = select(train)
    mlp = table[table.family == "mlp"].sort_values(["net_dollars", "trades"], ascending=[False, True]).iloc[0].to_dict()
    table.to_csv(RESULTS / "learned_policy_selection.csv", index=False)
    chosen = {"learned": best, "forced": forced, "mlp": mlp}
    (OUT / "selected.json").write_text(json.dumps(chosen, indent=2, default=str))
    print(table.sort_values("net_dollars", ascending=False).head(12).to_string(index=False))
    for k, v in chosen.items():
        print(k, "->", None if v is None else {x: v[x] for x in ("config", "tau", "trades", "net_dollars")})


# ---------------- test / holdout replays ----------------

def _fit(choice: dict, train: pd.DataFrame):
    if choice is None or choice["family"] == "never":
        return None
    return PolicyModel(choice["family"], choice["params"]).fit(train)


def _replay_arm(job: dict) -> str:
    from agents.graph import save_run
    from replay import Replay, load_tables

    pol = LearnedPolicy(job["span"], job["model"], job["tau"])
    t0 = time.time()
    decisions, fills = Replay(load_tables(), pol.policy).run(job["start"], job["end"])
    save_run(f"learned_policy/{job['period']}-{job['arm']}", decisions, fills)
    return f"{job['period']} {job['arm']} ({pol.name}): {len(fills)} fills in {(time.time() - t0) / 60:.1f} min"


def run_period(period: str, workers: int = 3):
    start, end = TEST if period == "test" else HOLDOUT
    marker = OUT / "holdout_done.json"
    if period == "holdout" and marker.exists():
        raise SystemExit(f"the holdout was already scored once ({marker}); use 'report' to re-score saved runs")
    rows = load_rows()
    train = split(rows, *TRAIN)
    chosen = json.loads((OUT / "selected.json").read_text())
    span = split(rows, start, end)
    models, jobs = {}, []
    for arm in ("learned", "forced", "mlp"):
        models[arm] = _fit(chosen[arm], train)
        folder = OUT / f"{period}-{arm}"
        if models[arm] is None:                       # never trade: no replay needed, no fills = $0
            if (folder / "fills.parquet").exists():
                (folder / "fills.parquet").unlink()
            print(f"{period} {arm}: never trade (validation chose abstention)")
            continue
        jobs.append({"period": period, "arm": arm, "span": span, "model": models[arm],
                     "tau": float(chosen[arm]["tau"]), "start": start, "end": end})
    from evaluation.learned_policy import _replay_arm as job_fn  # picklable under "python -m"
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        for line in pool.map(job_fn, jobs):
            print(line, flush=True)
    with open(OUT / f"{period}_models.pkl", "wb") as f:
        pickle.dump(models, f)
    if period == "holdout":
        marker.write_text(json.dumps({"scored_at": pd.Timestamp.now(tz="UTC").isoformat()}))


# ---------------- report ----------------

def arm_tables(period: str, tables: dict, days: list) -> dict:
    from evaluation.walkforward import load_fills
    from replay import RUNS

    games = tables["games"]
    folders = {a: OUT / f"{period}-{a}" for a in ("learned", "forced", "mlp")}
    folders.update(no_learning=RUNS / "walkforward" / "no_learning", full=RUNS / "walkforward" / "full")
    out = {}
    for arm in ARMS:
        f = load_fills(folders[arm]) if arm in folders else pd.DataFrame()
        if len(f):
            f = f[f.game_id.isin(games.loc[games.date.isin(days), "game_id"])]
        out[arm] = day_table(f, games, days)
    return out


def fmt(est, lo, hi, kind):
    if pd.isna(est):
        return "–"
    if kind == "clv":
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return f"{est:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"


def period_report(label: str, tabs: dict) -> tuple:
    solo, pairs = [], []
    for arm, t in tabs.items():
        b = bootstrap(t)
        solo.append({"period": label, "arm": ARMS[arm], "key": arm, "days": len(t), "trades": int(t.trades.sum()),
                     **{f"{m}{s}": b.loc[m, c] for m in ("mean_clv", "clv_dollars", "pnl")
                        for s, c in (("", "estimate"), ("_lo", "ci_low"), ("_hi", "ci_high"))},
                     "p_clv_gt_0": b.loc["mean_clv", "p_gt_0"]})
    for a, b_ in PAIRS:
        d = paired(tabs[a], tabs[b_])
        for m in ("clv_dollars", "pnl"):
            r = d.loc[m]
            pairs.append({"period": label, "comparison": f"{ARMS[a]} − {ARMS[b_]}", "metric": m,
                          "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_one_sided": r.p_a_gt_b})
    return pd.DataFrame(solo), pd.DataFrame(pairs)


def solo_md(solo: pd.DataFrame) -> list:
    lines = ["| Arm | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] |",
             "| --- | --- | --- | --- | --- |"]
    for r in solo.itertuples():
        lines.append(f"| {r.arm} | {r.trades} | {fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} "
                     f"| {fmt(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} "
                     f"| {fmt(r.pnl, r.pnl_lo, r.pnl_hi, '$')} |")
    return lines


def pairs_md(pairs: pd.DataFrame) -> list:
    lines = ["| Comparison | Metric | Difference [95% CI] | p (one-sided, first > second) |",
             "| --- | --- | --- | --- |"]
    for r in pairs.itertuples():
        p = "–" if pd.isna(r.p_one_sided) else f"{r.p_one_sided:.3f}"
        lines.append(f"| {r.comparison} | {'CLV $' if r.metric == 'clv_dollars' else 'P&L'} "
                     f"| {fmt(r.estimate, r.ci_low, r.ci_high, '$')} | {p} |")
    return lines


def reliability(rows: pd.DataFrame, model: PolicyModel, bins=10) -> pd.DataFrame:
    """Deciles of predicted net CLV on the period's option rows vs the realised mean (the value model's calibration)."""
    r = rows.dropna(subset=["net_clv"]).assign(pred=model.predict(rows.dropna(subset=["net_clv"])))
    r["bin"] = pd.qcut(r.pred.rank(method="first"), bins, labels=False)
    return r.groupby("bin").agg(pred=("pred", "mean"), realised=("net_clv", "mean"), n=("pred", "size")).reset_index()


def chart(solo: pd.DataFrame, rel: pd.DataFrame, sel: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for fam, color in (("ridge", "#1f77b4"), ("gbm", "#2ca02c"), ("mlp", "#9467bd")):
        for cfg, g in sel[sel.family == fam].groupby("config"):
            ax.plot(g.tau * 100, g.net_dollars, marker="o", color=color, alpha=0.8, label=cfg)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("threshold tau (cents of predicted net CLV)")
    ax.set_ylabel("validation net CLV $ (15–31 Jan)")
    ax.set_title("Selection on validation (training data only)")
    ax.legend(fontsize=6)
    ax = axes[1]
    t = solo[solo.period.str.startswith("Test")]
    err = np.vstack([t.clv_dollars - t.clv_dollars_lo, t.clv_dollars_hi - t.clv_dollars])
    colors = ["#9467bd", "#c5b0d5", "#8c564b", "#ff7f0e", "#1f77b4", "#7f7f7f"]
    ax.bar(range(len(t)), t.clv_dollars, yerr=err, capsize=4, color=colors[:len(t)])
    ax.set_xticks(range(len(t)), ["learned", "forced", "MLP", "no learn", "full", "never"], fontsize=8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("CLV $ (95% CI, day-clustered)")
    ax.set_title("Test 1 Feb–12 Apr: closing-line value")
    ax = axes[2]
    if rel is not None and len(rel):
        lo = min(rel.pred.min(), rel.realised.min())
        hi = max(rel.pred.max(), rel.realised.max())
        ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.8, label="perfect")
        ax.plot(rel.pred, rel.realised, marker="o", color="#9467bd", label="forced model, test options")
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.legend(fontsize=8)
    ax.set_xlabel("predicted net CLV per contract (decile mean)")
    ax.set_ylabel("realised net CLV per contract")
    ax.set_title("Reliability of the value model (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report():
    from evaluation.walkforward import market_days
    from replay import load_tables

    tables, rows = load_tables(), load_rows()
    chosen = json.loads((OUT / "selected.json").read_text())
    sel = pd.read_csv(RESULTS / "learned_policy_selection.csv")
    with open(OUT / "test_models.pkl", "rb") as f:
        models = pickle.load(f)
    solo, pairs = [], []
    periods = [("Test 1 Feb–12 Apr", "test", TEST)]
    if (OUT / "holdout_done.json").exists():
        periods.append(("Holdout 13 Apr–14 Jun (seen once)", "holdout", HOLDOUT))
    for label, key, (start, end) in periods:
        s, p = period_report(label, arm_tables(key, tables, market_days(tables, start, end)))
        solo.append(s)
        pairs.append(p)
    solo, pairs = pd.concat(solo, ignore_index=True), pd.concat(pairs, ignore_index=True)
    test_rows = split(rows, *TEST)
    rel_model = models.get("forced")
    rel = reliability(test_rows, rel_model) if rel_model is not None else None
    corr = (float(pd.Series(rel_model.predict(test_rows.dropna(subset=["net_clv"])))
                  .corr(test_rows.dropna(subset=["net_clv"]).net_clv.reset_index(drop=True), method="spearman"))
            if rel_model is not None else np.nan)
    train = split(rows, *TRAIN)
    oracle = test_rows.dropna(subset=["net_clv"]).sort_values("net_clv", ascending=False).groupby("game_id").head(1)
    oracle = oracle[oracle.net_clv > 0]
    rule = test_rows[test_rows.rule_act].dropna(subset=["net_clv"])
    out = pd.concat([solo.assign(kind="arm"), pairs.assign(kind="paired")], ignore_index=True)
    out.to_csv(RESULTS / "learned_policy.csv", index=False)
    if rel is not None:
        rel.to_csv(RESULTS / "learned_policy_reliability.csv", index=False)
    chart(solo, rel, sel, RESULTS / "learned_policy.png")

    def desc(c):
        return "none (fewer than 10 validation trades everywhere)" if c is None else (
            f"{c['config']}, tau = {c['tau']}, {int(c['trades'])} validation trades, "
            f"validation net CLV ${float(c['net_dollars']):+.2f}")

    md = ["# Learned trade/pass policy (option G)", "",
          "Pre-registered in `docs/preregistration_learned_policy.md` before the test window was scored. "
          f"Decision rows: every replay decision time × game-winner market × side (yes/no), as-of features "
          f"({', '.join(FEATURES)}), label = closing-line value after the Kalshi fee per contract (full information: "
          "the mid at tip is recorded for both sides). M4 is the walk-forward model retrained before each month, "
          "the same as the agent runs in `runs/walkforward/`.", "",
          f"Training rows: {len(train):,} options ({train.groupby(['game_id', 'as_of']).ngroups:,} decision points, "
          f"{train.game_id.nunique()} games), 1 Nov–31 Jan. Hyper-parameters and the threshold were chosen by fitting on "
          "1 Nov–14 Jan and scoring net CLV dollars on 15–31 Jan; the chosen configuration was then refit on all of "
          f"1 Nov–31 Jan. Test rows: {len(test_rows):,} options, 1 Feb–12 Apr.", "",
          "## What validation chose", "",
          f"- **Learned policy:** {desc(chosen['learned'])}.",
          f"- **Forced to trade (diagnostic):** {desc(chosen['forced'])}.",
          f"- **Best MLP (deep-learning arm, diagnostic):** {desc(chosen['mlp'])}.", "",
          "Full grid: `evaluation/results/learned_policy_selection.csv`.", ""]
    for label, _, _ in periods:
        md += [f"## {label}", "", *solo_md(solo[solo.period == label]), "", "Paired differences on the same game-days "
               f"(day-clustered bootstrap, {REPS} replicates, seed {SEED}):", "",
               *pairs_md(pairs[pairs.period == label]), ""]
    md += ["## Reliability of the value model", "",
           "Deciles of the forced model's predicted net CLV over every test option, against the realised mean. "
           "A value model with real skill would rise along the diagonal above zero in the top decile.", ""]
    if rel is not None:
        md += ["| Decile | Predicted net CLV | Realised net CLV | Options |", "| --- | --- | --- | --- |"]
        md += [f"| {int(r.bin) + 1} | {r.pred:+.4f} | {r.realised:+.4f} | {int(r.n)} |" for r in rel.itertuples()]
        md += ["", f"Spearman correlation between predicted and realised net CLV on test options: {corr:+.3f}.", ""]
    md += ["## Reference points on the test rows (not policies)", "",
           f"- **Deterministic rule on the same rows** (gap after fees > 4 points, every decision time, no caps): "
           f"{len(rule)} option-times, mean net CLV {rule.net_clv.mean():+.4f} per contract.",
           f"- **Oracle with look-ahead** (best positive-net-CLV option per game, knowing the close): {len(oracle)} games, "
           f"mean net CLV {oracle.net_clv.mean():+.4f}. This is the ceiling a perfect predictor could reach; "
           "it shows the label has signal to learn if the features carried it.", "",
           "## Reading", "",
           "Expected from M6 (the move to tip is a martingale) and pre-registered: the learned value model cannot "
           "predict net CLV above zero, so validation chooses to pass, or a forced policy loses roughly the "
           "half-spread plus fee per contract. See the pre-registration for the hypotheses and decision rules."]
    (RESULTS / "learned_policy.md").write_text("\n".join(md) + "\n")
    print((RESULTS / "learned_policy.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "select", "test", "holdout", "report"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.step == "extract":
        extract(args.workers)
    elif args.step == "select":
        run_select()
    elif args.step in ("test", "holdout"):
        run_period(args.step, args.workers)
        report()
    else:
        report()


if __name__ == "__main__":
    main()
