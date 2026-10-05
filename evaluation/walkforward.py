"""E4: walk-forward evaluation, the untouched play-off holdout and significance tests.

    python -m evaluation.walkforward                  # runs everything (about 30-60 minutes, offline)
    python -m evaluation.walkforward --report-only    # re-score the saved runs in runs/walkforward/ and runs/holdout/

Walk-forward: monthly windows over the 2025-26 season (Nov .. 12 Apr, then the
play-in and play-offs from 13 Apr). Before each window the M4 win model is
retrained in memory on every game before the window start; M2/M3 stay as
loaded (they do not affect trades). Each setup is one replay over the whole
season, so the learning agent carries its notebook forward and its reviewer
and gate only use earlier days. Setups: full agent (anchor + learning), agent
with no learning, raw model with no agent, never trade.

Holdout: every game after 12 Apr 2026 (play-in and play-offs), which no one had
looked at. Default models (trained before 1 Feb), the full agent starts from the
test-period notebook (runs/results/test-full) and keeps learning.

Writes walkforward.csv, walkforward_summary.md, walkforward.png, holdout.csv,
holdout.md and significance.md to evaluation/results/.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import REPS, SEED, bootstrap, day_table, paired

RESULTS = Path(__file__).resolve().parent / "results"
WINDOWS = [("Nov", "2025-11-01", "2025-11-30"), ("Dec", "2025-12-01", "2025-12-31"),
           ("Jan", "2026-01-01", "2026-01-31"), ("Feb", "2026-02-01", "2026-02-28"),
           ("Mar", "2026-03-01", "2026-03-31"), ("Apr 1-12", "2026-04-01", "2026-04-12"),
           ("Play-offs", "2026-04-13", "2026-06-14")]
SEASON_START = "2025-10-21"      # a model trained before the season serves gate backtests on October days
REGULAR_END = "2026-04-12"
HOLDOUT = ("2026-04-13", "2026-06-14")
SETUPS = {"full": "Full agent: anchor + learning", "no_learning": "Agent: anchor, no learning",
          "raw": "Raw model, no agent", "never": "Never trade"}
PAIRS = [("full", "never"), ("no_learning", "never"), ("raw", "never"), ("full", "no_learning"), ("full", "raw")]
TEST_RUNS = {"full": "test-full", "no_learning": "test-no-learning", "raw": "test-no-agent"}


class WalkForwardForecaster:
    """MarketAgent forecaster that uses, for each game, the M4 model retrained before that game's window."""

    def __init__(self, win_models: dict, play=None, points=None):
        from forecast.api import Forecaster
        self.starts = sorted(win_models)
        self.by_start = {s: Forecaster(win_models[s], play, points) for s in self.starts}
        self.name = f"walk-forward {self.by_start[self.starts[0]].name}"

    def model_for(self, date: str):
        key = max([s for s in self.starts if s <= date], default=self.starts[0])
        return self.by_start[key]

    def __call__(self, view, game, markets, overrides):
        return self.model_for(game.date)(view, game, markets, overrides)


def train_win_models(starts) -> tuple:
    """M4 fit on all games dated before each start; rows built once (features only use earlier games)."""
    from forecast.baselines import load_source
    from forecast.history import History
    from forecast.win import WinModel, training_rows

    player_games, games = load_source("frozen")
    rows = training_rows(History(player_games, games), games).merge(games[["game_id", "date"]], on="game_id")
    models, info = {}, []
    for s in starts:
        train = rows[rows.date < s]
        models[s] = WinModel().fit(train)
        info.append({"window_start": s, "training_games": len(train), **models[s].coefficients()})
    return models, pd.DataFrame(info)


def _run(job: dict) -> dict:
    """One setup over one span, in its own process. Returns the fills and timing."""
    import copy

    from agents.graph import MarketAgent, save_run
    from agents.notebook import Notebook
    from evaluation.ablations import plain_policy
    from forecast.api import Forecaster
    from replay import Replay, load_tables

    t0 = time.time()
    tables = load_tables()
    base = Forecaster.load()
    forecaster = (WalkForwardForecaster(pickle.loads(job["models"]), base.play_model, base.points_model)
                  if job.get("models") else base)
    if job["setup"] == "raw":
        agent, policy, hook = None, plain_policy(forecaster), None
    else:
        notebook = Notebook.load(job["notebook"]) if job.get("notebook") else Notebook()
        agent = MarketAgent(notebook=copy.deepcopy(notebook), forecaster=forecaster, learn=job["setup"] == "full")
        policy, hook = agent.policy, (agent.on_day_end if agent.learn else None)
    decisions, fills = Replay(tables, policy, on_day_end=hook).run(job["start"], job["end"])
    save_run(job["name"], decisions, fills, agent)
    return {"name": job["name"], "fills": len(fills), "seconds": time.time() - t0}


def run_all(workers: int):
    t0 = time.time()
    starts = [SEASON_START] + [w[1] for w in WINDOWS]
    print(f"training M4 for {len(starts)} window starts")
    models, info = train_win_models(starts)
    info.to_csv(RESULTS / "walkforward_models.csv", index=False)
    print(info[["window_start", "training_games", "home_intercept", "rating_diff"]].round(3).to_string(index=False))
    blob = pickle.dumps(models)
    jobs = [{"name": f"walkforward/{s}", "setup": s, "start": WINDOWS[0][1], "end": WINDOWS[-1][2], "models": blob}
            for s in ("full", "no_learning", "raw")]
    notebook = Path(__file__).resolve().parent.parent / "runs" / "results" / "test-full" / "notebook.json"
    jobs += [{"name": f"holdout/{s}", "setup": s, "start": HOLDOUT[0], "end": HOLDOUT[1],
              "notebook": notebook if s == "full" else None} for s in ("full", "no_learning", "raw")]
    from evaluation.walkforward import _run as run_job  # picklable under "python -m"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(run_job, jobs):
            print(f"  {r['name']}: {r['fills']} fills in {r['seconds'] / 60:.1f} min", flush=True)
    print(f"runs finished in {(time.time() - t0) / 60:.1f} min")


# ---------------- reporting ----------------

def market_days(tables: dict, start: str, end: str) -> list:
    g = tables["games"].dropna(subset=["tip_time"])
    g = g[g.game_id.isin(tables["markets"].game_id) & (g.date >= start) & (g.date <= end)]
    return sorted(g.date.unique())


def load_fills(folder: Path) -> pd.DataFrame:
    path = folder / "fills.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def tables_by_setup(runs: dict, games: pd.DataFrame, days: list) -> dict:
    """Setup -> day table over `days`; runs maps setup -> run folder (missing setup = never trade)."""
    out = {}
    for s in SETUPS:
        f = load_fills(runs[s]) if s in runs else pd.DataFrame()
        if len(f):
            f = f[f.game_id.isin(games.loc[games.date.isin(days), "game_id"])]
        out[s] = day_table(f, games, days)
    return out


def summary_row(label: str, setup: str, table: pd.DataFrame) -> dict:
    b = bootstrap(table)
    row = {"window": label, "setup": SETUPS[setup], "days": len(table), "trades": int(table.trades.sum())}
    for m in b.index:
        row.update({m: b.loc[m, "estimate"], f"{m}_lo": b.loc[m, "ci_low"], f"{m}_hi": b.loc[m, "ci_high"]})
    row["p_clv_gt_0"] = b.loc["mean_clv", "p_gt_0"]
    row["p_pnl_gt_0"] = b.loc["pnl", "p_gt_0"]
    return row


def fmt_ci(est, lo, hi, kind) -> str:
    if pd.isna(est):
        return "–"
    if kind == "clv":
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    if kind == "roi":
        return f"{est:+.1%} [{lo:+.1%}, {hi:+.1%}]"
    return f"{est:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"


def ci_table(rows: pd.DataFrame) -> str:
    lines = ["| Window | Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | ROI [95% CI] "
             "| p (CLV > 0) |", "|" + " --- |" * 8]
    for r in rows.itertuples():
        p = "–" if pd.isna(r.p_clv_gt_0) else f"{r.p_clv_gt_0:.3f}"
        lines.append(f"| {r.window} | {r.setup} | {r.trades} | {fmt_ci(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} "
                     f"| {fmt_ci(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, '$')} "
                     f"| {fmt_ci(r.pnl, r.pnl_lo, r.pnl_hi, '$')} | {fmt_ci(r.roi, r.roi_lo, r.roi_hi, 'roi')} | {p} |")
    return "\n".join(lines) + "\n"


def pair_rows(label: str, tabs: dict) -> list:
    rows = []
    for a, b in PAIRS:
        d = paired(tabs[a], tabs[b])
        for m in ("mean_clv", "clv_dollars", "pnl"):
            if b == "never" and m == "mean_clv":
                continue
            r = d.loc[m]
            rows.append({"period": label, "comparison": f"{SETUPS[a]} − {SETUPS[b]}", "metric": m,
                         "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_one_sided": r.p_a_gt_b})
    return rows


def chart(monthly: pd.DataFrame, totals: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [w[0] for w in WINDOWS]
    setups = list(SETUPS.values())
    colors = dict(zip(setups, ["#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [3, 1.3]})
    width = 0.8 / len(setups)
    for row, (metric, title) in enumerate((("pnl", "P&L after fees ($)"), ("clv_dollars", "closing-line value ($)"))):
        ax = axes[row, 0]
        for i, s in enumerate(setups):
            v = monthly[monthly.setup == s].set_index("window").reindex(labels)[metric].fillna(0)
            ax.bar(np.arange(len(labels)) + (i - 1.5) * width, v, width, label=s, color=colors[s])
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel(title)
        ax.set_title(f"Walk-forward by month: {title}")
        ax = axes[row, 1]
        t = totals.set_index("setup").reindex(setups)
        err = np.vstack([t[metric] - t[f"{metric}_lo"], t[f"{metric}_hi"] - t[metric]])
        ax.bar(range(len(setups)), t[metric], color=[colors[s] for s in setups], yerr=err, capsize=4)
        ax.set_xticks(range(len(setups)), ["full", "no learn", "raw", "never"])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title("Season total incl. play-offs\n95% CI, day-clustered bootstrap")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report():
    from replay import RUNS, load_tables

    tables = load_tables()
    games = tables["games"]
    wf = {s: RUNS / "walkforward" / s for s in ("full", "no_learning", "raw")}
    rows = []
    for label, start, end in WINDOWS:
        tabs = tables_by_setup(wf, games, market_days(tables, start, end))
        rows += [summary_row(label, s, t) for s, t in tabs.items()]
    monthly = pd.DataFrame(rows)
    spans = {"Season, Nov–12 Apr": (WINDOWS[0][1], REGULAR_END),
             "Season incl. play-offs": (WINDOWS[0][1], WINDOWS[-1][2])}
    total_rows, sig = [], []
    for label, (start, end) in spans.items():
        tabs = tables_by_setup(wf, games, market_days(tables, start, end))
        total_rows += [summary_row(label, s, t) for s, t in tabs.items()]
        sig += pair_rows(f"Walk-forward, {label}", tabs)
    totals = pd.DataFrame(total_rows)
    pd.concat([monthly, totals]).to_csv(RESULTS / "walkforward.csv", index=False)

    models = pd.read_csv(RESULTS / "walkforward_models.csv") if (RESULTS / "walkforward_models.csv").exists() else None
    md = ["# Walk-forward evaluation, 2025-26 season", "",
          "M4 retrained before each window on every earlier game; one replay per setup over the season "
          "(the learning agent starts with an empty notebook on 1 Nov and carries it forward). "
          f"Stake $20 per order, offline rules. 95% CIs and one-sided p-values from a day-clustered bootstrap "
          f"({REPS} replicates, seed {SEED}). Never trade is $0 by construction.", "",
          "## Season totals", "", ci_table(totals), "## By month", "", ci_table(monthly)]
    if models is not None:
        md += ["## Training games per window", "",
               "| Window start | Training games |", "| --- | --- |"]
        md += [f"| {r.window_start} | {r.training_games} |" for r in models.itertuples()]
    (RESULTS / "walkforward_summary.md").write_text("\n".join(md) + "\n")
    chart(monthly, totals[totals.window == "Season incl. play-offs"], RESULTS / "walkforward.png")

    days = market_days(tables, *HOLDOUT)
    htabs = tables_by_setup({s: RUNS / "holdout" / s for s in ("full", "no_learning", "raw")}, games, days)
    holdout = pd.DataFrame([summary_row("Holdout 13 Apr–14 Jun", s, t) for s, t in htabs.items()])
    holdout.to_csv(RESULTS / "holdout.csv", index=False)
    news_end = tables["news"].published_at.max()
    (RESULTS / "holdout.md").write_text("\n".join([
        "# Holdout: play-in and play-offs, 13 Apr – 14 Jun 2026", "",
        f"{len(days)} game-days, {games.date.isin(days).sum()} games with Kalshi markets. Not looked at before this "
        "run; each setup run once. Default models (M4 trained on games before 1 Feb 2026). The full agent starts "
        "from the test-period notebook and keeps learning. "
        f"Inactive-list news in the frozen data ends {news_end:%d %b %Y}; after that the agents see no news.", "",
        ci_table(holdout)]) + "\n")
    sig += pair_rows("Holdout 13 Apr–14 Jun", htabs)

    test_days = market_days(tables, "2026-02-01", REGULAR_END)
    ttabs = tables_by_setup({s: RUNS / "results" / r for s, r in TEST_RUNS.items()}, games, test_days)
    test = pd.DataFrame([summary_row("Original test 1 Feb–12 Apr", s, t) for s, t in ttabs.items()])
    sig += pair_rows("Original test 1 Feb–12 Apr (not a clean holdout)", ttabs)
    write_significance(pd.DataFrame(sig), pd.concat([totals, holdout, test]))
    print((RESULTS / "walkforward_summary.md").read_text())
    print((RESULTS / "holdout.md").read_text())
    print((RESULTS / "significance.md").read_text())


def write_significance(sig: pd.DataFrame, solo: pd.DataFrame):
    sig.to_csv(RESULTS / "significance.csv", index=False)
    lines = ["# Significance tests", "",
             f"Day-clustered bootstrap, {REPS} replicates, seed {SEED}. One-sided p-values: for single setups "
             "H1 is \"mean CLV > 0\"; for comparisons H1 is \"first setup > second setup\". A comparison against "
             "never trade is the setup's own total (never trade is $0).", "",
             "CLV is measured against the mid at tip, but orders fill at the ask, so a trader with no edge over "
             "the closing price shows CLV of about minus half the spread (the median half-spread in the last six "
             "hours before tip is 0.005).", "",
             "## Is mean CLV above zero?", "",
             "| Period | Setup | Trades | Mean CLV [95% CI] | p (CLV > 0) |", "| --- | --- | --- | --- | --- |"]
    for r in solo[solo.setup != SETUPS["never"]].itertuples():
        p = "–" if pd.isna(r.p_clv_gt_0) else f"{r.p_clv_gt_0:.3f}"
        lines.append(f"| {r.window} | {r.setup} | {r.trades} | {fmt_ci(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, 'clv')} | {p} |")
    lines += ["", "## Paired comparisons on the same game-days", "",
              "| Period | Comparison | Metric | Difference [95% CI] | p (one-sided) |", "| --- | --- | --- | --- | --- |"]
    for r in sig.itertuples():
        kind = "clv" if r.metric == "mean_clv" else "$"
        name = {"mean_clv": "mean CLV", "clv_dollars": "CLV $", "pnl": "P&L"}[r.metric]
        p = "–" if pd.isna(r.p_one_sided) else f"{r.p_one_sided:.3f}"
        lines.append(f"| {r.period} | {r.comparison} | {name} | {fmt_ci(r.estimate, r.ci_low, r.ci_high, kind)} "
                     f"| {p} |")
    (RESULTS / "significance.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true", help="re-score saved runs without replaying")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_all(args.workers)
    report()


if __name__ == "__main__":
    main()
