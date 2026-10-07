"""M4 recalibration: does a calibrated win model forecast better, and does it change the trading results?

    python -m evaluation.m4_calibration                    # fit, score, replay test period + walk-forward (~40 min)
    python -m evaluation.m4_calibration --skip-walkforward # test period only
    python -m evaluation.m4_calibration --report-only      # rescore saved runs in runs/m4cal/
    (runs already saved in runs/m4cal/ are kept; delete the folder to replay from scratch)

1. Out-of-sample M4 predictions for every month from Nov 2023 (rolling origin:
   M4 refit before each month on all earlier games). Calibrators (Platt scaling
   and isotonic regression) are fit on those predictions for games before
   1 Feb 2026 only. Sensitivity: Platt fit on January 2026 alone (M4 fit before
   1 Jan, scored on January), the chronological split.
2. On the 501 test games (evaluation/results/m4_games.csv): Brier, log loss and
   reliability bins for raw and calibrated M4 against the market, plus the
   agent's anchored estimate (24 h market mid + M4 news shift) with raw and
   calibrated shifts. Day-clustered bootstrap CIs for Brier differences.
3. Replays with the calibrated (Platt) M4 wrapped in forecast.api.Forecaster;
   default code paths untouched. Test period 1 Feb - 12 Apr: raw model, anchor
   agent without learning, anchor agent with learning (development notebook
   learned Nov - Jan with the same calibrated model, as for the existing
   test-full run). Walk-forward Nov - 12 Apr: M4 retrained monthly and a Platt
   calibrator refit before each window on out-of-sample predictions before it.
   Compared with the existing uncalibrated runs and never-trade
   (evaluation/stats.py, day-clustered bootstrap, paired on the same days).

Writes m4_calibration.md, m4_calibration.csv (forecast scores),
m4_calibration_trading.csv, m4_calibration_reliability.csv and
m4_calibration.png to evaluation/results/.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import REPS, SEED, bootstrap, day_table, paired
from forecast.calibrate import Calibrator, CalibratedWinModel, fit_before, month_starts, out_of_sample

RESULTS = Path(__file__).resolve().parent / "results"
SPLIT = "2026-02-01"
TEST = ("2026-02-01", "2026-04-12")
DEV = ("2025-11-01", "2026-01-31")
WF = ("2025-11-01", "2026-04-12")
WF_STARTS = ["2025-10-21", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
BINS = np.linspace(0, 1, 11)
ANCHOR_LEAD = pd.Timedelta(hours=24)


# ---------------- forecasts ----------------

def oos_predictions():
    from forecast.baselines import load_source
    from forecast.history import History
    from forecast.win import training_rows

    player_games, games = load_source("frozen")
    rows = training_rows(History(player_games, games), games).merge(games[["game_id", "date"]], on="game_id")
    return rows, out_of_sample(rows, month_starts("2023-11-01", "2026-06-01"), min_train=300)


def anchor_mids(t: pd.DataFrame, tables: dict) -> np.ndarray:
    """Home mid 24 h before tip (or the earliest quote), as MarketAgent._anchor_mid."""
    from evaluation.consumer_fairness import Quotes
    q = Quotes(tables["prices"])
    games = tables["games"].set_index("game_id")
    m = tables["markets"]
    home = m[(m.kind == "game")].merge(tables["games"][["game_id", "home_team"]], on="game_id")
    home = home[home.team == home.home_team].set_index("game_id").market_ticker
    out = []
    for gid in t.game_id:
        v = q.at(home.get(gid), pd.Timestamp(games.loc[gid, "tip_time"]) - ANCHOR_LEAD, earliest=True) \
            if gid in home.index else None
        out.append((v[0] + v[1]) / 2 if v else np.nan)
    return np.array(out)


def clip(p):
    return np.clip(np.asarray(p, float), 0.02, 0.98)


def brier(p, y):
    return float(np.mean((np.asarray(p) - y) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_diff_ci(a, b, y, days, reps=REPS, seed=SEED) -> tuple:
    """Brier(a) - Brier(b) with a 95% CI that resamples game-days."""
    d = pd.DataFrame({"day": days, "x": (a - y) ** 2 - (b - y) ** 2}).groupby("day").x.agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (reps, len(s)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def phase_of(dates) -> pd.Series:
    m = pd.to_datetime(pd.Series(dates)).dt.month
    return pd.Series(np.select([m.isin([2, 3, 4]), m.isin([10, 11, 12, 1])], ["Feb–Apr", "Oct–Jan"], "play-offs"),
                     index=m.index)


def late_season(oos: pd.DataFrame, start: str) -> Calibrator:
    """Exploratory: Platt fit only on out-of-sample Feb-Apr games of earlier seasons (before `start`)."""
    d = oos[(oos.date < start) & (phase_of(oos.date).to_numpy() == "Feb–Apr")]
    return Calibrator("platt").fit(d.p_raw, d.home_win)


def phase_table(oos: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample Platt slope and Brier of M4 by season and phase (slope > 1: M4 too close to 50%)."""
    d = oos.assign(phase=phase_of(oos.date).to_numpy(),
                   season=[f"{y}-{(y + 1) % 100:02d}" for y in
                           (pd.to_datetime(oos.date).dt.year - (pd.to_datetime(oos.date).dt.month < 9)).to_numpy()])
    rows = []
    for (s, p), g in d.groupby(["season", "phase"]):
        if len(g) >= 100:
            rows.append({"season": s, "phase": p, "games": len(g), "platt_slope":
                         Calibrator("platt").fit(g.p_raw, g.home_win).params()["slope"],
                         "brier": brier(g.p_raw, g.home_win.to_numpy()), "test_period": s == "2025-26" and p == "Feb–Apr"})
    return pd.DataFrame(rows)


def predictors(t: pd.DataFrame, cals: dict) -> dict:
    out = {"M4 raw, absences known": t.m4_after.to_numpy()}
    for name, c in cals.items():
        out[f"M4 {name}, absences known"] = c(t.m4_after)
    out["Anchor estimate, raw M4 shift"] = clip(t.anchor + t.m4_after - t.m4_before)
    for name, c in cals.items():
        out[f"Anchor estimate, {name} M4 shift"] = clip(t.anchor + c(t.m4_after) - c(t.m4_before))
    out["Market 24 h before tip"] = t.anchor.to_numpy()
    out["Market 1 h before tip"] = t.market_before.to_numpy()
    out["Market at tip"] = t.market_tip.to_numpy()
    return out


def forecast_scores(t: pd.DataFrame, preds: dict) -> pd.DataFrame:
    y, days = t.home_win.to_numpy(), t.date.to_numpy()
    raw, mkt = preds["M4 raw, absences known"], preds["Market 1 h before tip"]
    rows = []
    for name, p in preds.items():
        d_raw, d_mkt = brier_diff_ci(p, raw, y, days), brier_diff_ci(p, mkt, y, days)
        rows.append({"predictor": name, "games": len(y), "brier": brier(p, y), "log_loss": log_loss(p, y),
                     "accuracy": float(np.mean((p > 0.5) == y)),
                     "brier_minus_raw_m4": d_raw[0], "brier_minus_raw_m4_lo": d_raw[1], "brier_minus_raw_m4_hi": d_raw[2],
                     "brier_minus_market_1h": d_mkt[0], "brier_minus_market_1h_lo": d_mkt[1],
                     "brier_minus_market_1h_hi": d_mkt[2]})
    return pd.DataFrame(rows)


def reliability(t: pd.DataFrame, preds: dict, names) -> pd.DataFrame:
    y = t.home_win.to_numpy()
    rows = []
    for name in names:
        p = preds[name]
        b = pd.cut(p, BINS, include_lowest=True)
        g = pd.DataFrame({"p": p, "y": y, "bin": b}).groupby("bin", observed=True).agg(
            games=("y", "size"), mean_pred=("p", "mean"), win_rate=("y", "mean")).reset_index()
        rows.append(g.assign(predictor=name, bin=g.bin.astype(str)))
    return pd.concat(rows, ignore_index=True)


# ---------------- replays ----------------

def _job(job: dict) -> dict:
    """One replay setup in its own process (picklable for ProcessPoolExecutor)."""
    import copy

    from agents.graph import MarketAgent, save_run
    from agents.notebook import Notebook
    from evaluation.ablations import plain_policy
    from evaluation.walkforward import WalkForwardForecaster
    from forecast.api import Forecaster
    from replay import Replay, load_tables

    t0 = time.time()
    from replay import RUNS
    if (RUNS / job["name"] / "fills.parquet").exists():      # resume: keep runs already saved
        return {"name": job["name"], "fills": len(pd.read_parquet(RUNS / job["name"] / "fills.parquet")), "minutes": 0.0}
    tables = load_tables()
    base = Forecaster.load()
    win = pickle.loads(job["win"])
    forecaster = (WalkForwardForecaster(win, base.play_model, base.points_model) if isinstance(win, dict)
                  else Forecaster(win, base.play_model, base.points_model))
    notebook = Notebook()
    if job.get("dev"):
        dev = MarketAgent(notebook=Notebook(), forecaster=forecaster, learn=True)
        d, f = Replay(tables, dev.policy, on_day_end=dev.on_day_end).run(*job["dev"])
        save_run(job["name"] + "-dev", d, f, dev)
        notebook = copy.deepcopy(dev.notebook)
    if job["setup"] == "raw":
        agent, policy, hook = None, plain_policy(forecaster), None
    else:
        agent = MarketAgent(notebook=notebook, forecaster=forecaster, learn=job["setup"] == "full")
        policy, hook = agent.policy, (agent.on_day_end if agent.learn else None)
    decisions, fills = Replay(tables, policy, on_day_end=hook).run(*job["period"])
    save_run(job["name"], decisions, fills, agent)
    return {"name": job["name"], "fills": len(fills), "minutes": (time.time() - t0) / 60}


def run_replays(win_test, wf_models: dict | None, workers: int, extra: dict):
    blob = pickle.dumps(win_test)
    jobs = [{"name": "m4cal/test-raw", "setup": "raw", "period": TEST, "win": blob},
            {"name": "m4cal/test-no-learning", "setup": "no_learning", "period": TEST, "win": blob},
            {"name": "m4cal/test-full", "setup": "full", "period": TEST, "dev": DEV, "win": blob}]
    for name, w in extra.items():
        jobs += [{"name": f"m4cal/test-{name}-raw", "setup": "raw", "period": TEST, "win": pickle.dumps(w)},
                 {"name": f"m4cal/test-{name}-no-learning", "setup": "no_learning", "period": TEST,
                  "win": pickle.dumps(w)}]
    if wf_models is not None:
        wblob = pickle.dumps(wf_models)
        jobs = [{"name": f"m4cal/walkforward/{s}", "setup": s, "period": WF, "win": wblob}
                for s in ("full", "no_learning", "raw")] + jobs
    from evaluation.m4_calibration import _job as job_fn     # picklable under "python -m"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(job_fn, jobs):
            print(f"  {r['name']}: {r['fills']} fills in {r['minutes']:.1f} min", flush=True)


def trading_report(tables: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from evaluation.walkforward import load_fills, market_days
    from replay import RUNS

    games = tables["games"]
    raw, anchor, full = "Raw model, no agent", "Anchor agent, no learning", "Anchor agent + learning"
    specs = {"Test 1 Feb–12 Apr": (TEST, {      # setup: (existing raw-M4 run, [(M4 variant, calibrated run)])
        raw: ("results/test-no-agent", [("Platt", "m4cal/test-raw"), ("isotonic", "m4cal/test-isotonic-raw"),
                                        ("Feb–Apr Platt (exploratory)", "m4cal/test-late-raw")]),
        anchor: ("results/test-no-learning", [("Platt", "m4cal/test-no-learning"),
                                              ("isotonic", "m4cal/test-isotonic-no-learning"),
                                              ("Feb–Apr Platt (exploratory)", "m4cal/test-late-no-learning")]),
        full: ("results/test-full", [("Platt", "m4cal/test-full")])}),
             "Walk-forward 1 Nov–12 Apr": (WF, {
        raw: ("walkforward/raw", [("Platt", "m4cal/walkforward/raw")]),
        anchor: ("walkforward/no_learning", [("Platt", "m4cal/walkforward/no_learning")]),
        full: ("walkforward/full", [("Platt", "m4cal/walkforward/full")])})}
    rows, pairs = [], []
    for period, ((start, end), setups) in specs.items():
        days = market_days(tables, start, end)
        in_days = games.loc[games.date.isin(days), "game_id"]

        def table(run):
            f = load_fills(RUNS / run)
            return day_table(f[f.game_id.isin(in_days)] if len(f) else f, games, days)

        def row(setup, model, tab):
            b = bootstrap(tab)
            return {"period": period, "setup": setup, "model": model, "trades": int(tab.trades.sum()),
                    **{f"{m}{s}": b.loc[m, c] for m in ("pnl", "mean_clv")
                       for s, c in (("", "estimate"), ("_lo", "ci_low"), ("_hi", "ci_high"))}}

        never = day_table(pd.DataFrame(), games, days)
        rows.append({"period": period, "setup": "Never trade", "model": "—", "trades": 0, "pnl": 0.0,
                     "pnl_lo": 0.0, "pnl_hi": 0.0, "mean_clv": np.nan, "mean_clv_lo": np.nan, "mean_clv_hi": np.nan})
        for setup, (old, variants) in setups.items():
            base = table(old) if (RUNS / old / "fills.parquet").exists() else None
            if base is not None:
                rows.append(row(setup, "raw M4 (existing run)", base))
            for model, run in variants:
                if not (RUNS / run / "fills.parquet").exists():
                    continue
                tab = table(run)
                rows.append(row(setup, f"{model} M4", tab))
                for other, ref in (("never trade", never), ("same setup with raw M4", base)):
                    if ref is not None:
                        p = paired(tab, ref).loc["pnl"]
                        pairs.append({"period": period, "setup": setup, "comparison": f"{model} M4 − {other}",
                                      "pnl_diff": p.estimate, "lo": p.ci_low, "hi": p.ci_high,
                                      "p_one_sided": p.p_a_gt_b})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


# ---------------- report ----------------

def figure(rel: pd.DataFrame, scores: pd.DataFrame, oos: pd.DataFrame, cals: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    colors = {"M4 raw, absences known": "#91a7ff", "M4 Platt, absences known": "#3b5bdb",
              "M4 isotonic, absences known": "#2b8a3e", "M4 Feb–Apr Platt, absences known": "#ae3ec9",
              "Market 1 h before tip": "#e8590c"}
    for name, c in colors.items():
        g = rel[(rel.predictor == name) & (rel.games >= 5)]
        a.plot(g.mean_pred, g.win_rate, "o-", color=c, label=name, ms=4)
    a.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    a.set(xlabel="predicted home win probability", ylabel="actual home win rate", xlim=(0, 1), ylim=(0, 1),
          title="Reliability on 501 test games (bins with 5+ games)")
    a.legend(fontsize=8, loc="upper left")

    a = ax[1]
    x = np.linspace(0.02, 0.98, 200)
    for name, c in cals.items():
        a.plot(x, c(x), label=name)
    b = pd.cut(oos[oos.date < SPLIT].p_raw, BINS)
    g = oos[oos.date < SPLIT].groupby(b, observed=True).agg(p=("p_raw", "mean"), y=("home_win", "mean"))
    a.plot(g.p, g.y, "ko", ms=4, label="binned out-of-sample win rate")
    a.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    a.set(xlabel="raw M4 probability", ylabel="calibrated probability", title="Calibration maps (pre-test data only)")
    a.legend(fontsize=8)

    a = ax[2]
    s = scores.sort_values("brier", ascending=False)
    err = np.vstack([s.brier_minus_market_1h - s.brier_minus_market_1h_lo,
                     s.brier_minus_market_1h_hi - s.brier_minus_market_1h])
    a.barh(s.predictor, s.brier_minus_market_1h, xerr=err, capsize=3,
           color=["#e8590c" if "Market" in p else "#3b5bdb" for p in s.predictor])
    a.axvline(0, color="black", lw=0.8)
    a.set(xlabel="Brier minus market 1 h before tip (95% CI, game-days resampled)", title="Brier vs the market")
    a.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(e, lo, hi, d=4):
    return "—" if pd.isna(e) else f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def markdown(scores, cals, oos, sens, phases, trade, pairs) -> str:
    n_cal = int((oos.date < SPLIT).sum())
    lines = ["# M4 recalibration", "",
             f"Calibrators fit on {n_cal} out-of-sample M4 predictions for games from Nov 2023 to 31 Jan 2026 "
             "(rolling origin: M4 refit before each month on all earlier games). Scored on the 501 test games "
             "(1 Feb – 12 Apr 2026) with the default M4 (trained on games before 1 Feb). "
             f"Platt: slope {cals['Platt'].params()['slope']:.2f}, intercept {cals['Platt'].params()['intercept']:+.2f} "
             "on the logit (slope > 1 stretches M4 away from 50%). Brier-difference CIs resample game-days "
             f"({REPS} replicates).", "",
             "## Forecast accuracy on the 501 test games", "",
             "| Predictor | Brier | Log loss | Accuracy | Brier − raw M4 [95% CI] | Brier − market 1 h [95% CI] |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in scores.itertuples():
        lines.append(f"| {r.predictor} | {r.brier:.4f} | {r.log_loss:.4f} | {r.accuracy:.1%} | "
                     f"{fmt(r.brier_minus_raw_m4, r.brier_minus_raw_m4_lo, r.brier_minus_raw_m4_hi)} | "
                     f"{fmt(r.brier_minus_market_1h, r.brier_minus_market_1h_lo, r.brier_minus_market_1h_hi)} |")
    lines += ["", f"Sensitivity (chronological split): Platt fit only on January 2026 (M4 fit before 1 Jan, "
                  f"{sens['games']} games) gives slope {sens['slope']:.2f} and test Brier {sens['brier']:.4f} "
                  f"for M4 with absences known. Diagnostic only (uses test outcomes): a Platt map fit on the test "
                  f"games themselves has slope {sens['oracle_slope']:.2f} and Brier {sens['oracle_brier']:.4f}, the "
                  "best any monotone-logit recalibration could do; it is still worse than the market.", "",
              "## Why the pooled calibrators do not help: M4's calibration changes with the season", "",
              "Out-of-sample Platt slope by season phase (slope > 1 means M4 is too close to 50%, < 1 too far). "
              "The \"Feb–Apr Platt\" calibrator above is fit on the earlier seasons' Feb–Apr rows only; it was "
              "chosen after the pooled calibrators failed, so treat it as exploratory.", "",
              "| Season | Phase | Games | Platt slope | Brier |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r.season} | {r.phase}{' (test period)' if r.test_period else ''} | {r.games} | "
              f"{r.platt_slope:.2f} | {r.brier:.4f} |" for r in phases.itertuples()]
    lines.append("")
    if len(trade):
        lines += ["## Trading with calibrated M4 (Platt unless marked)", "",
                  "Day-clustered bootstrap, evaluation/stats.py. Raw-M4 rows are the existing runs.", "",
                  "| Period | Setup | M4 | Trades | P&L after fees [95% CI] | Mean CLV [95% CI] |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in trade.itertuples():
            lines.append(f"| {r.period} | {r.setup} | {r.model} | {r.trades} | {fmt(r.pnl, r.pnl_lo, r.pnl_hi, 0)} "
                         f"| {fmt(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi)} |")
    if len(pairs):
        lines += ["", "## Paired P&L differences on the same game-days", "",
                  "| Period | Setup | Comparison | P&L difference [95% CI] | p (one-sided, > 0) |", "| --- | --- | --- | --- | --- |"]
        for r in pairs.itertuples():
            lines.append(f"| {r.period} | {r.setup} | {r.comparison} | {fmt(r.pnl_diff, r.lo, r.hi, 0)} | {r.p_one_sided:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    from replay import load_tables

    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-walkforward", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    t0 = time.time()
    rows, oos = oos_predictions()
    cals = {"Platt": fit_before(oos, SPLIT, "platt"), "isotonic": fit_before(oos, SPLIT, "isotonic"),
            "Feb–Apr Platt": late_season(oos, SPLIT)}
    phases = phase_table(oos)
    print(phases.round(3).to_string(index=False))
    print(f"out-of-sample predictions: {len(oos)} ({(oos.date < SPLIT).sum()} before {SPLIT}); "
          f"Platt {cals['Platt'].params()} [{time.time() - t0:.0f}s]")
    tables = load_tables()
    t = pd.read_csv(RESULTS / "m4_games.csv", dtype={"game_id": str})
    t["anchor"] = anchor_mids(t, tables)
    t = t.dropna(subset=["anchor"])
    preds = predictors(t, cals)
    scores = forecast_scores(t, preds)
    jan = oos[(oos.date >= "2026-01-01") & (oos.date < SPLIT)]
    jan_cal = Calibrator("platt").fit(jan.p_raw, jan.home_win)
    oracle = Calibrator("platt").fit(t.m4_after, t.home_win)
    sens = {"games": len(jan), "slope": jan_cal.params()["slope"],
            "brier": brier(jan_cal(t.m4_after), t.home_win.to_numpy()),
            "oracle_slope": oracle.params()["slope"], "oracle_brier": brier(oracle(t.m4_after), t.home_win.to_numpy())}
    rel = reliability(t, preds, ["M4 raw, absences known", "M4 Platt, absences known", "M4 isotonic, absences known",
                                 "M4 Feb–Apr Platt, absences known",
                                 "Anchor estimate, raw M4 shift", "Anchor estimate, Platt M4 shift",
                                 "Market 1 h before tip"])
    print(scores[["predictor", "brier", "log_loss", "brier_minus_raw_m4", "brier_minus_market_1h"]].round(4).to_string(index=False))
    print(sens)

    if not args.report_only:
        from forecast.api import Forecaster
        from forecast.win import WinModel
        base = Forecaster.load().win_model
        wf = None
        if not args.skip_walkforward:
            wf = {s: CalibratedWinModel(WinModel().fit(rows[rows.date < s]), fit_before(oos, s, "platt"))
                  for s in WF_STARTS}
        run_replays(CalibratedWinModel(base, cals["Platt"]), wf, args.workers,
                    {"isotonic": CalibratedWinModel(base, cals["isotonic"]),
                     "late": CalibratedWinModel(base, cals["Feb–Apr Platt"])})
    trade, pairs = trading_report(tables)
    args.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out / "m4_calibration.csv", index=False)
    rel.to_csv(args.out / "m4_calibration_reliability.csv", index=False)
    trade.to_csv(args.out / "m4_calibration_trading.csv", index=False)
    pairs.to_csv(args.out / "m4_calibration_pairs.csv", index=False)
    figure(rel, scores, oos, cals, args.out / "m4_calibration.png")
    phases.to_csv(args.out / "m4_calibration_phases.csv", index=False)
    (args.out / "m4_calibration.md").write_text(markdown(scores, cals, oos, sens, phases, trade, pairs))
    print((args.out / "m4_calibration.md").read_text())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
