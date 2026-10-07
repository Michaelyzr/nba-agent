"""E4: day-clustered bootstrap for trading metrics, paired setup comparisons and a CLV > 0 test.

Trades on the same night share news, prices and outcomes, so the unit of
resampling is the game-day, not the trade. A setup is first reduced to one row
per game-day (days with no trades are kept as zeros, so "never trade" and
setups that skip a night are compared on the same days). Each bootstrap
replicate draws the days with replacement and recomputes:

    mean_clv      mean closing-line value per contract over trades (the scorer's mean_clv)
    clv_dollars   sum of CLV x contracts
    pnl           profit after fees
    roi           pnl / dollars staked

Confidence intervals are percentile intervals. One-sided p-values use the
bootstrap distribution shifted to the null (estimate 0), with the +1
correction so a p-value is never exactly 0. Everything is deterministic for a
fixed seed.
"""
import numpy as np
import pandas as pd

REPS = 2000
SEED = 7606
METRICS = ("mean_clv", "clv_dollars", "pnl", "roi")
DAY_COLS = ["trades", "clv_sum", "clv_dollars", "pnl", "staked"]


def day_table(fills: pd.DataFrame, games: pd.DataFrame, days) -> pd.DataFrame:
    """One row per game-day in `days`: trades, sum of per-contract CLV, CLV dollars, P&L, staked."""
    days = pd.Index(sorted(set(days)), name="date")
    if fills is None or fills.empty:
        return pd.DataFrame(0.0, index=days, columns=DAY_COLS)
    f = fills.merge(games[["game_id", "date"]], on="game_id", how="left")
    f = f.assign(trades=1.0, clv_sum=f.clv.fillna(0.0), clv_dollars=(f.clv * f.contracts).fillna(0.0),
                 pnl=f.pnl.fillna(0.0), staked=f.price * f.contracts)
    out = f.groupby("date")[DAY_COLS].sum().reindex(days, fill_value=0.0)
    missing = set(f.date) - set(days)
    if missing:
        raise ValueError(f"fills on days outside the window: {sorted(missing)[:3]}")
    return out.astype(float)


def _stats(sums: np.ndarray) -> dict:
    """Metrics from column sums in DAY_COLS order; sums may be (5,) or (reps, 5)."""
    trades, clv_sum, clv_dollars, pnl, staked = np.moveaxis(np.asarray(sums, float), -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"mean_clv": np.where(trades > 0, clv_sum / np.where(trades > 0, trades, 1), np.nan),
                "clv_dollars": clv_dollars, "pnl": pnl,
                "roi": np.where(staked > 0, pnl / np.where(staked > 0, staked, 1), np.nan)}


def _draws(n_days: int, reps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_days, size=(reps, n_days))


def _boot_sums(table: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return table[DAY_COLS].to_numpy(float)[idx].sum(axis=1)


def _ci(values: np.ndarray, level=0.95) -> tuple:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.nan, np.nan
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def p_greater(estimate: float, draws: np.ndarray) -> float:
    """One-sided bootstrap p-value for H1: true value > 0 (draws shifted so the null holds)."""
    d = draws[~np.isnan(draws)]
    if len(d) == 0 or np.isnan(estimate) or (estimate == 0 and not d.any()):
        return np.nan
    return float((1 + np.sum(d - estimate >= estimate)) / (len(d) + 1))


def bootstrap(table: pd.DataFrame, reps=REPS, seed=SEED, level=0.95) -> pd.DataFrame:
    """Estimate, CI and one-sided p (value > 0) for every metric of one setup."""
    point = _stats(table[DAY_COLS].sum().to_numpy())
    boot = _stats(_boot_sums(table, _draws(len(table), reps, seed))) if len(table) else {m: np.array([]) for m in METRICS}
    rows = []
    for m in METRICS:
        lo, hi = _ci(boot[m], level)
        rows.append({"metric": m, "estimate": float(point[m]), "ci_low": lo, "ci_high": hi,
                     "p_gt_0": p_greater(float(point[m]), boot[m])})
    return pd.DataFrame(rows).set_index("metric")


def paired(a: pd.DataFrame, b: pd.DataFrame, reps=REPS, seed=SEED, level=0.95) -> pd.DataFrame:
    """a minus b on the same game-days: estimate, CI and one-sided p for H1: a > b.

    Both tables must cover the same days; each replicate uses the same resampled
    days for both setups. A setup with no trades counts as mean CLV 0 and ROI 0.
    """
    if not a.index.equals(b.index):
        raise ValueError("paired comparison needs the same game-days for both setups")
    idx = _draws(len(a), reps, seed)

    def filled(stats):
        return {m: np.nan_to_num(v, nan=0.0) for m, v in stats.items()}

    pa, pb = filled(_stats(a[DAY_COLS].sum().to_numpy())), filled(_stats(b[DAY_COLS].sum().to_numpy()))
    ba, bb = filled(_stats(_boot_sums(a, idx))), filled(_stats(_boot_sums(b, idx)))
    rows = []
    for m in METRICS:
        diff, draws = float(pa[m] - pb[m]), ba[m] - bb[m]
        lo, hi = _ci(draws, level)
        rows.append({"metric": m, "estimate": diff, "ci_low": lo, "ci_high": hi, "p_a_gt_b": p_greater(diff, draws)})
    return pd.DataFrame(rows).set_index("metric")


def clv_positive_test(table: pd.DataFrame, reps=REPS, seed=SEED) -> dict:
    """One-sided test that mean CLV per contract is above zero."""
    r = bootstrap(table, reps, seed).loc["mean_clv"]
    return {"mean_clv": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_gt_0,
            "trades": int(table.trades.sum()), "days": len(table)}
