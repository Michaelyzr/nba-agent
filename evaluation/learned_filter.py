"""Approach C: a learned trade filter (no LLM calls) vs the current LLM sceptic, keep-all, random and the oracle.

Two labelled pools, kept apart:
  sceptic set  the 182 proposals the deepseek-reasoner sceptic reviewed (Dec 2025 - Jan 2026 walk-forward window,
               Feb - Apr 2026 test window); the labelled-set code is Approach A's (exp/sceptic-balanced,
               evaluation/sceptic_balanced.py). Like-for-like with the current sceptic.
  anchor pool  the deterministic anchor agent's (arm A) candidate at every decision point of the same windows: its
               actual trade where it traded, else its best option, kept when the gap after fees > NEAR_MISS (so
               the 0.04 trades plus near misses), flagged with whether arm A traded it.

Features are rebuilt as of each decision time from the replay view with the models the runs used
(forecast.policy.decision_rows: anchor + M4 news shift; production M4 for Feb - Apr, the walk-forward refit in
runs/walkforward/2025-12-01 for Dec - Jan), plus schedule (back-to-back), players ruled out per side and the
six-hour price move. Nothing at or after tip-off enters a feature (checked: label columns are excluded by name,
and features recomputed on tables truncated at the decision time are identical). Labels come from the
counterfactual $20 fill at the decision time (Replay._fill / _settle): CLV = side's mid at tip - price paid.

Filters (thresholds chosen on the training window only, from out-of-fold day-grouped predictions):
  LR      L2 logistic regression on CLV > 0, standardised features, C = LR_C
  GBM     shallow gradient boosting on CLV > 0 (depth 2, 60 trees, lr 0.05, min leaf 20, L2 1.0)
  Ridge   ridge regression on CLV $ (alpha RIDGE_ALPHA), keep if predicted CLV $ >= tau
  rules   "skip if the market moved against the side" (fixed) and the best single feature with a tuned cut
LR / GBM / Ridge are trained on the anchor pool (more data); "LR, sceptic set + LLM features" is trained on the
sceptic set's own training window with the analyst's estimate minus price and minus the anchor model as extra inputs.

    python -m evaluation.learned_filter extract     # as-of option rows for both windows (~8 min, cached)
    python -m evaluation.learned_filter             # datasets, fits, evaluation, evaluation/results/learned_filter.*
"""
import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.llm_agent_eval import OUT, RESULTS, WINDOWS, load_frozen, load_traces
from evaluation.stats import REPS, SEED

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "runs" / "learned_filter"
STEM = "learned_filter"
STAKE = 20.0
MIN_EDGE = 0.04
NEAR_MISS = 0.02
REASONER = "deepseek-reasoner"
PERIODS = {"prev": ("2025-12-01", "2026-01-31"), "test": ("2026-02-01", "2026-04-12")}
MODEL_DIRS = {"prev": ROOT / "runs" / "walkforward" / "2025-12-01", "test": ROOT / "models"}
ANCHOR_RUNS = {"prev": OUT / "prev-anchor" / "anchor", "test": OUT / "test-full-anchor" / "anchor"}
WINDOW_NAMES = {"prev": "Dec 2025 - Jan 2026", "test": "Feb - Apr 2026 (test)", "both": "both windows"}
OTHER = {"prev": "test", "test": "prev"}
HEAD = re.compile(r"Game: (\w+) \(away\) at (\w+) \(home\), game date (\S+)\. Tip-off in (\d+) minutes")
FEATURES = ["side_price", "gap", "model_shift", "market_move", "move_6h", "news_age_hours", "has_news",
            "hours_to_tip", "is_fav", "home_side", "spread", "log_volume", "b2b_side", "b2b_opp", "n_out_side",
            "n_out_opp"]
LLM_FEATURES = ["llm_gap", "llm_minus_model"]
LABEL_COLS = {"close", "clv", "net_clv", "won", "outcome", "pnl", "clv_usd", "clv_usd_after_fee", "close_price",
              "contracts", "fee_paid", "y"}
LR_C = 0.1
RIDGE_ALPHA = 30.0
GBM = {"max_depth": 2, "learning_rate": 0.05, "max_iter": 60, "min_samples_leaf": 20, "l2_regularization": 1.0}
INNER_FOLDS = 5
MIN_KEPT = {"sceptic": 5, "anchor": 10}
RANDOM_DRAWS = 20000
AUC_REPS = 500


# ---------------- as-of option rows ----------------

def _month_spans(start: str, end: str) -> list:
    edges = pd.date_range(start, end, freq="MS").strftime("%Y-%m-%d").tolist()
    starts = sorted(set([start] + edges))
    ends = [(pd.Timestamp(s) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for s in starts[1:]] + [end]
    return list(zip(starts, ends))


def _extract_job(job: dict) -> pd.DataFrame:
    from forecast.api import Forecaster
    from forecast.policy import decision_rows
    part = WORK / "parts" / f"{job['window']}_{job['start']}_{job['end']}.parquet"
    if part.exists():
        return pd.read_parquet(part)
    t0 = time.time()
    rows = decision_rows(load_frozen(), Forecaster.load(Path(job["models"])), job["start"], job["end"], stake=STAKE)
    rows["window"] = job["window"]
    part.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(part, index=False)
    print(f"  {job['window']} {job['start']}..{job['end']}: {len(rows)} option rows in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def extract(workers: int = 5) -> pd.DataFrame:
    jobs = [{"window": w, "start": s, "end": e, "models": str(MODEL_DIRS[w])}
            for w, (a, b) in PERIODS.items() for s, e in _month_spans(a, b)]
    from evaluation.learned_filter import _extract_job as job_fn
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        rows = pd.concat(list(pool.map(job_fn, jobs)), ignore_index=True)
    rows.to_parquet(WORK / "option_rows.parquet", index=False)
    print(f"{len(rows)} option rows -> {WORK / 'option_rows.parquet'}")
    return rows


# ---------------- extra as-of features ----------------

def add_features(rows: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Schedule, players ruled out and the 6 h move, all as of each row's decision time; side-relative signs."""
    from agents.graph import OUT_STATUSES
    from replay import NEWS_WINDOW
    g = tables["games"].set_index("game_id")
    out = rows.copy()
    home = g.home_team.reindex(out.game_id).to_numpy()
    away = g.away_team.reindex(out.game_id).to_numpy()
    yes_home = (out.team.to_numpy() == home)
    side_is_home = np.where(out.side == "yes", yes_home, ~yes_home)
    out["home_side"] = side_is_home.astype(float)
    out["side_team"] = np.where(side_is_home, home, away)
    out["opp_team"] = np.where(side_is_home, away, home)
    out["is_fav"] = (out.side_price > 0.5).astype(float)
    # Back-to-back: the team also played the day before (the schedule is known in advance).
    sched = pd.concat([tables["games"][["date", "home_team"]].rename(columns={"home_team": "team"}),
                       tables["games"][["date", "away_team"]].rename(columns={"away_team": "team"})])
    played = set(zip(sched.team, pd.to_datetime(sched.date).dt.strftime("%Y-%m-%d")))
    prev_day = (pd.to_datetime(out.date) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    out["b2b_side"] = [float((t, d) in played) for t, d in zip(out.side_team, prev_day)]
    out["b2b_opp"] = [float((t, d) in played) for t, d in zip(out.opp_team, prev_day)]
    # Players out or doubtful per side, from news published by the decision time (latest status per player).
    news = tables["news"]
    team_of = dict(zip(g.home_team_id, g.home_team)) | dict(zip(g.away_team_id, g.away_team))
    n_side, n_opp = [], []
    for r in out[["game_id", "as_of", "tip_time", "side_team", "opp_team"]].itertuples(index=False):
        n = news[(news.game_id == r.game_id) & (news.published_at <= r.as_of)
                 & (news.published_at >= r.tip_time - NEWS_WINDOW)]
        n = n.sort_values("published_at").groupby("player_id").tail(1)
        outs = n[n.status.str.lower().isin(OUT_STATUSES)].team_id.map(team_of)
        n_side.append(float((outs == r.side_team).sum()))
        n_opp.append(float((outs == r.opp_team).sum()))
    out["n_out_side"], out["n_out_opp"] = n_side, n_opp
    # Move of the side's mid over the last six hours (to the decision time).
    prices = tables["prices"]
    by_ticker = {k: v.sort_values("ts") for k, v in prices[prices.market_ticker.isin(set(out.ticker))]
                 .groupby("market_ticker")}
    empty = prices.iloc[:0]
    moves = []
    for r in out[["ticker", "side", "as_of"]].itertuples(index=False):
        p = by_ticker.get(r.ticker, empty)
        p = p[p.ts <= r.as_of]
        if p.empty:
            moves.append(0.0)
            continue
        before = p[p.ts <= r.as_of - pd.Timedelta(hours=6)]
        start = before.iloc[-1] if len(before) else p.iloc[0]
        d = (p.bid.iloc[-1] + p.ask.iloc[-1]) / 2 - (start.bid + start.ask) / 2
        moves.append(float(d if r.side == "yes" else -d))
    out["move_6h"] = moves
    return out


def counterfactual(rows: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """$20 fill at each row's decision time and its settlement: price, contracts, fee, close, CLV, P&L."""
    from replay import Order, Replay
    rp = Replay(tables, lambda *a: [])
    g = tables["games"].set_index("game_id")
    recs = []
    for r in rows.itertuples(index=False):
        tip = g.tip_time[r.game_id]
        fill, why = rp._fill(Order(r.ticker, r.side, 0.0, STAKE, "counterfactual"), r.as_of, tip)
        if fill is None:
            recs.append({"filled": False})
            continue
        s = rp._settle({**fill, "market_ticker": r.ticker, "side": r.side, "game_id": r.game_id}, tip)
        recs.append({"filled": True, "fill_price": s["price"], "contracts": s["contracts"], "fee_paid": s["fee"],
                     "close": s["close_price"], "clv": s["clv"], "clv_usd": s["clv"] * s["contracts"],
                     "clv_usd_after_fee": s["clv"] * s["contracts"] - s["fee"], "pnl": s["pnl"]})
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(recs)], axis=1)


# ---------------- anchor pool ----------------

def anchor_fills() -> pd.DataFrame:
    parts = []
    for w, folder in ANCHOR_RUNS.items():
        f = pd.read_parquet(folder / "fills.parquet").assign(window=w)
        f["as_of"] = pd.to_datetime(f.as_of, utc=True)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def anchor_pool(rows: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """One candidate per decision point (arm A's trade if it traded, else its best option), gap > NEAR_MISS."""
    fills = anchor_fills()
    key = ["game_id", "as_of", "ticker", "side"]
    traded = fills.rename(columns={"market_ticker": "ticker"})[key].assign(traded=True)
    r = rows.merge(traded, on=key, how="left")
    r["traded"] = r.traded.eq(True)
    # The traded route first, then the largest gap; equivalent routes (no on X = yes on Y) tie on gap.
    r = r.sort_values(["traded", "gap"], ascending=[False, False], kind="mergesort")
    cand = r.drop_duplicates(["game_id", "as_of"]).copy()
    if cand.traded.sum() != len(fills):
        raise SystemExit(f"anchor fills not found among option rows: {cand.traded.sum()} of {len(fills)}")
    first = cand.game_id.map(cand[cand.traded].groupby("game_id").as_of.min())
    cand["blocked"] = first.notna() & (cand.as_of > first)          # arm A already held this game
    cand = cand[(cand.gap > NEAR_MISS) | cand.traded].sort_values(["as_of", "game_id"]).reset_index(drop=True)
    cand = counterfactual(add_features(cand, tables), tables)
    tr = cand[cand.traded].merge(fills.rename(columns={"market_ticker": "ticker"}), on=key, suffixes=("", "_real"))
    assert len(tr) == len(fills) and tr.filled.all()
    for a, b in (("fill_price", "price_real"), ("contracts", "contracts_real"), ("clv", "clv_real")):
        assert np.allclose(tr[a], tr[b]), (a, tr[[a, b]].head())
    return cand[cand.filled].reset_index(drop=True)


# ---------------- sceptic set (Approach A's labelled set) ----------------

def reviewed_traces(games: pd.DataFrame) -> list:
    """(window, trace) for every proposal the reasoner sceptic reviewed, in both windows."""
    from evaluation.llm_sceptic_full import from_runs
    prev = load_traces(OUT / "prev-tool_sceptic" / "tool_sceptic")
    dates = games.date.astype(str)
    days = sorted(dates[(dates >= WINDOWS["test"][0]) & (dates <= WINDOWS["test"][1])].unique())
    test = [t for p in from_runs(games, days) for t in p["traces"]]
    return [("prev", t) for t in prev if "sceptic" in t] + [("test", t) for t in test if "sceptic" in t]


def cached_sceptic_prompts() -> dict:
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


def sceptic_set(rows: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """The reviewed proposals, joined to the as-of option row of the same (decision time, ticker, side)."""
    games = tables["games"]
    g = games.set_index("game_id")
    index = cached_sceptic_prompts()
    recs = []
    for window, t in reviewed_traces(games):
        gid, now = t["game_id"], pd.Timestamp(t["as_of"])
        tip = g.tip_time[gid]
        cand = t.get("candidate") or t.get("order")
        est = round(float(t["proposal"]["estimate_p"]), 4)
        key = (str(g.date[gid]), g.away_team[gid], g.home_team[gid], int(round((tip - now).total_seconds() / 60)),
               cand["side"], round(cand["price"], 4), est)
        hits = index.get(key, [])
        if not hits or len({h["prompt"] for h in hits}) > 1:
            raise SystemExit(f"{len(hits)} cached sceptic prompts for {key}")
        recs.append({"window": window, "game_id": gid, "as_of": now, "ticker": cand["ticker"], "side": cand["side"],
                     "estimate_p": est, "orig_approve": t["sceptic"]["verdict"] == "approve",
                     "orig_status": t["status"]})
    s = pd.DataFrame(recs)
    s["as_of"] = pd.to_datetime(s.as_of, utc=True)
    key = ["window", "game_id", "as_of", "ticker", "side"]
    df = s.merge(rows, on=key, how="left", validate="one_to_one")
    if df.gap.isna().any():
        raise SystemExit(f"{int(df.gap.isna().sum())} sceptic proposals without an as-of option row")
    df = counterfactual(add_features(df, tables), tables)
    assert df.filled.all() and np.allclose(df.fill_price, df.price)
    df["llm_gap"] = df.estimate_p - df.price - df.fee
    df["llm_minus_model"] = df.estimate_p - df.p_side
    check_against_fills(df, games)
    return df


def check_against_fills(df: pd.DataFrame, games: pd.DataFrame):
    """Counterfactual fills of approved proposals must reproduce the runs' real fills."""
    from evaluation.llm_sceptic_full import from_runs
    dates = games.date.astype(str)
    days = sorted(dates[(dates >= WINDOWS["test"][0]) & (dates <= WINDOWS["test"][1])].unique())
    fills = [p["fills"] for p in from_runs(games, days) if len(p["fills"])]
    fills.append(pd.read_parquet(OUT / "prev-tool_sceptic" / "tool_sceptic" / "fills.parquet"))
    real = pd.concat(fills, ignore_index=True)
    kept = df[df.orig_approve].merge(real, left_on=["game_id", "ticker"], right_on=["game_id", "market_ticker"],
                                     suffixes=("", "_real"))
    assert len(kept) == len(real) == df.orig_approve.sum(), (len(kept), len(real), df.orig_approve.sum())
    for a, b in (("fill_price", "price_real"), ("contracts", "contracts_real"), ("clv", "clv_real")):
        assert np.allclose(kept[a], kept[b]), (a, kept[[a, b]])


# ---------------- leakage checks ----------------

def truncated_tables(tables: dict, now) -> dict:
    """Only what was known at `now`: quotes and news up to now, box scores of games that tipped before now."""
    t = dict(tables)
    t["prices"] = tables["prices"][tables["prices"].ts <= now]
    t["news"] = tables["news"][tables["news"].published_at <= now]
    tipped = tables["games"].loc[tables["games"].tip_time < now, "game_id"]
    t["player_games"] = tables["player_games"][tables["player_games"].game_id.isin(tipped)]
    t["settlements"] = tables["settlements"].iloc[:0]
    return t


def leakage_check(pool: pd.DataFrame, tables: dict, n: int = 12, seed: int = SEED) -> int:
    """Recompute features for n decision points on tables truncated at the decision time; must be identical."""
    from forecast.api import Forecaster
    from forecast.policy import decision_rows
    assert not (set(FEATURES + LLM_FEATURES) & LABEL_COLS)
    sample = pool.sample(n, random_state=seed)
    for r in sample.itertuples(index=False):
        t = truncated_tables(tables, r.as_of)
        fc = Forecaster.load(MODEL_DIRS[r.window])
        rows = decision_rows(t, fc, str(r.date), str(r.date), stake=STAKE)
        rows = rows[(rows.game_id == r.game_id) & (rows.as_of == r.as_of) & (rows.ticker == r.ticker)
                    & (rows.side == r.side)]
        assert len(rows) == 1, (r.game_id, r.as_of, len(rows))
        mine = add_features(rows.assign(window=r.window), t).iloc[0]
        for f in FEATURES:
            a, b = float(mine[f]), float(getattr(r, f))
            assert abs(a - b) < 1e-9, (f, r.game_id, r.as_of, a, b)
    return len(sample)


# ---------------- models ----------------

def make_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(random_state=SEED, **GBM)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    raise ValueError(kind)


def target(df: pd.DataFrame, kind: str) -> np.ndarray:
    return df.clv_usd.to_numpy(float) if kind == "ridge" else (df.clv.to_numpy(float) > 0).astype(int)


def fit_score(train: pd.DataFrame, test: pd.DataFrame, kind: str, feats: list) -> np.ndarray:
    m = make_model(kind)
    y = target(train, kind)
    if kind != "ridge" and len(set(y)) < 2:
        return np.full(len(test), float(y[0]) if len(y) else 0.5)
    m.fit(train[feats].to_numpy(float), y)
    x = test[feats].to_numpy(float)
    return m.predict(x) if kind == "ridge" else m.predict_proba(x)[:, 1]


def oof_scores(train: pd.DataFrame, tune_rows: pd.DataFrame, kind: str, feats: list, folds=INNER_FOLDS):
    """Day-grouped out-of-fold scores of tune_rows (a subset of the training window) for choosing tau."""
    days = np.array(sorted(train.date.astype(str).unique()))
    fold_of = {d: i % folds for i, d in enumerate(days)}
    out = np.full(len(tune_rows), np.nan)
    tf = tune_rows.date.astype(str).map(fold_of).to_numpy()
    trf = train.date.astype(str).map(fold_of).to_numpy()
    for k in range(folds):
        if (tf == k).any():
            out[tf == k] = fit_score(train[trf != k], tune_rows[tf == k], kind, feats)
    return out


def tune_tau(scores: np.ndarray, clv_usd: np.ndarray, min_kept: int) -> float:
    """tau maximising CLV $ of {score >= tau} with at least min_kept kept; ties go to the higher tau."""
    best = None
    for tau in np.unique(scores):
        keep = scores >= tau
        if keep.sum() < min_kept:
            continue
        v = clv_usd[keep].sum()
        if best is None or v >= best[0] - 1e-12:
            best = (v, tau)
    return best[1] if best else np.inf


# ---------------- statistics ----------------

def day_boot_idx(days: pd.Series, reps=REPS, seed=SEED):
    """Day-clustered bootstrap: for each replicate, how many times each row's day is drawn."""
    uniq = np.array(sorted(days.astype(str).unique()))
    pos = pd.Index(uniq).get_indexer(days.astype(str))
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(reps, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
    return counts[:, pos]


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def auc(score, y) -> float:
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def auc_ci(score, y, boot) -> tuple:
    idx = np.arange(len(score))
    bs = [auc(np.repeat(score, b), np.repeat(y, b)) for b in boot[:AUC_REPS]]
    return (auc(score, y), *ci(bs))


def random_band(clv_usd: np.ndarray, k: int, draws=RANDOM_DRAWS, seed=SEED) -> np.ndarray:
    if k == 0:
        return np.zeros(draws)
    rng = np.random.default_rng(seed + k)
    return np.array([clv_usd[rng.choice(len(clv_usd), k, replace=False)].sum() for _ in range(draws)])


def filter_stats(w: pd.DataFrame, keep: np.ndarray, boot, current: np.ndarray | None) -> dict:
    c, cf, pnl, clv = (w[x].to_numpy(float) for x in ("clv_usd", "clv_usd_after_fee", "pnl", "clv"))
    keep = np.asarray(keep, bool)
    pos, k = clv > 0, int(keep.sum())
    oracle = c[pos].sum()
    r = {"n": len(w), "kept": k, "keep_rate": k / len(w), "clv_usd": c[keep].sum(),
         "clv_usd_after_fee": cf[keep].sum(), "mean_clv_c": 100 * clv[keep].mean() if k else np.nan,
         "pnl": np.nansum(pnl[keep]), "oracle_share": c[keep].sum() / oracle if oracle > 0 else np.nan,
         "wrong_veto": float(pos[~keep].mean()) if (~keep).any() else np.nan,
         "wrong_approve": float((~pos[keep]).mean()) if k else np.nan}
    r["clv_usd_lo"], r["clv_usd_hi"] = ci(boot @ (c * keep))
    with np.errstate(invalid="ignore", divide="ignore"):
        r["mean_clv_lo"], r["mean_clv_hi"] = ci(100 * (boot @ (clv * keep)) / (boot @ keep.astype(float)))
    r["pnl_lo"], r["pnl_hi"] = ci(boot @ (np.nan_to_num(pnl) * keep))
    r["diff_keep_all"] = c[keep].sum() - c.sum()
    r["diff_keep_all_lo"], r["diff_keep_all_hi"] = ci(boot @ (c * keep) - boot @ c)
    if current is not None:
        r["diff_current"] = c[keep].sum() - c[current].sum()
        r["diff_current_lo"], r["diff_current_hi"] = ci(boot @ (c * keep) - boot @ (c * current))
    band = random_band(c, k)
    r["random_mean"], r["random_lo"], r["random_hi"] = band.mean(), *np.percentile(band, [2.5, 97.5])
    tie = np.isclose(band, r["clv_usd"], atol=1e-6)
    r["random_pctile"] = 100 * (((band < r["clv_usd"]) & ~tie).mean() + 0.5 * tie.mean())
    return r


# ---------------- evaluation ----------------

def learned_specs(pool_name: str) -> list:
    """(name, kind, training pool, features)."""
    specs = [("LR, anchor pool", "lr", "anchor", FEATURES), ("GBM, anchor pool", "gbm", "anchor", FEATURES),
             ("Ridge on CLV $, anchor pool", "ridge", "anchor", FEATURES)]
    if pool_name == "sceptic":
        specs += [("LR, sceptic set + LLM features", "lr", "sceptic", FEATURES + LLM_FEATURES)]
    return specs


def single_rule(train: pd.DataFrame, min_kept: int) -> tuple:
    """Best single feature and cut on the training rows: keep if sign * feature >= cut (max CLV $ kept)."""
    best = None
    c = train.clv_usd.to_numpy(float)
    for f in FEATURES:
        x = train[f].to_numpy(float)
        for sign in (1, -1):
            tau = tune_tau(sign * x, c, min_kept)
            if not np.isfinite(tau):
                continue
            v = c[sign * x >= tau].sum()
            if best is None or v > best[0]:
                best = (v, f, sign, tau)
    return best[1:]


def scores_for(pools: dict, pool_name: str, spec, train_w: str, test_w: str) -> tuple:
    """(test scores, tau) for one learned filter: fit on the training window, tau from out-of-fold scores."""
    name, kind, src, feats = spec
    train = pools[src][pools[src].window == train_w]
    tune_rows = pools[pool_name][pools[pool_name].window == train_w]
    test = pools[pool_name][pools[pool_name].window == test_w]
    # Tune rows (the evaluated pool's training window) are scored by models that never saw their game-day.
    oof = oof_scores(train, tune_rows, kind, feats)
    tau = tune_tau(oof, tune_rows.clv_usd.to_numpy(float), MIN_KEPT[pool_name])
    return fit_score(train, test, kind, feats), tau


def current_mask(w: pd.DataFrame, pool_name: str) -> np.ndarray:
    return (w.orig_approve if pool_name == "sceptic" else w.traded & ~w.blocked).to_numpy(bool)


def evaluate_pool(pools: dict, pool_name: str) -> tuple:
    pool = pools[pool_name]
    cur_label = "Current sceptic (approve/reject)" if pool_name == "sceptic" else "Arm A's own trades (gap > 0.04)"
    rows, aucs, oos, taus = [], {}, {}, {}
    specs = learned_specs(pool_name)
    for train_w, test_w in (("prev", "test"), ("test", "prev")):
        w = pool[pool.window == test_w].reset_index(drop=True)
        boot = day_boot_idx(w.date)
        cur = current_mask(w, pool_name)
        y = w.clv.to_numpy() > 0
        direction = f"train {WINDOW_NAMES[train_w]} -> test {WINDOW_NAMES[test_w]}"

        def add(name, keep, tau=np.nan, kind="baseline", **extra):
            rows.append({"pool": pool_name, "direction": direction, "test_window": test_w, "filter": name,
                         "kind": kind, "tau": tau, **extra,
                         **filter_stats(w, keep, boot, None if name == cur_label else cur)})
        add("Keep all", np.ones(len(w), bool))
        add(cur_label, cur)
        add("Oracle (CLV > 0)", y)
        add("Rule: skip if the market moved against the side", w.market_move.to_numpy() >= 0, kind="rule")
        f, sign, tau = single_rule(pool[pool.window == train_w], MIN_KEPT[pool_name])
        add(f"Rule: {f} {'>=' if sign > 0 else '<='} cut tuned on train", sign * w[f].to_numpy() >= tau,
            sign * tau, kind="rule", feature=f)
        for spec in specs:
            s, tau = scores_for(pools, pool_name, spec, train_w, test_w)
            oos.setdefault(spec[0], np.full(len(pool), np.nan))[np.flatnonzero(pool.window == test_w)] = s
            taus[(spec[0], train_w)] = tau
            add(spec[0] + ", tau from train", s >= tau, tau, kind="learned")
            k = int(cur.sum())
            top = np.zeros(len(w), bool)
            top[np.argsort(-s, kind="mergesort")[:k]] = True
            add(spec[0] + f", top {k} (matched to current)", top, kind="learned_matched")
            aucs[(spec[0], test_w)] = auc_ci(s, y, boot)
    # Pooled leave-one-window-out: each window scored by the model trained on the other.
    w = pool.reset_index(drop=True)
    boot = day_boot_idx(w.date)
    cur = current_mask(w, pool_name)
    y = w.clv.to_numpy() > 0
    direction = "pooled leave-one-window-out"
    for name, keep, kind in (("Keep all", np.ones(len(w), bool), "baseline"), (cur_label, cur, "baseline"),
                             ("Oracle (CLV > 0)", y, "baseline"),
                             ("Rule: skip if the market moved against the side", w.market_move.to_numpy() >= 0,
                              "rule")):
        rows.append({"pool": pool_name, "direction": direction, "test_window": "both", "filter": name, "kind": kind,
                     "tau": np.nan, **filter_stats(w, keep, boot, None if name == cur_label else cur)})
    for spec in specs:
        s = oos[spec[0]]
        tau_row = w.window.map(lambda x: taus[(spec[0], OTHER[x])]).to_numpy()
        rows.append({"pool": pool_name, "direction": direction, "test_window": "both",
                     "filter": spec[0] + ", tau from the other window", "kind": "learned", "tau": np.nan,
                     **filter_stats(w, s >= tau_row, boot, cur)})
        k = int(cur.sum())
        top = np.zeros(len(w), bool)
        top[np.argsort(-s, kind="mergesort")[:k]] = True
        rows.append({"pool": pool_name, "direction": direction, "test_window": "both",
                     "filter": spec[0] + f", top {k} (matched to current)", "kind": "learned_matched",
                     "tau": np.nan, **filter_stats(w, top, boot, cur)})
        aucs[(spec[0], "both")] = auc_ci(s, y, boot)
    return pd.DataFrame(rows), aucs, oos, taus


def rolling_months(anchor: pd.DataFrame) -> tuple:
    """Anchor pool, expanding window by month: train on all earlier months, test on the next one."""
    a = anchor.assign(month=pd.to_datetime(anchor.date).dt.strftime("%Y-%m")).reset_index(drop=True)
    months = sorted(a.month.unique())
    out = {name: np.full(len(a), np.nan) for name, *_ in learned_specs("anchor")}
    keep = {name: np.zeros(len(a), bool) for name in out}
    for m in months[1:]:
        train, test = a[a.month < m], a.month == m
        for name, kind, _, feats in learned_specs("anchor"):
            s = fit_score(train, a[test], kind, feats)
            tau = tune_tau(oof_scores(train, train, kind, feats), train.clv_usd.to_numpy(float), MIN_KEPT["anchor"])
            out[name][test.to_numpy()] = s
            keep[name][test.to_numpy()] = s >= tau
    w = a[a.month > months[0]].reset_index(drop=True)
    sel = (a.month > months[0]).to_numpy()
    boot = day_boot_idx(w.date)
    cur = current_mask(w, "anchor")
    y = w.clv.to_numpy() > 0
    rows = [{"filter": n, **filter_stats(w, k, boot, None if n.startswith("Arm A") else cur)}
            for n, k in (("Keep all", np.ones(len(w), bool)), ("Arm A's own trades (gap > 0.04)", cur),
                         ("Oracle (CLV > 0)", y),
                         ("Rule: skip if the market moved against the side", w.market_move.to_numpy() >= 0))]
    aucs = {}
    for name in out:
        rows.append({"filter": name + ", tau from earlier months", **filter_stats(w, keep[name][sel], boot, cur)})
        aucs[name] = auc_ci(out[name][sel], y, boot)
    return pd.DataFrame(rows).assign(pool="anchor", direction=f"rolling monthly ({', '.join(months[1:])})",
                                     test_window="rolling"), aucs


def shuffled_label_auc(pools: dict, reps: int = 20) -> tuple:
    """Control: LR trained on permuted labels; out-of-sample AUC should centre on 0.5."""
    rng = np.random.default_rng(SEED)
    a = pools["anchor"]
    vals = []
    for _ in range(reps):
        for tr, te in (("prev", "test"), ("test", "prev")):
            train = a[a.window == tr].copy()
            train["clv"] = rng.permutation(train.clv.to_numpy())
            test = a[a.window == te]
            vals.append(auc(fit_score(train, test, "lr", FEATURES), test.clv.to_numpy() > 0))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def importances(pools: dict) -> pd.DataFrame:
    """Standardised LR coefficients (fit on each window's anchor pool and both) and GBM permutation importance."""
    from sklearn.inspection import permutation_importance
    a = pools["anchor"]
    out = pd.DataFrame(index=FEATURES)
    for w in ("prev", "test", "both"):
        d = a if w == "both" else a[a.window == w]
        m = make_model("lr").fit(d[FEATURES].to_numpy(float), target(d, "lr"))
        out[f"lr_coef_{w}"] = m[-1].coef_[0]
    imp = []
    for tr, te in (("prev", "test"), ("test", "prev")):
        m = make_model("gbm").fit(a[a.window == tr][FEATURES].to_numpy(float), target(a[a.window == tr], "gbm"))
        t = a[a.window == te]
        r = permutation_importance(m, t[FEATURES].to_numpy(float), target(t, "gbm"), scoring="roc_auc",
                                   n_repeats=20, random_state=SEED)
        imp.append(r.importances_mean)
    out["gbm_perm_auc_drop_oos"] = np.mean(imp, axis=0)
    s = pools["sceptic"]
    m = make_model("lr").fit(s[FEATURES + LLM_FEATURES].to_numpy(float), target(s, "lr"))
    out = out.reindex(FEATURES + LLM_FEATURES)
    out["lr_coef_sceptic_both"] = m[-1].coef_[0]
    return out.rename_axis("feature").reset_index()


# ---------------- chart ----------------

def chart(pools: dict, oos: dict, taus: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, name in zip(axes, ("sceptic", "anchor")):
        w = pools[name].reset_index(drop=True)
        c = w.clv_usd.to_numpy(float)
        ks = np.unique(np.linspace(0, len(w), 60).astype(int))
        bands = np.array([np.percentile(random_band(c, k, 3000), [2.5, 97.5]) for k in ks])
        ax.fill_between(ks / len(w), bands[:, 0], bands[:, 1], color="0.86", label="Random filter, 95% band")
        ax.plot(ks / len(w), ks * c.mean(), color="0.5", lw=1, ls="--", label="Random filter, mean")
        for i, (spec, s) in enumerate((k, v) for k, v in oos[name].items()):
            order = np.argsort(-s, kind="mergesort")
            ax.plot(np.arange(1, len(w) + 1) / len(w), np.cumsum(c[order]), lw=1.2, color=f"C{i}",
                    label=f"{spec} (out of sample, sweep)")
            tau_row = w.window.map(lambda x: taus[name][(spec, OTHER[x])]).to_numpy()
            keep = s >= tau_row
            ax.scatter([keep.mean()], [c[keep].sum()], color=f"C{i}", s=55, zorder=5, edgecolor="k")
        cur = current_mask(w, name)
        ax.scatter([cur.mean()], [c[cur].sum()], color="C3", marker="s", s=60, zorder=6, edgecolor="k",
                   label="Current sceptic" if name == "sceptic" else "Arm A's own trades")
        ax.scatter([(c > 0).mean()], [c[c > 0].sum()], color="C2", marker="^", s=60, zorder=6, edgecolor="k",
                   label="Oracle (CLV > 0)")
        ax.scatter([1.0], [c.sum()], color="k", marker="x", s=50, zorder=6, label="Keep all")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Keep rate")
        ax.set_ylabel(r"CLV \$ of kept trades (\$20 stake)")
        ax.set_title(("Sceptic set" if name == "sceptic" else "Anchor pool") +
                     f": {len(w)} proposals, both windows, leave-one-window-out")
        ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------- report ----------------

def fmt_row(r) -> str:
    f2 = lambda v: "–" if pd.isna(v) else f"{v:+.2f}"
    d_cur = "–" if pd.isna(r.get("diff_current", np.nan)) else \
        f"{f2(r['diff_current'])} [{f2(r['diff_current_lo'])}, {f2(r['diff_current_hi'])}]"
    return (f"| {r['filter']} | {r['kept']} | {r['keep_rate']:.0%} | {f2(r['clv_usd'])} [{f2(r['clv_usd_lo'])}, "
            f"{f2(r['clv_usd_hi'])}] | " + (f"{r['mean_clv_c']:+.2f} [{r['mean_clv_lo']:+.2f}, {r['mean_clv_hi']:+.2f}]"
                                           if r["kept"] else "–") +
            f" | {d_cur} | {f2(r['diff_keep_all'])} [{f2(r['diff_keep_all_lo'])}, {f2(r['diff_keep_all_hi'])}] | "
            f"{f2(r['random_mean'])} [{f2(r['random_lo'])}, {f2(r['random_hi'])}] | {r['random_pctile']:.0f} | "
            + ("–" if pd.isna(r["oracle_share"]) else f"{r['oracle_share']:.0%}") +
            f" | {f2(r['pnl'])} [{f2(r['pnl_lo'])}, {f2(r['pnl_hi'])}] |")


HEADER = ["| Filter | Kept | Keep rate | CLV $ [95% CI] | Mean CLV ¢ [95% CI] | Δ CLV $ vs current [95% CI] "
          "| Δ CLV $ vs keep-all [95% CI] | Random same size: mean [95% band] | Pctile vs random | Oracle share "
          "| P&L [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]


def write_report(pools, table, aucs, imp, control, leak_n, rolling_aucs) -> str:
    s, a = pools["sceptic"], pools["anchor"]
    md = ["# Approach C: a learned trade filter vs the current LLM sceptic", "",
          "**Question.** Can a cheap learned filter on decision-time features (no LLM calls) pick, among proposed "
          "trades, the ones that beat the closing line, better than random and better than the current "
          "deepseek-reasoner sceptic?", "",
          "**Pools** (kept separate; counterfactual $20 fill at each decision time, CLV = side's mid at tip − price "
          "paid, CLV $ = CLV × contracts, fee in P&L only):", "",
          f"- *Sceptic set* — every proposal the current sceptic reviewed: {len(s)} "
          f"({(s.window == 'prev').sum()} Dec 2025 – Jan 2026 with walk-forward refit models, "
          f"{(s.window == 'test').sum()} Feb – Apr 2026 test). Recomputed fills match the "
          f"{int(s.orig_approve.sum())} real fills of approved proposals.",
          f"- *Anchor pool* — the deterministic anchor agent's candidate at every decision point of the same "
          f"windows (its trade where it traded, else its best option) with gap after fees > {NEAR_MISS:.2f}: "
          f"{len(a)} ({(a.window == 'prev').sum()} Dec – Jan, {(a.window == 'test').sum()} Feb – Apr), of which "
          f"{int(a.traded.sum())} are arm A's real trades ({int((a.traded & (a.window == 'prev')).sum())} + "
          f"{int((a.traded & (a.window == 'test')).sum())}; recomputed fills match all of them) and "
          f"{int((~a.traded).sum())} are near misses or later decision points of a game already held.", ""]
    for name, d in (("Sceptic set", s), ("Anchor pool", a)):
        for w in ("prev", "test"):
            x = d[d.window == w]
            md.append(f"- {name}, {WINDOW_NAMES[w]}: n = {len(x)}, keep-all CLV $ {x.clv_usd.sum():+.2f}, "
                      f"{int((x.clv > 0).sum())} with CLV > 0, oracle ceiling CLV $ {x.clv_usd[x.clv > 0].sum():+.2f}.")
    md += ["", "**Features** (as of the decision time, side-relative): " + ", ".join(f"`{f}`" for f in FEATURES) +
           "; for the sceptic set also `llm_gap` (analyst estimate − price − fee) and `llm_minus_model` (analyst "
           "estimate − anchor model). Labels: CLV > 0 (classifiers), CLV $ (ridge).", "",
           f"**Leakage checks.** (1) No label-derived column is a feature (checked by name). (2) For {leak_n} "
           "random anchor-pool decision points the features were recomputed on tables truncated at the decision "
           "time (quotes and news up to then, box scores of games already tipped, no settlements): identical. "
           f"(3) LR trained on permuted labels: out-of-sample AUC {control[0]:.2f} (2.5–97.5%: {control[1]:.2f}–"
           f"{control[2]:.2f}).", "",
           f"**Models.** L2 logistic regression (standardised, C = {LR_C}), gradient boosting "
           f"(depth {GBM['max_depth']}, {GBM['max_iter']} trees, learning rate {GBM['learning_rate']}, min leaf "
           f"{GBM['min_samples_leaf']}, L2 {GBM['l2_regularization']}), ridge on CLV $ (α = {RIDGE_ALPHA}); "
           "hyperparameters fixed in advance, not tuned. The keep threshold τ is chosen on the training window "
           f"only, from day-grouped {INNER_FOLDS}-fold out-of-fold scores (max CLV $ kept, at least "
           f"{MIN_KEPT['sceptic']} kept on the sceptic set / {MIN_KEPT['anchor']} on the anchor pool). "
           "\"Top k (matched)\" keeps the k highest scores, k = the current filter's count in that test window. "
           f"CIs: day-clustered bootstrap ({REPS} replicates, seed {SEED}; AUC {AUC_REPS}); random filter: "
           f"{RANDOM_DRAWS:,} random subsets of the same size, percentile = share with lower CLV $.", ""]
    md += ["## AUC of the learned scores vs CLV > 0 (out of sample, 95% day-clustered CI)", "",
           "| Pool | Model | Dec–Jan (trained Feb–Apr) | Feb–Apr (trained Dec–Jan) | Pooled LOWO |",
           "| --- | --- | --- | --- | --- |"]
    for pool_name in ("sceptic", "anchor"):
        for name, *_ in learned_specs(pool_name):
            cells = [f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]" for x in
                     (aucs[pool_name][(name, w)] for w in ("prev", "test", "both"))]
            md.append(f"| {pool_name} | {name} | " + " | ".join(cells) + " |")
    for name, x in rolling_aucs.items():
        md.append(f"| anchor, rolling monthly | {name} | | | {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}] |")
    md.append("")
    for pool_name, title in (("sceptic", "Sceptic set (like-for-like with the current sceptic)"),
                             ("anchor", "Anchor pool")):
        md += [f"## {title}", ""]
        for direction, sub in table[table.pool == pool_name].groupby("direction", sort=False):
            md += [f"### {direction}", ""] + HEADER + [fmt_row(r) for _, r in sub.iterrows()] + [""]
    md += verdict(table, aucs, control)
    md += ["## Feature weights", "",
           "Standardised LR coefficients (positive = more likely CLV > 0) on the anchor pool per window and pooled, "
           "GBM out-of-sample permutation importance (mean AUC drop, both directions), and the LR on the pooled "
           "sceptic set with LLM features.", "",
           "| Feature | LR Dec–Jan | LR Feb–Apr | LR both | GBM perm. ΔAUC | LR sceptic set |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r in imp.itertuples():
        cell = lambda v: "–" if pd.isna(v) else f"{v:+.3f}"
        md.append(f"| {r.feature} | {cell(r.lr_coef_prev)} | {cell(r.lr_coef_test)} | {cell(r.lr_coef_both)} | "
                  f"{cell(r.gbm_perm_auc_drop_oos)} | {cell(r.lr_coef_sceptic_both)} |")
    md.append("")
    return "\n".join(md)


def verdict(table: pd.DataFrame, aucs: dict, control: tuple) -> list:
    pooled = table[table.direction == "pooled leave-one-window-out"]
    sc, an = pooled[pooled.pool == "sceptic"], pooled[pooled.pool == "anchor"]
    cur = sc[sc["filter"].str.startswith("Current")].iloc[0]
    learned = sc[sc.kind.isin(["learned", "learned_matched"])]
    best = learned.loc[learned.clv_usd.idxmax()]
    beat_cur = learned[learned.diff_current_lo > 0]
    worse_cur = learned[learned.diff_current_hi < 0]
    rule = sc[sc["filter"].str.startswith("Rule: skip")].iloc[0]
    a_learned = an[an.kind == "learned"]
    a_rule = an[an["filter"].str.startswith("Rule: skip")].iloc[0]
    a_cur = an[an["filter"].str.startswith("Arm A")].iloc[0]
    a_matched = an[an.kind == "learned_matched"]
    strong = [r["filter"].split(", top")[0] for _, r in a_matched.iterrows() if r.random_pctile >= 97.5]
    pooled_aucs = [v for (name, w), v in {**{(k, w): x for (k, w), x in aucs["sceptic"].items()},
                                          **{(k + " (anchor)", w): x for (k, w), x in aucs["anchor"].items()}}.items()
                   if w == "both"]
    lo_auc, hi_auc = min(x[0] for x in pooled_aucs), max(x[0] for x in pooled_aucs)
    any_sig = any(x[1] > 0.5 for x in pooled_aucs)
    return ["## Verdict", "",
            f"- **Ranking.** Out of sample the learned scores barely rank proposals: pooled AUC "
            f"{lo_auc:.2f}–{hi_auc:.2f}, " + ("and no pooled CI excludes 0.5." if not any_sig else
                                              "with some pooled CIs just above 0.5.") +
            f" For comparison, LR on permuted labels gives {control[0]:.2f} ({control[1]:.2f}–{control[2]:.2f}).",
            f"- **Sceptic set (like-for-like).** The current sceptic keeps {cur.kept} of {cur.n} for CLV $ "
            f"{cur.clv_usd:+.2f} (random percentile {cur.random_pctile:.0f}). The best learned filter "
            f"(\"{best['filter']}\") keeps {best.kept} for {best.clv_usd:+.2f} (percentile {best.random_pctile:.0f}), "
            f"{best.diff_current:+.2f} [{best.diff_current_lo:+.2f}, {best.diff_current_hi:+.2f}] vs the current "
            f"sceptic. {len(beat_cur)} of {len(learned)} learned variants beat the current sceptic with a CI above "
            f"zero; {len(worse_cur)} are worse with a CI below zero. All of them beat keep-all, as any small "
            "filter does when keep-all loses.",
            f"- **One-feature rule.** \"Skip if the market moved against the side\" keeps {rule.kept} "
            f"({rule.keep_rate:.0%}) of the sceptic set for {rule.clv_usd:+.2f} (percentile "
            f"{rule.random_pctile:.0f}, mean {rule.mean_clv_c:+.2f}¢); on the anchor pool {a_rule.kept} "
            f"({a_rule.keep_rate:.0%}) for {a_rule.clv_usd:+.2f} (percentile {a_rule.random_pctile:.0f}) vs arm A's "
            f"own {a_cur.clv_usd:+.2f}. It removes losers about as well as anything learned, but the trades it keeps "
            "only break even on CLV; ML adds nothing over it.",
            f"- **Anchor pool, tuned τ.** With τ from the other window the learned filters keep "
            f"{a_learned.kept.min()}–{a_learned.kept.max()} of {a_learned.n.iloc[0]} "
            f"(CLV $ {a_learned.clv_usd.min():+.2f} to {a_learned.clv_usd.max():+.2f}, random percentiles "
            f"{a_learned.random_pctile.min():.0f}–{a_learned.random_pctile.max():.0f})"
            + (": the tuned threshold mostly learns \"trade almost nothing\", which beats arm A only because arm A's "
               "trades lose CLV on average." if a_learned.keep_rate.max() < 0.1 else "."),
            "- **Anchor pool, matched to arm A's keep rate** (top k by score, k = arm A's trades): "
            + "; ".join(f"{r['filter'].split(', top')[0]} {r.clv_usd:+.2f} (percentile {r.random_pctile:.0f}, "
                        f"{r.diff_current:+.2f} [{r.diff_current_lo:+.2f}, {r.diff_current_hi:+.2f}] vs arm A)"
                        for _, r in a_matched.iterrows())
            + f"; arm A itself {a_cur.clv_usd:+.2f}. "
            + (f"{', '.join(strong)} ranks the anchor pool better than random at this keep rate, i.e. it drops "
               "the worst losers, but the kept set still loses CLV." if strong else
               "None ranks clearly better than random."),
            f"- **Bottom line.** With n = {cur.n} (sceptic set) / {a_cur.n} (anchor pool) and these features, "
            + ("a learned filter does not beat random selection at the sceptic's keep rate and does not beat the "
               "current sceptic out of sample" if not len(beat_cur) and best.random_pctile < 97.5 else
               "some learned variants look better on the sceptic set, but see the CIs above before trusting them")
            + (f"; on the larger anchor pool it can drop the worst losers at arm A's {a_cur.keep_rate:.0%} keep "
               "rate, but the kept set does not reach positive CLV." if strong else ".")
            + f" The current sceptic's CLV $ {cur.clv_usd:+.2f} on {cur.kept} trades remains the best point estimate"
            + (", but its own CI includes zero" if cur.clv_usd_lo <= 0 else "")
            + "; nothing here is evidence of a reliable positive-CLV selector.", ""]


def build_pools(tables: dict) -> dict:
    rows = pd.read_parquet(WORK / "option_rows.parquet")
    # decision_rows' own outcome columns are dropped: labels come only from counterfactual() below.
    rows = rows.drop(columns=["close", "clv", "net_clv", "won", "contracts"])
    rows["as_of"] = pd.to_datetime(rows.as_of, utc=True)
    rows["tip_time"] = pd.to_datetime(rows.tip_time, utc=True)
    rows["date"] = rows.date.astype(str)
    pools = {"anchor": anchor_pool(rows, tables), "sceptic": sceptic_set(rows, tables)}
    for p in pools.values():
        p["y"] = (p.clv > 0).astype(int)
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="report", choices=["extract", "report"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--leak-n", type=int, default=12)
    args = ap.parse_args()
    if args.step == "extract":
        extract(args.workers)
        return
    tables = load_frozen()
    pools = build_pools(tables)
    for name, p in pools.items():
        print(f"{name}: {len(p)} rows {p.window.value_counts().to_dict()}", flush=True)
        p.to_parquet(WORK / f"{name}_pool.parquet", index=False)
    leak_n = leakage_check(pools["anchor"], tables, args.leak_n) if args.leak_n else 0
    print(f"leakage check passed on {leak_n} decision points", flush=True)
    tabs, aucs, oos, taus = [], {}, {}, {}
    for name in ("sceptic", "anchor"):
        t, aucs[name], oos[name], taus[name] = evaluate_pool(pools, name)
        tabs.append(t)
    roll, rolling_aucs = rolling_months(pools["anchor"])
    table = pd.concat(tabs + [roll], ignore_index=True)
    control = shuffled_label_auc(pools)
    imp = importances(pools)
    table.to_csv(RESULTS / f"{STEM}.csv", index=False)
    imp.to_csv(RESULTS / f"{STEM}_features.csv", index=False)
    chart(pools, oos, taus, RESULTS / f"{STEM}.png")
    md = write_report(pools, table, aucs, imp, control, leak_n, rolling_aucs)
    (RESULTS / f"{STEM}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
