"""M4-NN against M4, the anchor estimate and the market on the 501 test games.

    python -m evaluation.win_nn_eval            # ~5 min on CPU; caches feature rows in runs/win_nn/

Same rows (forecast.win.training_rows, absences known), same split (train
before 2026-02-01) and same test games (evaluation/results/m4_games.csv) as
evaluation/m4_calibration.py. Three seeds per model: per-seed mean +- std, and
the 3-seed average probability as the model's forecast. Brier and log-loss
differences have 95% CIs that resample game-days (2000 replicates). "Anchor +
X shift" is the agent's estimate: home mid 24 h before tip plus model X's
after-minus-before news shift, clipped to [0.02, 0.98].

Writes win_nn.md, win_nn.csv, win_nn_seeds.csv and win_nn.png to evaluation/results/.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import REPS, SEED
from forecast.win_nn import SPLIT, VAL_START, NeuralWinModel, sequences

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
CACHE = ROOT / "runs" / "win_nn"
SEEDS = (0, 1, 2)
GRID = [{"hidden": h, "lr": lr} for h in (8, 16) for lr in (1e-3, 3e-4)]
AGENT = "Anchor + M4 shift (agent's estimate)"
BINS = np.linspace(0, 1, 11)


def brier_terms(p, y):
    return (np.asarray(p, float) - y) ** 2


def log_terms(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def diff_ci(a_terms, b_terms, days, reps=REPS, seed=SEED) -> tuple:
    """mean(a) - mean(b) per game with a 95% CI resampling game-days."""
    d = pd.DataFrame({"day": days, "x": a_terms - b_terms}).groupby("day").x.agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (reps, len(s)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def load_rows():
    from forecast.baselines import load_source
    from forecast.history import History
    from forecast.win import training_rows

    player_games, games = load_source("frozen")
    path = CACHE / "rows.parquet"
    if path.exists():
        rows = pd.read_parquet(path)
    else:
        rows = training_rows(History(player_games, games), games)
        CACHE.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path)
    return rows, games


def anchors(t: pd.DataFrame) -> np.ndarray:
    path = CACHE / "anchor.csv"
    if path.exists():
        a = pd.read_csv(path, dtype={"game_id": str}).set_index("game_id").anchor
        return t.game_id.map(a).to_numpy(float)
    from evaluation.m4_calibration import anchor_mids
    from replay import load_tables
    a = anchor_mids(t, load_tables())
    pd.DataFrame({"game_id": t.game_id, "anchor": a}).to_csv(path, index=False)
    return a


def select(tr, s_tr, kind) -> tuple[dict, pd.DataFrame]:
    """Pick hidden size and learning rate by mean validation BCE over SEEDS (validation slice only, no test data)."""
    rows = []
    for cfg in GRID:
        losses = [min(NeuralWinModel(kind, seed=s, **cfg).fit(tr, s_tr).curve) for s in SEEDS]
        rows.append({"model": kind, **cfg, "val_bce_mean": float(np.mean(losses)), "val_bce_std": float(np.std(losses, ddof=1))})
        print(f"  {kind} {cfg}: validation BCE {rows[-1]['val_bce_mean']:.4f}", flush=True)
    g = pd.DataFrame(rows)
    best = g.loc[g.val_bce_mean.idxmin()]
    return {k: (int(best[k]) if k == "hidden" else float(best[k])) for k in GRID[0]}, g


def logistic_val_bce(tr) -> float:
    from sklearn.linear_model import LogisticRegression
    from forecast.win import WIN_FEATURES
    is_val = pd.DatetimeIndex(tr.tip_time) >= pd.Timestamp(VAL_START, tz="UTC")
    m = LogisticRegression(C=1.0, max_iter=1000).fit(tr.loc[~is_val, WIN_FEATURES], tr.home_win[~is_val])
    return float(log_terms(m.predict_proba(tr.loc[is_val, WIN_FEATURES])[:, 1], tr.home_win[is_val].to_numpy()).mean())


def train_all(rows, seq, test_mask):
    train_mask = (pd.DatetimeIndex(rows.tip_time) < pd.Timestamp(SPLIT, tz="UTC"))
    tr, te = rows[train_mask].reset_index(drop=True), rows[test_mask].reset_index(drop=True)
    preds, seeds, grids = {}, [], []
    for kind in ("mlp", "gru"):
        s_tr, s_te = (seq[train_mask], seq[test_mask]) if kind == "gru" else (None, None)
        cfg, g = select(tr, s_tr, kind)
        grids.append(g.assign(chosen=[all(r[k] == v for k, v in cfg.items()) for _, r in g.iterrows()]))
        after, before = [], []
        for seed in SEEDS:
            t0 = time.time()
            m = NeuralWinModel(kind, seed=seed, **cfg).fit(tr, s_tr)
            after.append(m.predict(te, s_te))
            before.append(m.predict_before_news(te, s_te))
            seeds.append({"model": kind, "seed": seed, **cfg, "best_epoch": m.best_epoch,
                          "val_loss": min(m.curve) if m.curve else np.nan, "seconds": time.time() - t0})
            print(f"  {kind} seed {seed}: best epoch {m.best_epoch}, val BCE {seeds[-1]['val_loss']:.4f} "
                  f"[{seeds[-1]['seconds']:.0f}s]", flush=True)
        preds[kind] = (np.array(after), np.array(before))
    grid = pd.concat(grids, ignore_index=True)
    grid.attrs["logistic_val_bce"] = logistic_val_bce(tr)
    return te, preds, pd.DataFrame(seeds), int(train_mask.sum()), grid


def clip(p):
    return np.clip(np.asarray(p, float), 0.02, 0.98)


def predictors(t, preds) -> dict:
    out = {"M4 (logistic regression)": t.m4_after.to_numpy(),
           "M4-NN MLP (3-seed mean)": preds["mlp"][0].mean(0),
           "M4-NN GRU (3-seed mean)": preds["gru"][0].mean(0),
           "Anchor + M4 shift (agent's estimate)": clip(t.anchor + t.m4_after - t.m4_before),
           "Anchor + MLP shift": clip(t.anchor + preds["mlp"][0].mean(0) - preds["mlp"][1].mean(0)),
           "Anchor + GRU shift": clip(t.anchor + preds["gru"][0].mean(0) - preds["gru"][1].mean(0)),
           "Market 24 h before tip": t.anchor.to_numpy(),
           "Market 1 h before tip": t.market_before.to_numpy()}
    return out


def scores(t, preds_by_name) -> pd.DataFrame:
    y, days = t.home_win.to_numpy(), t.date.to_numpy()
    ref = {"m4": preds_by_name["M4 (logistic regression)"], "mkt": preds_by_name["Market 1 h before tip"],
           "agent": preds_by_name[AGENT]}
    rows = []
    for name, p in preds_by_name.items():
        r = {"predictor": name, "games": len(y), "brier": brier_terms(p, y).mean(), "log_loss": log_terms(p, y).mean(),
             "accuracy": float(np.mean((p > 0.5) == y))}
        for tag, q in ref.items():
            for metric, f in (("brier", brier_terms), ("log_loss", log_terms)):
                e, lo, hi = diff_ci(f(p, y), f(q, y), days)
                r.update({f"{metric}_minus_{tag}": e, f"{metric}_minus_{tag}_lo": lo, f"{metric}_minus_{tag}_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def seed_table(t, preds) -> pd.DataFrame:
    y = t.home_win.to_numpy()
    rows = []
    for kind, (after, _) in preds.items():
        b = np.array([brier_terms(p, y).mean() for p in after])
        ll = np.array([log_terms(p, y).mean() for p in after])
        rows.append({"model": kind, "brier_mean": b.mean(), "brier_std": b.std(ddof=1), "log_loss_mean": ll.mean(),
                     "log_loss_std": ll.std(ddof=1), "brier_min": b.min(), "brier_max": b.max()})
    return pd.DataFrame(rows)


def figure(t, preds_by_name, sc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = t.home_win.to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    a = ax[0]
    for name, c in (("M4 (logistic regression)", "#3b5bdb"), ("M4-NN MLP (3-seed mean)", "#2b8a3e"),
                    ("M4-NN GRU (3-seed mean)", "#ae3ec9"), ("Market 1 h before tip", "#e8590c")):
        p = preds_by_name[name]
        g = pd.DataFrame({"p": p, "y": y, "b": pd.cut(p, BINS, include_lowest=True)}).groupby("b", observed=True) \
            .agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
        g = g[g.n >= 5]
        a.plot(g.p, g.y, "o-", color=c, label=name, ms=4)
    a.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    a.set(xlabel="predicted home win probability", ylabel="actual home win rate", xlim=(0, 1), ylim=(0, 1),
          title=f"Reliability on {len(y)} test games (bins with 5+ games)")
    a.legend(fontsize=8, loc="upper left")
    a = ax[1]
    s = sc.sort_values("brier", ascending=False)
    err = np.vstack([s.brier_minus_mkt - s.brier_minus_mkt_lo, s.brier_minus_mkt_hi - s.brier_minus_mkt])
    a.barh(s.predictor, s.brier_minus_mkt, xerr=err, capsize=3,
           color=["#e8590c" if "Market" in p else "#ae3ec9" if "NN" in p or "MLP" in p or "GRU" in p else "#3b5bdb"
                  for p in s.predictor])
    a.axvline(0, color="black", lw=0.8)
    a.set(xlabel="Brier minus market 1 h before tip (95% CI, game-days resampled)", title="Brier vs the market")
    a.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(e, lo, hi, d=4):
    return f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def markdown(sc, st, seeds, n_train, grid) -> str:
    lines = ["# M4-NN: neural win models vs M4 and the market", "",
             f"Trained on {n_train} games before {SPLIT} (early stopping on 1 Dec – 31 Jan, then refit on all "
             "training games for the chosen epoch count), scored on the 501 test games (1 Feb – 12 Apr 2026) with "
             "absences known, as M4. MLP: M4's six difference features → hidden → 1. GRU: shared GRU over each "
             "team's last 10 completed games plus the six features. Hidden size and learning rate chosen by mean "
             "validation BCE (table at the end); three seeds each; the forecast is the 3-seed mean. CIs resample "
             f"game-days ({REPS} replicates, seed {SEED}). \"Anchor + X shift\" = market mid 24 h before tip + "
             "model X's (absences known − absences ignored) shift: the estimate the deterministic agent trades on.",
             "",
             "## Forecast accuracy", "",
             "| Predictor | Brier | Log loss | Acc. | Brier − M4 [95% CI] | Log loss − M4 [95% CI] | "
             "Brier − market 1 h [95% CI] |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sc.itertuples():
        lines.append(f"| {r.predictor} | {r.brier:.4f} | {r.log_loss:.4f} | {r.accuracy:.1%} | "
                     f"{fmt(r.brier_minus_m4, r.brier_minus_m4_lo, r.brier_minus_m4_hi)} | "
                     f"{fmt(r.log_loss_minus_m4, r.log_loss_minus_m4_lo, r.log_loss_minus_m4_hi)} | "
                     f"{fmt(r.brier_minus_mkt, r.brier_minus_mkt_lo, r.brier_minus_mkt_hi)} |")
    lines += ["", "## Seed variability (single-seed models)", "",
              "| Model | Brier mean ± std | Brier range | Log loss mean ± std | Best epochs |", "| --- | --- | --- | --- | --- |"]
    for r in st.itertuples():
        ep = ", ".join(str(int(e)) for e in seeds[seeds.model == r.model].best_epoch)
        lines.append(f"| {r.model.upper()} | {r.brier_mean:.4f} ± {r.brier_std:.4f} | {r.brier_min:.4f} – "
                     f"{r.brier_max:.4f} | {r.log_loss_mean:.4f} ± {r.log_loss_std:.4f} | {ep} |")
    lines += ["", "## Anchor estimates vs the agent's current estimate (anchor + M4 shift)", "",
              "| Predictor | Brier − agent's estimate [95% CI] | Log loss − agent's estimate [95% CI] |",
              "| --- | --- | --- |"]
    for r in sc[sc.predictor.str.startswith("Anchor") & (sc.predictor != AGENT)].itertuples():
        lines.append(f"| {r.predictor} | {fmt(r.brier_minus_agent, r.brier_minus_agent_lo, r.brier_minus_agent_hi)} | "
                     f"{fmt(r.log_loss_minus_agent, r.log_loss_minus_agent_lo, r.log_loss_minus_agent_hi)} |")
    lines += ["", "## Validation grid (1 Dec – 31 Jan, fit on earlier games)", "",
              f"Logistic regression on the same features and slice: validation BCE {grid.attrs['logistic_val_bce']:.4f}.",
              "", "| Model | Hidden | Learning rate | Validation BCE mean ± std (3 seeds) | Chosen |",
              "| --- | --- | --- | --- | --- |"]
    for r in grid.itertuples():
        lines.append(f"| {r.model.upper()} | {r.hidden} | {r.lr:g} | {r.val_bce_mean:.4f} ± {r.val_bce_std:.4f} | "
                     f"{'yes' if r.chosen else ''} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    rows, games = load_rows()
    print(f"{len(rows)} rows [{time.time() - t0:.0f}s]", flush=True)
    seq = sequences(rows, games)
    t = pd.read_csv(RESULTS / "m4_games.csv", dtype={"game_id": str})
    rows["game_id"] = rows.game_id.astype(str)
    test_mask = rows.game_id.isin(t.game_id).to_numpy()
    te, preds, seeds, n_train, grid = train_all(rows, seq, test_mask)
    t = t.set_index("game_id").loc[te.game_id].reset_index()
    assert (t.home_win.to_numpy() == te.home_win.to_numpy()).all()
    t["anchor"] = anchors(t)
    keep = ~np.isnan(t.anchor.to_numpy())
    t = t[keep].reset_index(drop=True)
    preds = {k: (a[:, keep], b[:, keep]) for k, (a, b) in preds.items()}
    by_name = predictors(t, preds)
    sc, st = scores(t, by_name), seed_table(t, preds)
    args.out.mkdir(parents=True, exist_ok=True)
    sc.to_csv(args.out / "win_nn.csv", index=False)
    seeds.merge(st, on="model").to_csv(args.out / "win_nn_seeds.csv", index=False)
    pd.DataFrame({"game_id": t.game_id, "date": t.date, "home_win": t.home_win, **{
        f"{k}_{w}": v[i].mean(0) for k, v in preds.items() for i, w in ((0, "after"), (1, "before"))}}) \
        .to_csv(CACHE / "test_preds.csv", index=False)
    figure(t, by_name, sc, args.out / "win_nn.png")
    grid.to_csv(args.out / "win_nn_grid.csv", index=False)
    (args.out / "win_nn.md").write_text(markdown(sc, st, seeds, n_train, grid))
    print((args.out / "win_nn.md").read_text())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
