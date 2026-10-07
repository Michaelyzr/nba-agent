"""M6 robustness: is the null result (no better than a zero move) just low capacity or a short price history?

    python -m evaluation.m6_robustness                 # dataset (cached in runs/m6robust/), grid, tables, chart
    python -m evaluation.m6_robustness --report-only   # rescore cached predictions

A grid declared here before any result was seen, same splits as forecast/impact.py
(train before 15 Jan, early stopping on 15-31 Jan, test 1 Feb - 12 Apr), same
pinball loss, early stopping and static features, 3 seeds per configuration:

    GRU hidden 16 / 64 / 128 over 24 x 15 min (6 h; hidden 16 is the shipped M6)
    GRU hidden 16 / 64 over 48 x 15 min (12 h)
    GRU hidden 16 / 64 over 24 x 1 h (24 h)
    MLP on the 18 static features only (no sequence)
    1D CNN (64 channels) over 48 x 15 min

The price sequence is built once at 96 x 15 min (24 h) and cut or resampled for
each configuration. Scored on the median quantile against the zero-move
baseline: MAE gap in cents with a game-clustered bootstrap 95% CI (day-clustered
too), R^2 against zero, for each seed and for the 3-seed average. M6's default
model and code path are unchanged.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import forecast.impact as impact
from forecast.impact import QUANTILES, STATIC, ImpactModel, mae_ci, metrics, split_of

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
CACHE = ROOT / "runs" / "m6robust"
FULL_STEPS = 96                           # 24 h of 15-minute steps
SEEDS = (0, 1, 2)
COST_C = 5.5                              # cents a predicted move must clear to pay spread + fee (README M6)
GRID = {                                  # name: (net, hidden, sequence)
    "GRU-16, 6 h @ 15 min (shipped)": ("gru", 16, "15m24"),
    "GRU-64, 6 h @ 15 min": ("gru", 64, "15m24"),
    "GRU-128, 6 h @ 15 min": ("gru", 128, "15m24"),
    "GRU-16, 12 h @ 15 min": ("gru", 16, "15m48"),
    "GRU-64, 12 h @ 15 min": ("gru", 64, "15m48"),
    "GRU-16, 24 h @ 1 h": ("gru", 16, "1h24"),
    "GRU-64, 24 h @ 1 h": ("gru", 64, "1h24"),
    "MLP, static features only": ("mlp", 32, "15m24"),
    "1D CNN-64, 12 h @ 15 min": ("cnn", 64, "15m48"),
}


def sequences(full: np.ndarray, kind: str) -> np.ndarray:
    """Cut the [n, 96, 4] 15-minute sequence to the last 24 or 48 steps, or resample it to 24 hourly steps."""
    if kind == "15m24":
        return full[:, -24:]
    if kind == "15m48":
        return full[:, -48:]
    hourly = full.reshape(len(full), 24, 4, full.shape[-1])
    out = hourly[:, :, -1].copy()                                    # mid, spread, has_quote at the hour's end
    out[:, :, 2] = np.log1p(np.expm1(hourly[:, :, :, 2]).sum(2))     # volume summed over the hour
    return out


def quantile_head(out):
    base, steps = out[:, :1], nn.functional.softplus(out[:, 1:])
    return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)


class StaticMLP(nn.Module):
    def __init__(self, n_seq, n_static, hidden=32):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, len(QUANTILES)))

    def forward(self, seq, static):
        return quantile_head(self.head(static))


class CNNNet(nn.Module):
    def __init__(self, n_seq, n_static, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_seq, hidden, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + n_static, 32), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(32, len(QUANTILES)))

    def forward(self, seq, static):
        h = self.conv(seq.transpose(1, 2))
        return quantile_head(self.head(torch.cat([h.mean(2), h[:, :, -1], static], dim=1)))


@contextmanager
def net_class(cls):
    """ImpactModel.fit builds forecast.impact.ImpactNet; swap in another architecture for one fit."""
    old = impact.ImpactNet
    impact.ImpactNet = cls
    try:
        yield
    finally:
        impact.ImpactNet = old


def build(rebuild=False):
    """(rows, [n, 96, 4] sequences) for every game since 1 Oct 2025, cached."""
    path = CACHE / "dataset.pkl"
    if path.exists() and not rebuild:
        return pickle.loads(path.read_bytes())
    from replay import load_tables
    tables = load_tables()
    win = pickle.loads((ROOT / "models" / "m4" / "win.pkl").read_bytes())
    games = tables["games"][tables["games"].date >= "2025-10-01"]
    impact.SEQ_LEN = FULL_STEPS                    # price_part and build_dataset read the module constant
    try:
        rows, seq = impact.build_dataset(tables, win, games)
    finally:
        impact.SEQ_LEN = 24
    rows["split"] = split_of(rows.date)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps((rows, seq)))
    return rows, seq


def fit_one(job) -> dict:
    """Train one (configuration, seed) and return median-quantile predictions on val and test."""
    torch.set_num_threads(2)
    name, seed = job
    net, hidden, kind = GRID[name]
    rows, full = build()
    seq = sequences(full, kind)
    part = {s: (rows[rows.split == s].reset_index(drop=True), seq[(rows.split == s).to_numpy()])
            for s in ("train", "val", "test")}
    t0 = time.time()
    model = ImpactModel(None, hidden=hidden, seed=seed)
    cls = {"gru": impact.ImpactNet, "mlp": StaticMLP, "cnn": CNNNet}[net]
    with net_class(cls):
        model.fit(*part["train"], *part["val"], verbose=False)
    med = QUANTILES.index(0.5)
    return {"name": name, "seed": seed, "epochs": model.best_epoch, "seconds": time.time() - t0,
            "val": model.predict_arrays(*part["val"])[:, med], "test": model.predict_arrays(*part["test"])[:, med]}


def score(rows: pd.DataFrame, preds: list) -> pd.DataFrame:
    test = rows[rows.split == "test"].reset_index(drop=True)
    y = test.move.to_numpy()
    out = []
    for name in GRID:
        mine = [p for p in preds if p["name"] == name]
        if not mine:
            continue
        ens = np.mean([p["test"] for p in mine], axis=0)
        for rowset, mask in (("decision times", test.is_decision.to_numpy()), ("all rows", np.ones(len(test), bool))):
            per_seed = [mae_ci(p["test"][mask], y[mask], test.date.to_numpy()[mask])[0] for p in mine]
            g = mae_ci(ens[mask], y[mask], test.game_id.to_numpy()[mask])
            d = mae_ci(ens[mask], y[mask], test.date.to_numpy()[mask])
            m = metrics(ens[mask], y[mask])
            big = np.abs(ens[mask]) * 100 > COST_C
            out.append({"config": name, "rows_used": rowset, "rows": int(mask.sum()), "seeds": len(mine),
                        "mae_c": m["mae_c"], "zero_mae_c": float(np.abs(y[mask]).mean() * 100),
                        "mae_gap_c": g[0], "gap_lo_game": g[1], "gap_hi_game": g[2],
                        "gap_lo_day": d[1], "gap_hi_day": d[2],
                        "gap_seed_min": min(per_seed), "gap_seed_max": max(per_seed),
                        "r2_vs_zero": m["r2_vs_zero"], "corr": m["corr"],
                        "epochs_mean": float(np.mean([p["epochs"] for p in mine])),
                        "rows_pred_over_cost": int(big.sum()),
                        "mean_signed_move_over_cost_c": float((np.sign(ens[mask][big]) * y[mask][big]).mean() * 100)
                        if big.any() else np.nan})
    return pd.DataFrame(out)


def chart(s: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, rowset in zip(axes, ("decision times", "all rows")):
        d = s[s.rows_used == rowset].iloc[::-1]
        err = np.vstack([d.mae_gap_c - d.gap_lo_game, d.gap_hi_game - d.mae_gap_c])
        ax.barh(d.config, d.mae_gap_c, xerr=err, capsize=4,
                color=["#2b8a3e" if hi < 0 else "#c92a2a" if lo > 0 else "#868e96"
                       for lo, hi in zip(d.gap_lo_game, d.gap_hi_game)])
        ax.scatter(d.gap_seed_min, d.config, marker="|", color="black", s=60, zorder=3, label="seed min / max")
        ax.scatter(d.gap_seed_max, d.config, marker="|", color="black", s=60, zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set(xlabel="test MAE minus zero-move MAE (cents; < 0 beats the martingale)",
               title=f"Test, {rowset} ({int(d.rows.iloc[0])} rows): 3-seed average, 95% game-clustered CI")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown(s: pd.DataFrame, rows: pd.DataFrame) -> str:
    counts = rows.groupby("split").agg(rows=("move", "size"), games=("game_id", "nunique"))
    lines = ["# M6 robustness grid", "",
             "Pre-declared grid, same splits, loss, early stopping and static features as `forecast/impact.py`; "
             f"3 seeds each. Rows: train {counts.loc['train', 'rows']} ({counts.loc['train', 'games']} games), "
             f"validation {counts.loc['val', 'rows']}, test {counts.loc['test', 'rows']} "
             f"({counts.loc['test', 'games']} games). Gap = MAE of the 3-seed average median prediction minus the "
             "zero-move MAE, in cents (negative beats the martingale); 95% CI resamples games (day-clustered CI in "
             "the CSV). R² is against predicting zero.", ""]
    for rowset in ("decision times", "all rows"):
        d = s[s.rows_used == rowset]
        lines += [f"## Test, {rowset}", "",
                  "| Configuration | MAE (¢) | Zero-move MAE (¢) | Gap [95% CI, games] | Gap per seed (min – max) | R² vs zero |",
                  "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {r.config} | {r.mae_c:.3f} | {r.zero_mae_c:.3f} | {r.mae_gap_c:+.3f} [{r.gap_lo_game:+.3f}, "
                  f"{r.gap_hi_game:+.3f}] | {r.gap_seed_min:+.3f} – {r.gap_seed_max:+.3f} | {r.r2_vs_zero:+.4f} |"
                  for r in d.itertuples()]
        lines.append("")
    best = s[s.rows_used == "decision times"].sort_values("mae_gap_c").iloc[0]
    beats = s[(s.gap_hi_game < 0)]
    worse = int((s.gap_lo_game > 0).sum())
    over = s[s.rows_used == "decision times"].rows_pred_over_cost
    lines += ["## Conclusion", "",
              (f"No configuration beats the zero-move baseline. {worse} of {len(s)} gap CIs (configuration x row "
               f"set) lie entirely above zero, i.e. significantly worse than predicting no move. The best on "
               f"decision times is {best.config} at {best.mae_gap_c:+.3f}¢ [{best.gap_lo_game:+.3f}, {best.gap_hi_game:+.3f}]."
               if beats.empty else
               "Configurations whose CI is below zero: " + "; ".join(
                   f"{r.config} ({r.rows_used}) {r.mae_gap_c:+.3f}¢ [{r.gap_lo_game:+.3f}, {r.gap_hi_game:+.3f}]"
                   for r in beats.itertuples()) + "."),
              f"Rows where any configuration predicts a move larger than the {COST_C}¢ needed to cover spread and "
              f"fee: at most {int(over.max())} of {int(best.rows)} decision-time rows."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached dataset")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="subset of configuration names")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    t0 = time.time()
    rows, full = build(args.rebuild)
    print(f"dataset: {len(rows)} rows, sequences {full.shape} [{time.time() - t0:.0f}s]", flush=True)
    pred_path = CACHE / "predictions.pkl"
    preds = pickle.loads(pred_path.read_bytes()) if pred_path.exists() else []
    if not args.report_only:
        done = {(p["name"], p["seed"]) for p in preds}
        jobs = [(n, s) for n in (args.only or GRID) for s in SEEDS if (n, s) not in done]
        from evaluation.m6_robustness import fit_one as fit       # picklable under "python -m"
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(fit, jobs):
                preds.append(r)
                pred_path.write_bytes(pickle.dumps(preds))
                print(f"  {r['name']} seed {r['seed']}: best epoch {r['epochs']}, {r['seconds']:.0f}s", flush=True)
    s = score(rows, preds)
    args.out.mkdir(parents=True, exist_ok=True)
    s.to_csv(args.out / "m6_robustness.csv", index=False)
    chart(s, args.out / "m6_robustness.png")
    (args.out / "m6_robustness.md").write_text(markdown(s, rows))
    print((args.out / "m6_robustness.md").read_text())


if __name__ == "__main__":
    main()
