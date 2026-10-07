"""M6: market-impact model, graded by the market's own reaction to public news.

    python -m forecast.impact                       # dataset, ImpactNet, baselines, held-out tables, models/m6/impact.pkl

One row per (game, decision time) between tip - 6 h and tip - 15 min: the
replay's own decision times (news inside NEWS_WINDOW, tip - LEAD) plus a grid
every 30 minutes. The label is the move of the home market's mid from the
decision time to tip-off, so the market grades the model and no settlement is
used. Inputs are as-of only: the home market's last 6 hours of quotes (24
steps of 15 minutes: mid relative to now, spread, volume), the current quote,
the mid 24 h before tip, the clock, the latest news, M4 before and after the
news and the missing rotation minutes and points per team.

ImpactNet: a GRU over the quote sequence plus the static features, outputting
QUANTILES of the move (in cents), made monotone with softplus increments and
trained with pinball loss. Early stopping on 15-31 Jan; test 1 Feb - 12 Apr;
games after 12 Apr (play-in and playoffs) are a holdout scored once.
Baselines on the same rows: zero move (the market is a martingale), linear
regression and gradient boosting on the static features, and a fitted
coefficient times M4's news shift.

The same row builder serves training (full tables, History queried by tip)
and the agent (an AsOf view), so the live features match the training ones.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from agents.graph import OUT_STATUSES
from forecast.history import History
from forecast.win import team_side
from replay import LEAD, MAX_QUOTE_AGE, NEWS_WINDOW

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "m6" / "impact.pkl"
RESULTS = ROOT / "evaluation" / "results"
QUANTILES = (0.1, 0.5, 0.9)
SEQ_LEN = 24
STEP = pd.Timedelta(minutes=15)
SAMPLE_FROM = pd.Timedelta(hours=6)
SAMPLE_EVERY = pd.Timedelta(minutes=30)
LAST_SAMPLE = pd.Timedelta(minutes=15)
ANCHOR_LEAD = pd.Timedelta(hours=24)
VOLUME_LOOKBACK = pd.Timedelta(minutes=60)
NEWS_AGE_CAP = 360.0             # minutes; also the value when a game has no news yet
SEQ_COLS = ["dmid_c", "spread_c", "log_volume", "has_quote"]
STATIC = ["mid", "spread_c", "log_volume_60", "anchor", "move_since_anchor_c", "hours_to_tip", "has_news",
          "news_age_min", "m4_before", "m4_after", "m4_shift_c", "m4_gap_c", "n_out_home", "n_out_away",
          "missing_min_home", "missing_min_away", "missing_pts_home", "missing_pts_away"]
TRAIN_END, VAL_END, TEST_END = "2026-01-15", "2026-02-01", "2026-04-13"     # first date not in each split


def _ns(t) -> int:
    t = pd.Timestamp(t)
    return (t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")).as_unit("ns").value


def price_arrays(p: pd.DataFrame) -> dict:
    p = p.sort_values("ts")
    ts = pd.DatetimeIndex(p.ts)
    ts = (ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")).as_unit("ns").asi8
    bid, ask = p.bid.to_numpy(float), p.ask.to_numpy(float)
    vol = np.nan_to_num(p.volume.to_numpy(float))
    return {"ts": ts, "mid": (bid + ask) / 2, "spread": ask - bid, "cum": np.concatenate([[0.0], np.cumsum(vol)])}


def price_part(a: dict, now, tip):
    """Quote features using only prints at or before `now`; None when the quote is stale (the agent would not act)."""
    ts, now_ns = a["ts"], _ns(now)
    k = int(np.searchsorted(ts, now_ns, "right"))
    if k == 0 or now_ns - ts[k - 1] > MAX_QUOTE_AGE.value:
        return None
    mid, spread = a["mid"][k - 1], a["spread"][k - 1]
    vol60 = a["cum"][k] - a["cum"][np.searchsorted(ts, now_ns - VOLUME_LOOKBACK.value, "right")]
    grid = now_ns - STEP.value * np.arange(SEQ_LEN - 1, -1, -1)
    hi = np.searchsorted(ts, grid, "right")
    ok, j = hi > 0, np.maximum(hi - 1, 0)
    vol = a["cum"][hi] - a["cum"][np.searchsorted(ts, grid - STEP.value, "right")]
    seq = np.stack([np.where(ok, (a["mid"][j] - mid) * 100, 0.0), np.where(ok, a["spread"][j] * 100, 0.0),
                    np.log1p(np.maximum(vol, 0)), ok.astype(float)], axis=1)
    e = int(np.searchsorted(ts, _ns(tip) - ANCHOR_LEAD.value, "right"))
    anchor = a["mid"][e - 1] if e > 0 else a["mid"][0]          # matches MarketAgent._anchor_mid
    return seq.astype(np.float32), {"mid": mid, "spread_c": spread * 100, "log_volume_60": np.log1p(max(vol60, 0)),
                                    "anchor": anchor, "move_since_anchor_c": (mid - anchor) * 100}


def mid_at(a: dict, t) -> float:
    k = int(np.searchsorted(a["ts"], _ns(t), "right"))
    return float(a["mid"][k - 1]) if k else np.nan


def news_part(news: pd.DataFrame, now, tip) -> tuple:
    """(players ruled out, minutes since the latest item) from news public at `now`, as the agent's investigator reads it."""
    n = news[(news.published_at <= now) & (news.published_at >= tip - NEWS_WINDOW)]
    latest = n.sort_values("published_at").groupby("player_id").tail(1)
    out = tuple(sorted(p for p, s in zip(latest.player_id, latest.status) if str(s).lower() in OUT_STATUSES))
    age = min(pd.Timedelta(now - n.published_at.max()).total_seconds() / 60, NEWS_AGE_CAP) if len(n) else None
    return out, age


def team_part(h: History, win, game, out: tuple) -> dict:
    home = team_side(h, game.home_team_id, game.tip_time, set(out))
    away = team_side(h, game.away_team_id, game.tip_time, set(out))
    row = {f"{k}_diff": home[k] - away[k] for k in ("rating", "margin10", "rest", "b2b", "missing_min", "missing_pts")}
    after = float(win.predict(pd.DataFrame([row]))[0])
    before = float(win.predict(pd.DataFrame([{**row, "missing_min_diff": 0.0, "missing_pts_diff": 0.0}]))[0])
    return {"m4_before": before, "m4_after": after, "m4_shift_c": (after - before) * 100,
            "n_out_home": float(sum(p in out for p in home["rotation"])),
            "n_out_away": float(sum(p in out for p in away["rotation"])),
            "missing_min_home": home["missing_min"], "missing_min_away": away["missing_min"],
            "missing_pts_home": home["missing_pts"], "missing_pts_away": away["missing_pts"]}


def build_row(a: dict, news: pd.DataFrame, h: History, win, game, now, cache: dict | None = None):
    """(sequence [SEQ_LEN, 4], static dict) for the home market at `now`, or None without a fresh quote."""
    priced = price_part(a, now, game.tip_time)
    if priced is None:
        return None
    seq, static = priced
    out, age = news_part(news, now, game.tip_time)
    key = (game.game_id, out)
    cache = {} if cache is None else cache
    if key not in cache:
        cache[key] = team_part(h, win, game, out)
    static.update(cache[key])
    static.update(hours_to_tip=pd.Timedelta(game.tip_time - now).total_seconds() / 3600,
                  has_news=float(age is not None), news_age_min=NEWS_AGE_CAP if age is None else age,
                  m4_gap_c=(static["m4_after"] - static["mid"]) * 100)
    return seq, static


def home_ticker(markets: pd.DataFrame, game):
    m = markets[(markets.game_id == game.game_id) & (markets.kind == "game") & (markets.team == game.home_team)]
    return None if m.empty else m.market_ticker.iloc[0]


def decision_times(game, news: pd.DataFrame) -> list:
    """Same as replay.Replay.decision_times; `news` is this game's news."""
    tip = game.tip_time
    times = news.loc[(news.published_at >= tip - NEWS_WINDOW) & (news.published_at < tip), "published_at"]
    return sorted(set(times) | {tip - LEAD})


def sample_times(game) -> list:
    grid = pd.date_range(game.tip_time - SAMPLE_FROM, game.tip_time - SAMPLE_EVERY, freq=SAMPLE_EVERY)
    return list(grid) + [game.tip_time - LAST_SAMPLE]


def build_dataset(tables: dict, win, games: pd.DataFrame | None = None, history: History | None = None):
    """(rows, sequences) for every game with a home game-winner market. History is queried by tip, so leakage-safe."""
    games = tables["games"] if games is None else games
    h = history or History(tables["player_games"], tables["games"])
    prices = {k: v for k, v in tables["prices"].groupby("market_ticker")}
    news_by_game = {k: v for k, v in tables["news"].groupby("game_id")}
    empty = tables["news"].iloc[0:0]
    rows, seqs, cache = [], [], {}
    for g in games.dropna(subset=["tip_time"]).itertuples():
        ticker = home_ticker(tables["markets"], g)
        if ticker is None or ticker not in prices:
            continue
        a, news = price_arrays(prices[ticker]), news_by_game.get(g.game_id, empty)
        close = mid_at(a, g.tip_time)
        real = set(decision_times(g, news))
        for now in sorted(real | set(sample_times(g))):
            built = build_row(a, news, h, win, g, now, cache)
            if built is None or np.isnan(close):
                continue
            seqs.append(built[0])
            rows.append({"game_id": g.game_id, "date": g.date, "as_of": now, "market_ticker": ticker,
                         "is_decision": now in real, **built[1], "close": close, "move": close - built[1]["mid"]})
    return pd.DataFrame(rows), np.asarray(seqs, np.float32).reshape(-1, SEQ_LEN, len(SEQ_COLS))


# ---------------- model ----------------

class ImpactNet(nn.Module):
    def __init__(self, n_seq=len(SEQ_COLS), n_static=len(STATIC), hidden=16):
        super().__init__()
        self.gru = nn.GRU(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(32, len(QUANTILES)))

    def forward(self, seq, static):
        _, h = self.gru(seq)
        out = self.head(torch.cat([h[-1], static], dim=1))
        base, steps = out[:, :1], nn.functional.softplus(out[:, 1:])
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)


def pinball(pred, y):
    q = torch.tensor(QUANTILES, dtype=pred.dtype)
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


class ImpactModel:
    """M6. Moves are learned in cents; predict() returns them as probability points."""
    name = "impact-gru"

    def __init__(self, win=None, epochs=300, batch=256, lr=1e-3, hidden=16, patience=20, seed=0):
        self.win = win
        self.epochs, self.batch, self.lr, self.hidden, self.patience, self.seed = epochs, batch, lr, hidden, patience, seed
        self._history = {}

    def _tensors(self, rows: pd.DataFrame, seq: np.ndarray):
        s = (seq - self.seq_mean) / self.seq_std
        st = ((rows[STATIC].to_numpy(float) - self.st_mean) / self.st_std).astype(np.float32)
        return torch.from_numpy(s.astype(np.float32)), torch.from_numpy(np.nan_to_num(st))

    def fit(self, rows, seq, val_rows, val_seq, verbose=True):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        flat = seq.reshape(-1, seq.shape[-1])
        self.seq_mean, self.seq_std = flat.mean(0), flat.std(0) + 1e-6
        self.seq_mean[-1], self.seq_std[-1] = 0.0, 1.0                # keep the mask channel as 0/1
        self.st_mean = rows[STATIC].mean().to_numpy(float)
        self.st_std = rows[STATIC].std().replace(0, 1).fillna(1).to_numpy(float)
        x_seq, x_st = self._tensors(rows, seq)
        v_seq, v_st = self._tensors(val_rows, val_seq)
        y = torch.tensor(rows.move.to_numpy(np.float32) * 100)
        vy = torch.tensor(val_rows.move.to_numpy(np.float32) * 100)
        self.net = ImpactNet(seq.shape[-1], len(STATIC), self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state, wait, self.curve = np.inf, None, 0, []
        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(y))
            for i in range(0, len(y), self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                pinball(self.net(x_seq[idx], x_st[idx]), y[idx]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(pinball(self.net(v_seq, v_st), vy))
            self.curve.append(val)
            if val < best - 1e-4:
                best, best_state, wait, self.best_epoch = val, {k: v.clone() for k, v in self.net.state_dict().items()}, 0, epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if verbose:
            print(f"  ImpactNet: best validation pinball {best:.4f} cents at epoch {self.best_epoch} of {epoch + 1}")
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_arrays(self, rows: pd.DataFrame, seq: np.ndarray) -> np.ndarray:
        """[n, len(QUANTILES)] predicted moves of the home mid to tip, in probability points, ordered."""
        x_seq, x_st = self._tensors(rows, seq)
        with torch.no_grad():
            return self.net(x_seq, x_st).numpy() / 100

    def history(self, view) -> History:
        key = (id(view._t), view.now.tz_convert("America/New_York").date())
        if key not in self._history:
            self._history = {key: History(view.player_games(), view.games())}
        return self._history[key]

    def predict(self, view, game, now) -> dict | None:
        """Agent interface: current home mid and predicted move to tip, from an as-of view; None without a fresh quote."""
        ticker = home_ticker(view.markets(game.game_id), game)
        if ticker is None:
            return None
        p = view.prices(ticker)
        if p.empty:
            return None
        built = build_row(price_arrays(p), view.news(game.game_id), self.history(view), self.win, game, now)
        if built is None:
            return None
        q = self.predict_arrays(pd.DataFrame([built[1]]), built[0][None])[0]
        return {"ticker": ticker, "mid": float(built[1]["mid"]), "move": float(q[QUANTILES.index(0.5)]),
                "q10": float(q[0]), "q90": float(q[-1]), "m4_shift": built[1]["m4_shift_c"] / 100}

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_history"}

    def __setstate__(self, state):
        self.__dict__.update(state, _history={})


def load_impact(path: Path = MODEL_PATH) -> ImpactModel:
    if not Path(path).exists():
        raise FileNotFoundError(f"no M6 impact model at {path}; train it with python -m forecast.impact")
    return pickle.loads(Path(path).read_bytes())


# ---------------- baselines and metrics ----------------

class Baselines:
    """Zero move, linear regression, gradient boosting and k x M4 news shift, fitted on the same training rows."""

    def fit(self, rows: pd.DataFrame):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x, y = rows[STATIC].to_numpy(float), rows.move.to_numpy(float)
        self.linear = make_pipeline(StandardScaler(), LinearRegression()).fit(x, y)
        self.gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                                                 l2_regularization=1.0, random_state=0).fit(x, y)
        s = rows.m4_shift_c.to_numpy(float) / 100
        self.shift_k = float((s * y).sum() / (s * s).sum()) if (s * s).sum() > 0 else 0.0
        return self

    def predict(self, rows: pd.DataFrame) -> dict:
        x = rows[STATIC].to_numpy(float)
        return {"zero move": np.zeros(len(rows)), "linear": self.linear.predict(x), "gbm": self.gbm.predict(x),
                "M4 shift only": self.shift_k * rows.m4_shift_c.to_numpy(float) / 100}


def metrics(pred, y) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    err = (pred - y) * 100
    big = np.abs(pred) > 0.01
    return {"rows": len(y), "mae_c": float(np.abs(err).mean()), "rmse_c": float(np.sqrt((err ** 2).mean())),
            "r2_vs_zero": float(1 - (err ** 2).sum() / ((y * 100) ** 2).sum()),
            "dir_rows": int(big.sum()),
            "dir_acc": float((np.sign(pred[big]) == np.sign(y[big])).mean()) if big.any() else np.nan,
            "corr": float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 and y.std() > 0 else np.nan}


def mae_ci(pred, y, days, reps=2000, seed=0) -> tuple:
    """MAE minus zero-move MAE (cents), with a 95% CI that resamples game-days."""
    gain = (np.abs(pred - y) - np.abs(y)) * 100
    frame = pd.DataFrame({"d": days, "g": gain}).groupby("d").g.agg(["sum", "count"])
    s, c = frame["sum"].to_numpy(), frame["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (reps, len(s)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def split_of(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select([dates < TRAIN_END, dates < VAL_END, dates < TEST_END],
                               ["train", "val", "test"], "holdout"), index=dates.index)


def chart(test: pd.DataFrame, preds: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = test.is_decision.to_numpy()
    y = test.move.to_numpy() * 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.scatter(preds["M6 ImpactNet"][real] * 100, y[real], s=6, alpha=0.35, label="real decision times")
    lim = max(5, np.nanpercentile(np.abs(y[real]), 99))
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="M6 predicted move to tip (cents)",
           ylabel="realised move to tip (cents)", title="Test: predicted vs realised (decision times)")
    ax = axes[1]
    for name in ("M6 ImpactNet", "gbm", "linear"):
        p = preds[name] * 100
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        ax.plot(cal.p, cal.y, marker="o", label=name)
    ax.plot([-3, 3], [-3, 3], color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set(xlabel="mean prediction in decile (cents)", ylabel="mean realised move (cents)",
           title="Test: calibration by prediction decile (all rows)")
    ax.legend(fontsize=8)
    ax = axes[2]
    names = [n for n in preds if n != "zero move"]
    stats = [mae_ci(preds[n][real], test.move.to_numpy()[real], test.date.to_numpy()[real]) for n in names]
    est = np.array([s[0] for s in stats])
    err = np.array([[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]])
    ax.barh(names, est, xerr=err, color=["C3" if e > 0 else "C2" for e in est], capsize=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="MAE minus zero-move MAE (cents; < 0 beats the martingale)",
           title="Test: error vs zero move, 95% day-clustered CI")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from data_sources import FROZEN
    from forecast.impact import Baselines, ImpactModel, build_dataset
    from replay import load_tables

    ap = argparse.ArgumentParser()
    ap.add_argument("--season-start", default="2025-10-01")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--data", type=Path, default=FROZEN, help="folder with the frozen tables")
    args = ap.parse_args()
    tables = load_tables(args.data)
    win = pickle.loads((ROOT / "models" / "m4" / "win.pkl").read_bytes())     # trained on games before 2026-02-01
    games = tables["games"][tables["games"].date >= args.season_start]
    rows, seq = build_dataset(tables, win, games)
    rows["split"] = split_of(rows.date)
    print(rows.groupby("split").agg(rows=("move", "size"), games=("game_id", "nunique"),
                                    decision_rows=("is_decision", "sum")).to_string())
    part = {s: (rows[rows.split == s].reset_index(drop=True), seq[(rows.split == s).to_numpy()])
            for s in ("train", "val", "test", "holdout")}
    model = ImpactModel(win, epochs=args.epochs).fit(*part["train"], *part["val"])
    base = Baselines().fit(part["train"][0])
    print(f"  M4 shift only: move = {base.shift_k:+.3f} x news shift")
    table, test_preds = [], None
    for split in ("val", "test", "holdout"):
        r, s = part[split]
        if r.empty:
            continue
        q = model.predict_arrays(r, s)
        preds = {"M6 ImpactNet": q[:, QUANTILES.index(0.5)], **base.predict(r)}
        if split == "test":
            test_preds = preds
        for rowset, mask in (("decision times", r.is_decision.to_numpy()), ("all rows", np.ones(len(r), bool))):
            for name, p in preds.items():
                extra = {}
                if name == "M6 ImpactNet":
                    y = r.move.to_numpy()[mask]
                    extra["coverage_10_90"] = float(((q[mask, 0] <= y) & (y <= q[mask, -1])).mean())
                d, lo, hi = mae_ci(p[mask], r.move.to_numpy()[mask], r.date.to_numpy()[mask])
                table.append({"split": split, "rows_used": rowset, "model": name, **metrics(p[mask], r.move.to_numpy()[mask]),
                              "mae_minus_zero_c": d, "mae_minus_zero_lo": lo, "mae_minus_zero_hi": hi, **extra})
    table = pd.DataFrame(table)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "m6_heldout.csv", index=False)
    show = ["split", "rows_used", "model", "rows", "mae_c", "rmse_c", "r2_vs_zero", "dir_acc", "dir_rows", "corr",
            "mae_minus_zero_c", "mae_minus_zero_lo", "mae_minus_zero_hi"]
    print(table[show].round(4).to_string(index=False))
    chart(part["test"][0], test_preds, args.out / "m6_vs_baselines.png")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model))
    print(f"saved {MODEL_PATH}; tables and chart in {args.out}")


if __name__ == "__main__":
    main()
