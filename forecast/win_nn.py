"""M4-NN: neural home-win models on the same no-look-ahead inputs and split as M4.

    python -m evaluation.win_nn_eval              # trains 3 seeds of each model, scores the 501 test games

Two models, both trained with binary cross-entropy on games before 2026-02-01:

  MLP   WIN_FEATURES (M4's six home-minus-away differences) -> 16 -> 1.
  GRU   a shared GRU reads each team's last SEQ_LEN completed games (margin,
        won, at home, pre-game rating difference, missing-minutes difference,
        rest difference, days since that game), seen from that team's side; the
        two final states plus WIN_FEATURES go through a small MLP to a logit.

Every sequence step is a game that finished before tonight's tip, and the
static features are M4's (History is queried by tip), so neither model sees
the future. Early stopping: fit on games before VAL_START, stop on the
validation slice VAL_START..split, then refit on every game before the split
for the chosen number of epochs (the same data M4 is fit on). The "before the
news" probability zeroes tonight's missing-minutes and missing-points
differences, exactly as M4 does, so the anchor estimate (24 h market mid +
model news shift) can use either model.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast.win import WIN_FEATURES

SEQ_LEN = 10
STEP_COLS = ["margin", "won", "home", "rating_diff", "missing_min_diff", "rest_diff", "log_days_ago", "mask"]
VAL_START = "2025-12-01"
SPLIT = "2026-02-01"
NEWS_COLS = ["missing_min_diff", "missing_pts_diff"]


def team_log(rows: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, completed game) with that game's features from the team's side."""
    g = rows.merge(games[["game_id", "home_team_id", "away_team_id", "home_pts", "away_pts"]], on="game_id")
    sides = []
    for team_col, s in (("home_team_id", 1.0), ("away_team_id", -1.0)):
        m = s * (g.home_pts - g.away_pts).to_numpy(float)
        sides.append(pd.DataFrame({"team": g[team_col].to_numpy(), "tip_time": g.tip_time.to_numpy(),
                                   "margin": m / 10, "won": (m > 0).astype(float), "home": float(s > 0),
                                   "rating_diff": s * g.rating_diff.to_numpy(float) / 10,
                                   "missing_min_diff": s * g.missing_min_diff.to_numpy(float) / 48,
                                   "rest_diff": s * g.rest_diff.to_numpy(float)}))
    return pd.concat(sides, ignore_index=True).sort_values(["team", "tip_time"]).reset_index(drop=True)


def sequences(rows: pd.DataFrame, games: pd.DataFrame, log: pd.DataFrame | None = None) -> np.ndarray:
    """[n, 2, SEQ_LEN, len(STEP_COLS)]: home then away team's last SEQ_LEN games strictly before each tip."""
    log = team_log(rows, games) if log is None else log
    teams = {t: (pd.DatetimeIndex(d.tip_time).asi8, d[STEP_COLS[:-2]].to_numpy(float))
             for t, d in log.groupby("team")}
    ids = rows[["game_id"]].merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="left")
    tips = pd.DatetimeIndex(rows.tip_time).asi8
    out = np.zeros((len(rows), 2, SEQ_LEN, len(STEP_COLS)), np.float32)
    for i, (tip, home, away) in enumerate(zip(tips, ids.home_team_id, ids.away_team_id)):
        for j, team in enumerate((home, away)):
            if team not in teams:
                continue
            ts, x = teams[team]
            k = int(np.searchsorted(ts, tip, "left"))             # games that tipped strictly before tonight
            lo = max(0, k - SEQ_LEN)
            n = k - lo
            if n == 0:
                continue
            days = (tip - ts[lo:k]) / 86_400e9
            out[i, j, SEQ_LEN - n:, :-2] = x[lo:k]
            out[i, j, SEQ_LEN - n:, -2] = np.log1p(days)
            out[i, j, SEQ_LEN - n:, -1] = 1.0
    return out


class WinNet(nn.Module):
    def __init__(self, kind="gru", n_static=len(WIN_FEATURES), n_step=len(STEP_COLS), hidden=16, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.gru = nn.GRU(n_step, hidden, batch_first=True) if kind == "gru" else None
        n_in = n_static + (2 * hidden if kind == "gru" else 0)
        self.head = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, static, seq=None):
        x = static
        if self.gru is not None:
            b = seq.shape[0]
            _, h = self.gru(seq.reshape(b * 2, *seq.shape[2:]))
            x = torch.cat([h[-1].reshape(b, -1), static], dim=1)
        return self.head(x).squeeze(-1)


class NeuralWinModel:
    """Same predict(rows) interface as forecast.win.WinModel; GRU rows also need sequences()."""

    def __init__(self, kind="gru", hidden=16, epochs=200, patience=20, lr=1e-3, weight_decay=1e-3,
                 batch=128, seed=0):
        self.kind, self.hidden, self.epochs, self.patience = kind, hidden, epochs, patience
        self.lr, self.weight_decay, self.batch, self.seed = lr, weight_decay, batch, seed
        self.name = f"win-{kind}"

    def _x(self, rows, seq):
        st = torch.from_numpy(((rows[WIN_FEATURES].to_numpy(float) - self.st_mean) / self.st_std).astype(np.float32))
        if self.kind != "gru":
            return st, None
        s = seq.copy()
        s[..., :-1] = (s[..., :-1] - self.seq_mean) / self.seq_std * s[..., -1:]     # padding stays 0
        return st, torch.from_numpy(s.astype(np.float32))

    def _train(self, x, y, epochs, val=None):
        torch.manual_seed(self.seed)
        net = WinNet(self.kind, hidden=self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_epoch, wait, curve = np.inf, epochs, 0, []
        for epoch in range(epochs):
            net.train()
            order = torch.randperm(len(y), generator=gen)
            for i in range(0, len(y), self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                loss_fn(net(x[0][idx], None if x[1] is None else x[1][idx]), y[idx]).backward()
                opt.step()
            if val is None:
                continue
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(net(*val[0]), val[1]))
            curve.append(v)
            if v < best - 1e-4:
                best, best_epoch, wait = v, epoch + 1, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.eval()
        return net, best_epoch, curve

    def fit(self, rows: pd.DataFrame, seq: np.ndarray | None = None, val_start=VAL_START):
        """Early-stop on rows from val_start on, then refit on all rows for the best epoch count."""
        np.random.seed(self.seed)
        self.st_mean = rows[WIN_FEATURES].mean().to_numpy(float)
        self.st_std = rows[WIN_FEATURES].std().replace(0, 1).fillna(1).to_numpy(float)
        if seq is not None:
            steps = seq[..., :-1][seq[..., -1] > 0]
            self.seq_mean, self.seq_std = steps.mean(0), steps.std(0) + 1e-6
        y = torch.tensor(rows.home_win.to_numpy(np.float32))
        is_val = (pd.DatetimeIndex(rows.tip_time) >= pd.Timestamp(val_start, tz="UTC"))
        fit_m, val_m = ~is_val, is_val
        x = self._x(rows, seq)

        def take(m):
            idx = torch.from_numpy(np.flatnonzero(m))
            return (x[0][idx], None if x[1] is None else x[1][idx]), y[idx]

        if val_m.any() and fit_m.sum() > 50:
            (xf, yf), (xv, yv) = take(fit_m), take(val_m)
            _, self.best_epoch, self.curve = self._train(xf, yf, self.epochs, val=(xv, yv))
        else:
            self.best_epoch, self.curve = self.epochs, []
        self.net, _, _ = self._train(x, y, self.best_epoch)
        return self

    def predict(self, rows: pd.DataFrame, seq: np.ndarray | None = None) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.net(*self._x(rows, seq))).numpy().astype(float)

    def predict_before_news(self, rows: pd.DataFrame, seq: np.ndarray | None = None) -> np.ndarray:
        return self.predict(rows.assign(**{c: 0.0 for c in NEWS_COLS}), seq)
