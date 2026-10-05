"""M2: GRU over each player's last 20 games, outputting points and minutes quantiles.

    python -m forecast.gru --source frozen --split 2026-02-01

Input: the player's last SEQ_LEN played games (minutes, points, shot attempts,
started, home, rest), left-padded with a mask channel, plus the shared M1
features for tonight (teammates out, opponent defence, rest, home). Output: the
QUANTILES of points and minutes, made monotone by adding softplus increments,
trained with pinball loss. Early stopping on the last 10% of the training
period, in time order. Compared with M1 on the same held-out games.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast.baselines import QUANTILES, TARGETS, _qcols, evaluate
from forecast.features import FEATURES
from forecast.history import SEQ_COLS, History

SEQ_LEN = 20
SCALE = 10.0                     # targets are learned in tens of points / minutes


class Net(nn.Module):
    def __init__(self, n_seq, n_static, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(64, len(TARGETS) * len(QUANTILES)))

    def forward(self, seq, static):
        _, h = self.gru(seq)
        out = self.head(torch.cat([h[-1], static], dim=1)).view(-1, len(TARGETS), len(QUANTILES))
        base, steps = out[..., :1], nn.functional.softplus(out[..., 1:])
        return torch.cat([base, base + torch.cumsum(steps, dim=-1)], dim=-1)


def pinball(pred, y):
    q = torch.tensor(QUANTILES, dtype=pred.dtype)
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


class GRUForecaster:
    name = "gru"

    def __init__(self, epochs=30, batch=512, lr=2e-3, hidden=64, seed=0):
        self.epochs, self.batch, self.lr, self.hidden, self.seed = epochs, batch, lr, hidden, seed

    def _arrays(self, rows: pd.DataFrame, h: History):
        seq = np.zeros((len(rows), SEQ_LEN, len(SEQ_COLS) + 1), dtype=np.float32)
        for i, (pid, tip) in enumerate(zip(rows.player_id, rows.tip_time)):
            r = h.player_rows(pid, tip, SEQ_LEN)
            if len(r):
                seq[i, SEQ_LEN - len(r):, :-1] = (r - self.seq_mean) / self.seq_std
                seq[i, SEQ_LEN - len(r):, -1] = 1.0
        static = ((rows[self.features].astype(float) - self.st_mean) / self.st_std).fillna(0.0).to_numpy(np.float32)
        return torch.from_numpy(seq), torch.from_numpy(static)

    def fit(self, rows: pd.DataFrame, h: History, verbose=True):
        torch.manual_seed(self.seed)
        rows = rows.sort_values("tip_time")
        self.features = [f for f in FEATURES if rows[f].notna().any()]
        sample = np.concatenate([h.player_rows(p, t, SEQ_LEN) for p, t in
                                 zip(rows.player_id[::50], rows.tip_time[::50])] or [np.zeros((1, len(SEQ_COLS)))])
        self.seq_mean, self.seq_std = sample.mean(0), sample.std(0) + 1e-6
        self.st_mean, self.st_std = rows[self.features].mean(), rows[self.features].std().replace(0, 1)
        seq, static = self._arrays(rows, h)
        y = torch.tensor(rows[list(TARGETS)].to_numpy(np.float32) / SCALE)
        cut = int(len(rows) * 0.9)
        self.net = Net(seq.shape[2], static.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, patience = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(cut)
            for i in range(0, cut, self.batch):
                idx = order[i:i + self.batch]
                opt.zero_grad()
                loss = pinball(self.net(seq[idx], static[idx]), y[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(pinball(self.net(seq[cut:], static[cut:]), y[cut:]))
            if verbose:
                print(f"  epoch {epoch + 1}: validation pinball {val * SCALE:.3f}")
            if val < best - 1e-4:
                best, best_state, patience = val, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 4:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, rows: pd.DataFrame, h: History) -> pd.DataFrame:
        seq, static = self._arrays(rows, h)
        with torch.no_grad():
            q = np.clip(self.net(seq, static).numpy() * SCALE, 0, None)
        out = pd.DataFrame(index=rows.index)
        for t_i, t in enumerate(TARGETS):
            for q_i, col in enumerate(_qcols(t)):
                out[col] = q[:, t_i, q_i]
        return out


def main():
    from forecast.baselines import MODEL_DIR, GBMForecaster, RollingAverage, load_source
    from forecast.features import make_features
    from forecast.gru import GRUForecaster

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--split", default="2026-02-01")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    player_games, games = load_source(args.source)
    rows = make_features(player_games, games)
    h = History(player_games, games)
    split = pd.Timestamp(args.split, tz="UTC")
    train, test = rows[rows.tip_time < split], rows[rows.tip_time >= split]
    print(f"{len(train)} training rows, {len(test)} test rows")
    results = []
    for model in (RollingAverage().fit(train), GBMForecaster().fit(train)):
        results.append(evaluate(model.predict(test), test).assign(model=model.name))
    gru = GRUForecaster(epochs=args.epochs).fit(train, h)
    results.append(evaluate(gru.predict(test, h), test).assign(model=gru.name))
    table = pd.concat(results)[["model", "target", "pinball", "coverage_10_90", "median_abs_error"]]
    print(table.round(3).to_string(index=False))
    path = MODEL_DIR.parent / "m2" / "gru.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(gru))
    table.to_csv(path.parent / "heldout.csv", index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
