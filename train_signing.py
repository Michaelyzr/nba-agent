"""Train the signing-pay network: one hidden layer, h = ReLU(W1 x + b1), y = W2 h + b2, loss (y - target)^2.

Reports held-out MAE against linear regression and the median annual pay. Writes models/signing/.
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from checks import signing_features
from models import SIGNING_FEATURES, build_mlp
from tables import MODELS, load

HIDDEN, EPOCHS, LR, Y_SCALE = 32, 3000, 1e-2, 1e7


def dataset():
    contracts = load("contracts")
    rows, y, split = [], [], []
    for c in contracts.itertuples():
        feats = signing_features(c.PLAYER_ID)
        if feats is None or c.ANNUAL_PAY != c.ANNUAL_PAY:
            continue
        rows.append([feats[k] for k in SIGNING_FEATURES])
        y.append(c.ANNUAL_PAY)
        split.append(c.SPLIT)
    return np.array(rows, dtype=np.float32), np.array(y, dtype=np.float32), np.array(split)


def main():
    torch.manual_seed(7606)
    X, y, split = dataset()
    tr, te = split == "train", split == "heldout"
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mean) / std

    net = build_mlp(X.shape[1], HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    xt, yt = torch.from_numpy(Xs[tr]), torch.from_numpy(y[tr] / Y_SCALE).unsqueeze(1)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xs[te])).squeeze(1).numpy() * Y_SCALE

    lin = LinearRegression().fit(Xs[tr], y[tr])
    metrics = {
        "mlp_mae": float(mean_absolute_error(y[te], pred)),
        "linreg_mae": float(mean_absolute_error(y[te], lin.predict(Xs[te]))),
        "median_mae": float(mean_absolute_error(y[te], np.full(te.sum(), np.median(y[tr])))),
        "n_train": int(tr.sum()), "n_heldout": int(te.sum()),
        "features": SIGNING_FEATURES, "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "y_scale": Y_SCALE, "hidden": HIDDEN,
    }
    out = MODELS / "signing"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"held-out MAE  network ${metrics['mlp_mae'] / 1e6:.2f}M | linear regression "
          f"${metrics['linreg_mae'] / 1e6:.2f}M | median ${metrics['median_mae'] / 1e6:.2f}M "
          f"({metrics['n_train']} train, {metrics['n_heldout']} held out)")


if __name__ == "__main__":
    main()
