"""Train M1-M4 on games before the split, score them on games after it, save to models/.

    python -m forecast.train                                   # data/frozen, split 2026-02-01
    python -m forecast.train --source synthetic --out runs/train-synthetic

Writes held-out tables to reports/results/ (m2_heldout.csv, m3_heldout.csv,
m4_heldout.csv, m4_coefficients.json) for the report and the demo.
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from forecast import play, win
from forecast.baselines import GBMForecaster, RollingAverage, evaluate, load_source, save
from forecast.features import make_features
from forecast.gru import GRUForecaster
from forecast.history import History
from nba_agent_paths import MODELS, RESULTS



def _save(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--split", default="2026-02-01", help="first held-out date (the test period)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(args.split, tz="UTC")
    player_games, games = load_source(args.source)
    h = History(player_games, games)

    print("M1/M2: points and minutes distributions")
    rows = make_features(player_games, games)
    train, test = rows[rows.tip_time < split], rows[rows.tip_time >= split]
    print(f"  {len(train)} training rows, {len(test)} held-out rows")
    tables = []
    for model in (RollingAverage(), GBMForecaster()):
        model.fit(train)
        save(model)
        tables.append(evaluate(model.predict(test), test).assign(model=model.name))
    gru = GRUForecaster(epochs=args.epochs).fit(train, h)
    _save(gru, MODELS / "m2" / "gru.pkl")
    tables.append(evaluate(gru.predict(test, h), test).assign(model=gru.name))
    m2 = pd.concat(tables)[["model", "target", "pinball", "coverage_10_90", "median_abs_error"]]
    m2.to_csv(args.out / "m2_heldout.csv", index=False)
    print(m2.round(3).to_string(index=False))

    print("M3: play classifier")
    prows = play.training_rows(h, games)
    ptrain, ptest = prows[prows.tip_time < split], prows[prows.tip_time >= split]
    pm = play.PlayModel().fit(ptrain)
    _save(pm, MODELS / "m3" / "play.pkl")
    m3 = play.evaluate(pm, ptest)
    m3.to_csv(args.out / "m3_heldout.csv", index=False)
    print(m3.round(4).to_string(index=False))

    print("M4: win model")
    wrows = win.training_rows(h, games)
    wtrain, wtest = wrows[wrows.tip_time < split], wrows[wrows.tip_time >= split]
    wm = win.WinModel().fit(wtrain)
    _save(wm, MODELS / "m4" / "win.pkl")
    m4 = win.evaluate(wm, wtest)
    m4.to_csv(args.out / "m4_heldout.csv", index=False)
    (args.out / "m4_coefficients.json").write_text(json.dumps(wm.coefficients(), indent=2))
    print(m4.round(4).to_string(index=False))
    print({k: round(v, 4) for k, v in wm.coefficients().items()})
    print(f"models in {MODELS}, tables in {args.out}")


if __name__ == "__main__":
    main()
