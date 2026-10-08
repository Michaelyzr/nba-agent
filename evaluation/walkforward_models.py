"""Walk-forward refits of M4 (win) and M6 (impact) for an evaluation window before the default 1 Feb split.

    python -m evaluation.walkforward_models --split 2025-12-01 --val-from 2025-11-20

M4 is fit on games tipping before SPLIT; M6 on Kalshi-era games before VAL_FROM, early-stopped on VAL_FROM..SPLIT,
using the refit M4 for its news features. Nothing dated on or after SPLIT is read for fitting. Writes
runs/walkforward/SPLIT/{m4/win.pkl, m6/impact.pkl}; point NBA_MODEL_DIR there for llm_agent_eval.
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    from evaluation.llm_agent_eval import load_frozen
    from forecast.history import History
    from forecast.impact import ImpactModel, build_dataset
    from forecast.win import WinModel, training_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2025-12-01")
    ap.add_argument("--val-from", default="2025-11-20")
    ap.add_argument("--season-start", default="2025-10-01")
    args = ap.parse_args()
    out = ROOT / "runs" / "walkforward" / args.split
    tables = load_frozen()
    games, split = tables["games"], pd.Timestamp(args.split, tz="UTC")
    h = History(tables["player_games"], games)

    rows = training_rows(h, games[games.tip_time < split])
    win = WinModel().fit(rows)
    print(f"M4: {len(rows)} training games before {args.split}; {({k: round(v, 4) for k, v in win.coefficients().items()})}")
    (out / "m4").mkdir(parents=True, exist_ok=True)
    (out / "m4" / "win.pkl").write_bytes(pickle.dumps(win))

    pre = games[(games.date >= args.season_start) & (games.tip_time < split)]
    m6_rows, seq = build_dataset(tables, win, pre, h)
    val = (m6_rows.date >= args.val_from).to_numpy()
    print(f"M6: {int((~val).sum())} train rows / {m6_rows[~val].game_id.nunique()} games, "
          f"{int(val.sum())} val rows / {m6_rows[val].game_id.nunique()} games")
    impact = ImpactModel(win).fit(m6_rows[~val].reset_index(drop=True), seq[~val],
                                  m6_rows[val].reset_index(drop=True), seq[val])
    (out / "m6").mkdir(parents=True, exist_ok=True)
    (out / "m6" / "impact.pkl").write_bytes(pickle.dumps(impact))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
