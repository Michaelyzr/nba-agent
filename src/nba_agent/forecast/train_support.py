"""Fine-tune a second copy of the cross-encoder to judge (note sentence, checked figures) -> supported?

Reads data/eval/support_labels.csv (split = train/test). Writes models/support/.
F1 against a majority-class baseline is reported by evaluate.py.
"""
import argparse

import pandas as pd

from nba_agent.data.tables import EVAL, MODELS
from nba_agent.forecast.train_ranker import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    labels = pd.read_csv(EVAL / "support_labels.csv")
    train = labels[labels.split == "train"]
    print(f"fine-tuning support check on {len(train)} pairs ({int(train.label.sum())} supported)")
    fit(train, MODELS / "support", args.epochs)
    print("saved models/support; run `python evaluate.py` for F1")


if __name__ == "__main__":
    main()
