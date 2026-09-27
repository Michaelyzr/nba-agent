"""Fine-tune cross-encoder/ms-marco-MiniLM-L6-v2 on (question, recap or paragraph, relevant?) labels.

Reads data/eval/rank_labels.csv (split = train/test). Writes models/ranker/.
Recall@5 against BM25 is reported by evaluate.py.
"""
import argparse

import pandas as pd
from sentence_transformers import CrossEncoder

from tables import EVAL, MODELS

BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"


def fit(labels, out, epochs, base=BASE):
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample

    model = CrossEncoder(base, num_labels=1)
    examples = [InputExample(texts=[r.query if "query" in labels else r.sentence,
                                    r.text if "text" in labels else r.evidence], label=float(r.label))
                for r in labels.itertuples()]
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=max(1, len(loader) // 10), show_progress_bar=True)
    model.save(str(out))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    labels = pd.read_csv(EVAL / "rank_labels.csv")
    train = labels[labels.split == "train"]
    print(f"fine-tuning ranker on {len(train)} pairs ({int(train.label.sum())} relevant)")
    fit(train, MODELS / "ranker", args.epochs)
    print("saved models/ranker; run `python evaluate.py` for Recall@5 against BM25")


if __name__ == "__main__":
    main()
