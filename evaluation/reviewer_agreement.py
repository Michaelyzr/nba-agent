"""Cohen's kappa for reviewer blame labels (two annotators).

    python -m evaluation.reviewer_agreement
    python -m evaluation.reviewer_agreement evaluation/labels/reviewer_label_sheet.csv

Expects columns human_label_a and human_label_b (or human_label + a second
annotator column). Does not invent labels; prints "pending annotation" when
either column is empty.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = Path(__file__).resolve().parent / "labels" / "reviewer_label_sheet.csv"
CATEGORIES = ("news_stale", "already_moved", "longshot", "thin_book", "model_error",
              "bad_fill", "other", "unclear")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Unweighted Cohen's kappa on two categorical label series (same index)."""
    cats = sorted(set(a) | set(b))
    if not cats:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    mat = np.zeros((len(cats), len(cats)), float)
    for x, y in zip(a, b):
        mat[idx[x], idx[y]] += 1
    mat /= n
    po = float(np.trace(mat))
    pe = float(mat.sum(0) @ mat.sum(1))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", nargs="?", type=Path, default=LABELS)
    args = ap.parse_args()
    frame = pd.read_csv(args.sheet)
    cols = [c for c in frame.columns if c.startswith("human_label")]
    if len(cols) < 2:
        print(f"{args.sheet}: need two human_label* columns; found {cols or 'none'}. "
              "Pending annotation.")
        return
    a, b = frame[cols[0]].fillna("").astype(str).str.strip(), frame[cols[1]].fillna("").astype(str).str.strip()
    done = (a != "") & (b != "")
    if done.sum() < 2:
        print(f"{args.sheet}: {done.sum()}/{len(frame)} rows have both annotators. "
              "Pending annotation — do not report kappa yet.")
        return
    k = cohen_kappa(a[done], b[done])
    print(f"n={int(done.sum())}  Cohen's kappa={k:.3f}")
    print("categories observed:", sorted(set(a[done]) | set(b[done])))


if __name__ == "__main__":
    main()
