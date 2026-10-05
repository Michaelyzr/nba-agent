"""Probability quality metrics; no simulated return claims."""
from collections import defaultdict
import math


def summarize(rows, total_games):
    if not rows:
        return {"sample_size": 0, "skipped": total_games, "metrics": None, "bins": [], "rows": []}
    n = len(rows)
    brier = sum((r["home_probability"] - r["home_win"]) ** 2 for r in rows) / n
    log_loss = -sum(
        r["home_win"] * math.log(max(1e-12, r["home_probability"]))
        + (1 - r["home_win"]) * math.log(max(1e-12, 1 - r["home_probability"]))
        for r in rows
    ) / n
    accuracy = sum(int(r["home_probability"] >= .5) == r["home_win"] for r in rows) / n
    buckets = defaultdict(list)
    for row in rows:
        buckets[min(9, int(row["home_probability"] * 10))].append(row)
    bins = [{
        "range": f"{i * 10}–{(i + 1) * 10}%", "count": len(bucket),
        "predicted": sum(r["home_probability"] for r in bucket) / len(bucket),
        "actual": sum(r["home_win"] for r in bucket) / len(bucket),
    } for i, bucket in sorted(buckets.items())]
    ece = sum(b["count"] * abs(b["predicted"] - b["actual"]) for b in bins) / n
    # Rank-based AUC with proper tie handling, avoiding quadratic pair comparisons.
    positive = sum(r["home_win"] for r in rows)
    negative = n - positive
    ranked = sorted(rows, key=lambda r: r["home_probability"])
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and ranked[j]["home_probability"] == ranked[i]["home_probability"]:
            j += 1
        rank_sum += sum(r["home_win"] for r in ranked[i:j]) * ((i + 1 + j) / 2)
        i = j
    auc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative) if positive and negative else None
    return {
        "sample_size": n, "skipped": total_games - n,
        "metrics": {"brier_score": brier, "log_loss": log_loss, "accuracy": accuracy,
                    "calibration_error": ece, "roc_auc": auc},
        "bins": bins, "rows": rows,
    }
