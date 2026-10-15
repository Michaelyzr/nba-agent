"""Edge scan: combine parts 3a (injury reports), 3b (trade timing) and 3c (prop-data probe) into one CSV.

    python -m evaluation.edge_scan_parse_injuries && python -m evaluation.edge_scan_injuries
    python -m evaluation.edge_scan_timing
    python -m evaluation.edge_scan_props_probe
    python -m evaluation.edge_scan            -> evaluation/results/edge_scan.csv
The narrative and go/no-go table live in evaluation/results/edge_scan.md.
"""
from __future__ import annotations

import json

import pandas as pd

from evaluation.edge_scan_common import RESULTS, RUNS


def main():
    parts = []
    if (RUNS / "injury_moves.csv").exists():
        parts.append(pd.read_csv(RUNS / "injury_moves.csv"))
    if (RUNS / "timing.csv").exists():
        parts.append(pd.read_csv(RUNS / "timing.csv"))
    if (RUNS / "props_probe.json").exists():
        p = json.loads((RUNS / "props_probe.json").read_text())
        for series, r in p.get("kalshi", {}).items():
            if not isinstance(r, dict) or series in ("candles", "trades"):
                continue
            for path, x in r.items():
                parts.append(pd.DataFrame([{"part": "3c", "analysis": f"kalshi {path}", "bucket": series,
                                            "n": x.get("n"), "days": x.get("events"),
                                            "mean": x.get("median_volume")}]))
    if (RUNS / "props_history.json").exists():
        h = json.loads((RUNS / "props_history.json").read_text())
        k, pm = h["kalshi"], h["polymarket"]
        parts.append(pd.DataFrame([
            {"part": "3c", "analysis": "kalshi KXNBAPTS Dec-Apr", "bucket": "markets / games / median volume",
             "n": h["kalshi_pts_markets_dec_apr"], "days": h["kalshi_pts_events_dec_apr"],
             "mean": h["kalshi_pts_volume_quartiles"][1]},
            {"part": "3c", "analysis": "kalshi 1-min candles with bid", "bucket": "sampled markets with history",
             "n": len(k), "days": sum(x["with_bid"] > 0 for x in k)},
            {"part": "3c", "analysis": "polymarket points Dec-Apr", "bucket": "markets / games / median volume $",
             "n": h["poly_points_markets_dec_apr"], "days": h["poly_events_dec_apr"],
             "mean": h["poly_prop_volume_quartiles"][1]},
            {"part": "3c", "analysis": "polymarket prices-history", "bucket": "sampled markets with history",
             "n": len(pm), "days": sum(x["points"] > 0 for x in pm)}]))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULTS / "edge_scan.csv", index=False)
    print(f"{len(out)} rows -> {RESULTS / 'edge_scan.csv'}")


if __name__ == "__main__":
    main()
