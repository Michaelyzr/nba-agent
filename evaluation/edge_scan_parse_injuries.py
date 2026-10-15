"""Parse official NBA injury report PDFs into (report_time, matchup, team, player, status) rows.

Uses pypdf text coordinates; the reports have a fixed column layout.
    for i in 0 1 2 3 4 5; do python -m evaluation.edge_scan_parse_injuries --shard $i/6 & done; wait
    python -m evaluation.edge_scan_parse_injuries --merge 6
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PDFS = ROOT / "data" / "raw" / "injury_reports"
OUT = ROOT / "runs" / "edge_scan" / "injury_rows.parquet"
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
NAME_RE = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf")


def report_time(path: Path) -> pd.Timestamp | None:
    m = NAME_RE.match(path.name)
    if not m:
        return None
    day, hh, mm, ap = m.groups()
    h = int(hh) % 12 + (12 if ap == "PM" else 0)
    return pd.Timestamp(f"{day} {h:02d}:{mm}", tz="America/New_York").tz_convert("UTC")


def parse(path: Path) -> list[dict]:
    import pypdf
    ts = report_time(path)
    rows, matchup, team = [], None, None
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception:
        return []
    for page in reader.pages:
        toks = []

        def visit(text, cm, tm, fd, fs):
            if text.strip():
                toks.append((tm[4] * cm[0] + cm[4], tm[5] * cm[3] + cm[5], text.strip()))
        try:
            page.extract_text(visitor_text=visit)
        except Exception:
            continue
        status_rows = sorted({round(y, 1) for x, y, t in toks if 575 <= x < 660 and t in STATUSES}, reverse=True)
        anchors = sorted({round(y, 1) for x, y, t in toks if 195 <= x < 420 and 60 < y < 470}, reverse=True)
        for y in sorted(set(status_rows) | set(anchors), reverse=True):
            near = sorted([(x, t) for x, yy, t in toks if abs(yy - y) < 2.5 and 60 < yy < 470])
            mt = " ".join(t for x, t in near if 195 <= x < 260)
            tm = " ".join(t for x, t in near if 260 <= x < 420)
            if mt and "@" in mt:
                matchup = mt.split()[0]
            if tm:
                team = tm
            if y not in status_rows:
                continue
            player = " ".join(t for x, t in near if 420 <= x < 575)
            status = next((t for x, t in near if 575 <= x < 660 and t in STATUSES), None)
            if player and status:
                rows.append({"report_time": ts, "file": path.name, "matchup": matchup, "team": team,
                             "player": player.replace(" ,", ","), "status": status})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-04-13")
    ap.add_argument("--shard", default=None, help="i/n: parse every n-th file starting at i (run n processes)")
    ap.add_argument("--merge", type=int, default=0, help="concatenate n shard files into the output")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if a.merge:
        df = pd.concat([pd.read_parquet(OUT.with_suffix(f".{i}.parquet")) for i in range(a.merge)], ignore_index=True)
        df.to_parquet(OUT, index=False)
        print(f"{df.file.nunique()} reports -> {len(df)} rows, {df.player.nunique()} players -> {OUT}")
        return
    files = sorted(p for p in PDFS.glob("Injury-Report_*.pdf") if a.start <= p.name[14:24] < a.end)
    i, n = map(int, (a.shard or "0/1").split("/"))
    df = pd.DataFrame([r for f in files[i::n] for r in parse(f)])
    df.to_parquet(OUT.with_suffix(f".{i}.parquet") if a.shard else OUT, index=False)
    print(f"shard {i}/{n}: {len(files[i::n])} reports -> {len(df)} rows")


if __name__ == "__main__":
    main()
