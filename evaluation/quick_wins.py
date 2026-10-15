"""Offline analyses for the quick-wins branch (no API calls).

    python -m evaluation.quick_wins banned       # texts the widened BANNED regex newly matches
    python -m evaluation.quick_wins prefilter    # LLM calls a deterministic pre-filter would have saved

Outputs go to runs/quick_wins/ (JSON) and are summarised in evaluation/results/quick_wins.md.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import BANNED
from replay import RUNS

OUT = RUNS / "quick_wins"
OLD_BANNED = re.compile(r"\block\b|guarantee|can'?t lose|sure thing|risk[- ]free", re.I)
PUBLISHED_LLM = {"test": "llm_agent/test-reasoner-full/tool_sceptic", "dec_jan": "llm_agent/prev-tool_sceptic/tool_sceptic"}


def newly_banned(text: str) -> str | None:
    m = BANNED.search(text or "")
    return m.group(0) if m and not OLD_BANNED.search(text or "") else None


def scan_banned() -> dict:
    """Proposal rationales and sceptic reasons in every saved trace, plus every cached LLM reply."""
    hits = []
    for path in sorted(RUNS.glob("**/trace.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            texts = [("rationale", (t.get("proposal") or {}).get("rationale")), ("sceptic", (t.get("sceptic") or {}).get("reason"))]
            for kind, text in texts:
                word = newly_banned(str(text or ""))
                if word:
                    hits.append({"run": str(path.parent.relative_to(RUNS)), "kind": kind, "word": word,
                                 "game_id": t.get("game_id"), "as_of": t.get("as_of"), "status": t.get("status"),
                                 "decision": (t.get("proposal") or {}).get("decision"),
                                 "verdict": (t.get("sceptic") or {}).get("verdict"), "text": str(text)[:300]})
    cache_hits, cache_n = [], 0
    for path in (RUNS / "llm_cache").glob("*/*.json"):
        cache_n += 1
        try:
            entry = json.loads(path.read_text())
            reply = json.loads(entry.get("text") or "{}")
        except (ValueError, OSError):
            continue
        if not isinstance(reply, dict):
            continue
        for field in ("rationale", "reason"):
            word = newly_banned(str(reply.get(field) or "")[:300] if field == "reason" else str(reply.get(field) or ""))
            if word:
                cache_hits.append({"role": entry.get("role"), "field": field, "word": word,
                                   "decision": reply.get("decision"), "verdict": reply.get("verdict"),
                                   "text": str(reply.get(field))[:300]})
    # A rationale matters only on a buy (validation), a sceptic reason only on an approval (fail closed).
    changed = [h for h in hits if (h["kind"] == "rationale" and str(h["decision"]).lower() == "buy")
               or (h["kind"] == "sceptic" and h["verdict"] == "approve")]
    published = [h for h in changed if h["run"] in PUBLISHED_LLM.values()]
    res = {"trace_hits": hits, "trace_hits_n": len(hits), "decision_changing": changed,
           "decision_changing_published": published, "cache_entries": cache_n, "cache_hits": cache_hits,
           "cache_hits_n": len(cache_hits),
           "cache_hits_decision_changing": sum((h["field"] == "rationale" and str(h["decision"]).lower() == "buy")
                                               or (h["field"] == "reason" and h["verdict"] == "approve")
                                               for h in cache_hits)}
    return res


PREFILTER_LEVELS = (0.01, 0.02, 0.03)
MODEL_DIRS = {"test": None, "dec_jan": RUNS / "walkforward" / "2025-12-01"}   # dec_jan: walk-forward M4 refit


def _anchor_edges(job: dict) -> list:
    """Deterministic anchor + M4 best-side gap (empty baseline, as published) at each decision point."""
    from agents.tool_agent import P_MAX, P_MIN
    from agents.tools import AsOfTools
    from evaluation.llm_agent_eval import load_frozen
    from forecast.api import MODELS, Forecaster
    from replay import AsOf, Replay
    tables = load_frozen()
    forecaster = Forecaster.load(Path(job["model_dir"]) if job["model_dir"] else MODELS)
    index = Replay(tables, lambda *a: []).price_index
    games = tables["games"].assign(game_id=lambda d: d.game_id.astype(str)).set_index("game_id")
    out = []
    for gid, as_of in job["points"]:
        now = pd.Timestamp(as_of)
        game = next(games.loc[[gid]].reset_index().itertuples(index=False))
        tools = AsOfTools(AsOf(tables, now, index), game, now, forecaster, min_edge=0.04)
        best = tools.anchor_edge(P_MIN, P_MAX) if tools.fresh_teams() else None
        out.append({"game_id": gid, "as_of": as_of, "gap": None if best is None else best["gap"],
                    "team": None if best is None else best["team"]})
    return out


def prefilter_frame(window: str, workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    folder = RUNS / PUBLISHED_LLM[window]
    traces = [json.loads(x) for x in (folder / "trace.jsonl").read_text().splitlines() if x.strip()]
    cache = OUT / f"prefilter_{window}.parquet"
    if cache.exists():
        edges = pd.read_parquet(cache)
    else:
        points = [(str(t["game_id"]), t["as_of"]) for t in traces]
        size = -(-len(points) // workers)
        jobs = [{"points": points[i:i + size], "model_dir": str(MODEL_DIRS[window] or "")}
                for i in range(0, len(points), size)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            edges = pd.DataFrame([r for part in pool.map(_anchor_edges, jobs) for r in part])
        edges.to_parquet(cache, index=False)
    fills = pd.read_parquet(folder / "fills.parquet")
    rows = []
    for t, e in zip(traces, edges.itertuples()):
        assert str(t["game_id"]) == e.game_id and t["as_of"] == e.as_of
        llm = t.get("llm", {})
        rows.append({"window": window, "game_id": e.game_id, "as_of": e.as_of, "status": t["status"],
                     "gap": e.gap, "llm_calls": llm.get("calls", 0), "cost": llm.get("cost", 0.0),
                     "tokens_in": llm.get("tokens_in", 0), "tokens_out": llm.get("tokens_out", 0),
                     "reviewed": "sceptic" in t,
                     "approved": "sceptic" in t and t["sceptic"]["verdict"] == "approve"})
    return pd.DataFrame(rows).merge(
        fills.assign(game_id=fills.game_id.astype(str))[["game_id", "clv", "pnl", "contracts"]], on="game_id",
        how="left")


def prefilter_report(workers: int = 8) -> dict:
    """Skips, LLM calls and $ saved, and proposals / approved trades lost, per threshold (published runs)."""
    frames = [prefilter_frame(w, workers) for w in PUBLISHED_LLM]
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(OUT / "prefilter_points.csv", index=False)
    asked = d[~d.status.isin(["held", "idle"])]
    res = {"points": {w: int((d.window == w).sum()) for w in PUBLISHED_LLM},
           "asked": {w: int((asked.window == w).sum()) for w in PUBLISHED_LLM},
           "calls": int(asked.llm_calls.sum()), "cost": float(asked.cost.sum()),
           "reviewed": int(asked.reviewed.sum()), "approved": int(asked.approved.sum()),
           "gap_missing": int(asked.gap.isna().sum()), "levels": {}}
    for level in PREFILTER_LEVELS:
        skip = asked[asked.gap.isna() | (asked.gap < level)]
        lost = skip[skip.approved]
        res["levels"][str(level)] = {
            "skipped": {w: int((skip.window == w).sum()) for w in PUBLISHED_LLM}, "skipped_total": len(skip),
            "skip_share": len(skip) / len(asked), "calls_saved": int(skip.llm_calls.sum()),
            "cost_saved": float(skip.cost.sum()), "reviewed_lost": int(skip.reviewed.sum()),
            "approved_lost": int(len(lost)), "lost_trades": lost[["window", "game_id", "as_of", "gap", "clv", "pnl"]]
            .to_dict("records")}
    approved = asked[asked.approved]
    res["approved_trades"] = approved[["window", "game_id", "as_of", "gap", "clv", "pnl"]].to_dict("records")
    (OUT / "prefilter.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["banned", "prefilter"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.what == "banned":
        res = scan_banned()
        (OUT / "banned_scan.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))
        for h in res["decision_changing"]:
            print(h)
        for h in res["trace_hits"][:20]:
            print(" ", h["run"], h["kind"], h["word"], h["status"], h["text"][:120])
    else:
        prefilter_report(args.workers)


if __name__ == "__main__":
    main()
