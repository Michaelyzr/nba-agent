"""Replay a published LLM run from runs/llm_cache/ with the network refused: every HTTP request is counted.

    python -m evaluation.llm_replay_check --run test-reasoner-full/tool_sceptic
    NBA_MODEL_DIR=runs/walkforward/2025-12-01 python -m evaluation.llm_replay_check --run prev-tool_sceptic/tool_sceptic

Passes when the replay makes 0 HTTP requests and reproduces the published fills and decision count.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from evaluation import llm_agent_eval as ev


def _refusing_backend(kind, model, min_interval):
    import httpx

    from agents.llm_client import DeepSeekBackend
    if kind != "deepseek":
        raise ValueError("only DeepSeek runs are checked")

    def handler(request):
        _refusing_backend.requests += 1
        raise httpx.ConnectError("network refused by llm_replay_check")

    client = httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler))
    return DeepSeekBackend(model, client=client, max_retries=0, min_interval=min_interval)


_refusing_backend.requests = 0


def _chunk(job: dict) -> dict:
    ev.make_backend = _refusing_backend
    out = ev._run_chunk(job)
    calls = [x for t in out["traces"] for x in [t.get("llm", {})]]
    return {"fills": out["fills"], "points": len(out["traces"]), "http": _refusing_backend.requests,
            "misses": sum(x.get("calls", 0) - x.get("cached", 0) for x in calls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="published run folder under runs/llm_agent/ (tag/setup)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    folder = ev.OUT / args.run
    meta = json.loads((folder / "meta.json").read_text())
    tables = ev.load_frozen()
    jobs = [{"setup": meta["setup"], "start": a, "end": b, "chunk": a, "backend": meta["backend"],
             "model": meta["model"], "frac": meta["subsample"], "seed": meta["seed"], "min_interval": 0.0,
             "min_edge": meta["min_edge"]} for a, b in ev.chunks(tables, meta["start"], meta["end"])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(_chunk, jobs))
    fills = pd.concat([p["fills"] for p in parts if len(p["fills"])], ignore_index=True)
    published = pd.read_parquet(folder / "fills.parquet")
    cols = ["game_id", "market_ticker", "side", "price", "contracts"]
    cols = [c for c in cols if c in fills.columns and c in published.columns]
    same = fills[cols].sort_values(cols).reset_index(drop=True).equals(
        published[cols].sort_values(cols).reset_index(drop=True))
    published_points = sum(1 for line in (folder / "trace.jsonl").read_text().splitlines() if line.strip())
    res = {"run": args.run, "http_requests": sum(p["http"] for p in parts),
           "cache_misses": sum(p["misses"] for p in parts), "decision_points": sum(p["points"] for p in parts),
           "published_points": published_points, "fills": len(fills), "published_fills": len(published),
           "fills_identical": bool(same), "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(res))
    ok = res["http_requests"] == 0 and same and res["decision_points"] == published_points
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
