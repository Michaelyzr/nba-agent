"""Part 3c: probe public (no-auth) Kalshi and Polymarket APIs for NBA player-prop history.

    python -m evaluation.edge_scan_props_probe   -> runs/edge_scan/props_probe.json
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "edge_scan"
OUT = RUNS_DIR / "props_probe.json"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
START, END = int(time.mktime((2025, 12, 1, 0, 0, 0, 0, 0, 0))), int(time.mktime((2026, 4, 13, 0, 0, 0, 0, 0, 0)))
SERIES = ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBAPRA")


def get(url, **params):
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2)
                continue
            return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300])
        except Exception as e:      # network hiccup
            err = str(e)
            time.sleep(1)
    return None, err


def kalshi_markets(series, path="markets", max_pages=40):
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series, "limit": 1000, "min_close_ts": START, "max_close_ts": END}
        if path == "markets":
            params["status"] = "settled"
        if cursor:
            params["cursor"] = cursor
        code, d = get(f"{KALSHI}/{path}", **params)
        if code != 200 or not isinstance(d, dict):
            return out, code, d
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            break
    return out, 200, None


def kalshi_part() -> dict:
    res = {}
    for s in SERIES:
        rows = {}
        for path in ("markets", "historical/markets"):
            ms, code, err = kalshi_markets(s, path)
            rows[path] = {"code": code, "n": len(ms), "err": str(err)[:200] if err else None}
            if ms:
                events = Counter(m["event_ticker"] for m in ms)
                vol = [float(m.get("volume_fp") or m.get("volume") or 0) for m in ms]
                rows[path].update({
                    "events": len(events), "markets_per_event_median": sorted(events.values())[len(events) // 2],
                    "first_close": min(m["close_time"] for m in ms), "last_close": max(m["close_time"] for m in ms),
                    "median_volume": sorted(vol)[len(vol) // 2], "share_volume_gt_1000": sum(v > 1000 for v in vol) / len(vol),
                    "example": {k: ms[0].get(k) for k in ("ticker", "title", "yes_sub_title", "close_time", "volume", "volume_fp", "result")}})
                rows[path]["_sample"] = [m["ticker"] for m in ms[:: max(1, len(ms) // 5)]][:5]
        res[s] = rows
    # one candlestick history
    tick = next((r[p]["_sample"][0] for r in res.values() for p in r if r[p].get("_sample")), None)
    if tick:
        series = tick.split("-")[0]
        for path in (f"series/{series}/markets/{tick}/candlesticks", f"historical/markets/{tick}/candlesticks"):
            code, d = get(f"{KALSHI}/{path}", start_ts=START, end_ts=END, period_interval=60)
            c = d.get("candlesticks", []) if isinstance(d, dict) else []
            res.setdefault("candles", {})[path] = {"code": code, "n": len(c), "first": c[0] if c else None,
                                                    "err": None if code == 200 else str(d)[:200]}
        code, d = get(f"{KALSHI}/markets/trades", ticker=tick, limit=1000)
        res["trades"] = {"ticker": tick, "code": code,
                         "n": len(d.get("trades", [])) if isinstance(d, dict) else 0}
    return res


def polymarket_part() -> dict:
    res = {}
    code, tags = get(f"{GAMMA}/sports")
    res["sports_code"] = code
    nba = [t for t in tags if isinstance(t, dict) and t.get("sport") == "nba"] if isinstance(tags, list) else []
    res["nba_sport"] = nba[:1]
    found, offset = [], 0
    for _ in range(20):
        code, ev = get(f"{GAMMA}/events", tag_slug="nba", closed="true", limit=500, offset=offset,
                       start_date_min="2025-12-01", end_date_max="2026-04-13")
        if code != 200 or not ev:
            res["events_err"] = None if code == 200 else str(ev)[:200]
            break
        found += ev
        offset += 500
    res["nba_closed_events"] = len(found)
    props = [m for e in found for m in e.get("markets", []) if any(
        k in (m.get("question") or "").lower() for k in ("points", "rebounds", "assists", "pts")) and
        " vs" not in (m.get("question") or "").lower()[:0] and "o/u" in (m.get("question") or "").lower()]
    player_like = [m for m in props if not any(t in (m.get("question") or "") for t in ("Total", "total"))]
    res["prop_like_markets"] = len(player_like)
    res["prop_examples"] = [m.get("question") for m in player_like[:8]]
    res["sports_market_types"] = Counter(m.get("sportsMarketType") for e in found for m in e.get("markets", [])).most_common(15)
    if player_like:
        m = player_like[0]
        tok = json.loads(m.get("clobTokenIds") or "[]")
        if tok:
            code, d = get(f"{CLOB}/prices-history", market=tok[0], interval="max", fidelity=60)
            h = d.get("history", []) if isinstance(d, dict) else []
            res["price_history"] = {"question": m.get("question"), "code": code, "n": len(h), "volume": m.get("volume")}
    return res


def history_check(n_kalshi=12) -> dict:
    """Price history for in-window prop markets: Kalshi hourly candles, Polymarket clob prices-history."""
    res = {"kalshi": [], "polymarket": []}
    ms, _, _ = kalshi_markets("KXNBAPTS", "historical/markets")
    ms = [m for m in ms if "2025-12-01" <= m["close_time"][:10] <= "2026-04-13"]
    res["kalshi_pts_markets_dec_apr"] = len(ms)
    res["kalshi_pts_events_dec_apr"] = len({m["event_ticker"] for m in ms})
    vol = sorted(float(m.get("volume_fp") or 0) for m in ms)
    res["kalshi_pts_volume_quartiles"] = [vol[len(vol) // 4], vol[len(vol) // 2], vol[3 * len(vol) // 4]] if vol else []
    for m in ms[:: max(1, len(ms) // n_kalshi)][:n_kalshi]:
        close = int(pd_ts(m["close_time"]))
        code, d = get(f"{KALSHI}/historical/markets/{m['ticker']}/candlesticks", start_ts=close - 2 * 86400,
                      end_ts=close, period_interval=1)
        c = d.get("candlesticks", []) if isinstance(d, dict) else []
        quoted = [x for x in c if (x.get("yes_bid") or {}).get("close_dollars") or (x.get("yes_bid") or {}).get("close")]
        res["kalshi"].append({"ticker": m["ticker"], "volume": m.get("volume_fp"), "code": code, "minute_candles": len(c),
                              "with_bid": len(quoted), "sample": c[len(c) // 2] if c else None})
    events, offset = [], 0
    for _ in range(40):
        code, ev = get(f"{GAMMA}/events", series_id=10345, closed="true", limit=100, offset=offset,
                       end_date_min="2025-12-01", end_date_max="2026-04-13")
        if code != 200 or not ev:
            break
        events += ev
        offset += 100
    props = [m for e in events for m in e.get("markets", []) if m.get("sportsMarketType") in ("points", "rebounds", "assists")]
    res["poly_events_dec_apr"] = len(events)
    res["poly_prop_markets_dec_apr"] = len(props)
    res["poly_points_markets_dec_apr"] = sum(m.get("sportsMarketType") == "points" for m in props)
    pv = sorted(float(m.get("volume") or 0) for m in props)
    res["poly_prop_volume_quartiles"] = [pv[len(pv) // 4], pv[len(pv) // 2], pv[3 * len(pv) // 4]] if pv else []
    for m in [p for p in props if p.get("sportsMarketType") == "points"][:: max(1, len(props) // 30)][:8]:
        tok = json.loads(m.get("clobTokenIds") or "[]")
        end = int(pd_ts(m.get("endDate") or m.get("closedTime") or "2026-04-13T00:00:00Z"))
        code, d = get(f"{CLOB}/prices-history", market=tok[0], startTs=end - 3 * 86400, endTs=end + 86400, fidelity=1)
        h = d.get("history", []) if isinstance(d, dict) else []
        res["polymarket"].append({"question": m.get("question"), "end": m.get("endDate"), "volume": m.get("volume"),
                                  "code": code, "points": len(h)})
    return res


def pd_ts(s: str) -> float:
    import pandas as pd
    return pd.Timestamp(s).timestamp()


def main():
    import sys
    if "--history" in sys.argv:
        out = history_check()
        (RUNS_DIR / "props_history.json").write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps(out, indent=1, default=str)[:5000])
        return
    out = {"kalshi": kalshi_part(), "polymarket": polymarket_part()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
