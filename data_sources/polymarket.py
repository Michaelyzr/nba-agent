"""D3 backup: Polymarket NBA moneyline markets, 1-minute prices and settlements.

    python -m data_sources.polymarket --start 2026-02-01 --end 2026-02-07

Public, no API key. Price history is a traded price per minute with no order
book and no volume, so bid and ask are the price minus and plus HALF_SPREAD and
volume is left empty (the replay then applies its fixed size cap). Say so in the
report if Polymarket prices are used.
Writes markets_polymarket, prices_polymarket and settlements_polymarket so a
Kalshi download is never overwritten; concatenate them in the replay if needed.
"""
import argparse
import json
from datetime import date, timedelta

import pandas as pd

from data_sources import FROZEN, RAW, get_json, read_table

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CACHE = RAW / "polymarket"
HALF_SPREAD = 0.01
WINDOW = pd.Timedelta(hours=30)


def nickname_codes() -> dict:
    from nba_api.stats.static import teams

    return {t["nickname"].lower(): t["abbreviation"] for t in teams.get_teams()}


def events(start: date, end: date) -> list[dict]:
    out, offset = [], 0
    while True:
        page = get_json(f"{GAMMA}/events", {
            "tag_slug": "nba", "closed": "true", "limit": 100, "offset": offset,
            "end_date_min": f"{start}T00:00:00Z", "end_date_max": f"{end + timedelta(days=1)}T12:00:00Z"})
        out += [e for e in page if e.get("slug", "").startswith("nba-")]
        if len(page) < 100:
            return out
        offset += 100


def parse_slug(slug: str) -> dict:
    """nba-was-bkn-2026-02-07 -> away WAS, home BKN, date 2026-02-07 (Eastern)."""
    _, away, home, *day = slug.split("-")
    return {"away_team": away.upper(), "home_team": home.upper(), "date": "-".join(day)}


def history(token: str, tip: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / f"{token[:40]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    page = get_json(f"{CLOB}/prices-history", {"market": token, "fidelity": 1,
                                                "startTs": int((tip - WINDOW).timestamp()),
                                                "endTs": int((tip + pd.Timedelta(hours=4)).timestamp())})
    frame = pd.DataFrame(page.get("history", []), columns=["t", "p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def build(start: date, end: date):
    games = read_table("games")
    codes = nickname_codes()
    markets, prices, settlements = [], [], []
    for e in events(start, end):
        info = parse_slug(e["slug"])
        for m in e.get("markets", []):
            if m.get("sportsMarketType") != "moneyline" or not m.get("gameStartTime"):
                continue
            tip = pd.Timestamp(m["gameStartTime"])
            outcomes, tokens = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])
            finals = json.loads(m.get("outcomePrices") or "[]")
            for i, (name, token) in enumerate(zip(outcomes, tokens)):
                team = codes.get(name.lower())
                if team is None:
                    continue
                ticker = f"PM-{e['slug']}-{team}"
                markets.append({"venue": "polymarket", "market_ticker": ticker, "kind": "game", **info,
                                "team": team, "player_id": None, "line": None, "title": m.get("question", "")})
                h = history(token, tip)
                prices.append(pd.DataFrame({
                    "venue": "polymarket", "market_ticker": ticker,
                    "ts": pd.to_datetime(h.t, unit="s", utc=True),
                    "bid": (h.p - HALF_SPREAD).clip(0.01, 0.99), "ask": (h.p + HALF_SPREAD).clip(0.01, 0.99),
                    "volume": float("nan")}))
                if len(finals) == len(outcomes) and finals[i] in ("0", "1"):
                    settlements.append({"market_ticker": ticker, "outcome": int(finals[i]),
                                        "settled_at": pd.Timestamp(m.get("closedTime") or tip + pd.Timedelta(hours=3))})
    frame = pd.DataFrame(markets).merge(games[["game_id", "date", "home_team", "away_team"]],
                                        on=["date", "home_team", "away_team"], how="left")
    print(f"{frame.game_id.isna().sum()} of {len(frame)} markets had no game match")
    frame = frame.dropna(subset=["game_id"]).drop(columns=["date", "home_team", "away_team"])
    FROZEN.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FROZEN / "markets_polymarket.parquet", index=False)
    pd.concat(prices, ignore_index=True).to_parquet(FROZEN / "prices_polymarket.parquet", index=False)
    pd.DataFrame(settlements).to_parquet(FROZEN / "settlements_polymarket.parquet", index=False)
    print(f"wrote {len(frame)} Polymarket markets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    args = ap.parse_args()
    build(args.start, args.end)


if __name__ == "__main__":
    main()
