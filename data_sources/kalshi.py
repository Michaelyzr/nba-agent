"""D3: Kalshi NBA game-winner and player-points markets, 1-minute prices and settlements.

    python -m data_sources.kalshi depth                                   # months covered per series
    python -m data_sources.kalshi download --start 2026-02-01 --end 2026-02-07
    python -m data_sources.kalshi download --start 2026-02-01 --end 2026-04-15 --series KXNBAGAME

Public market data needs no API key. Markets settled before Kalshi's historical
cutoff are served from /historical/*; newer ones from the live endpoints.
Needs data/frozen/games.parquet and players.parquet to attach game_id and player_id.
Writes markets, prices and settlements (venue = "kalshi").
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd

from data_sources import RAW, get_json, name_key, player_lookup, read_table, write_table

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"KXNBAGAME": "game", "KXNBAPTS": "pts"}
CACHE = RAW / "kalshi"
WINDOW = pd.Timedelta(hours=30)
CHUNK = pd.Timedelta(hours=30)          # one request returns up to 1,800 one-minute candles
# Kalshi team codes that differ from NBA tricodes; extend when `download` reports unmatched codes.
ALIASES: dict[str, str] = {}


def list_markets(series: str) -> list[dict]:
    """Every settled market in a series, historical tier first, then the live tier."""
    out, seen = [], set()
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cursor = ""
        while True:
            params = {"series_ticker": series, "limit": 1000, **extra, **({"cursor": cursor} if cursor else {})}
            page = get_json(BASE + path, params)
            for m in page.get("markets", []):
                if m["ticker"] not in seen:
                    seen.add(m["ticker"])
                    out.append(m)
            cursor = page.get("cursor")
            if not cursor or not page.get("markets"):
                break
    return out


def parse_ticker(ticker: str) -> dict:
    """KXNBAGAME-26FEB07WASBKN-BKN -> date 2026-02-07, away WAS, home BKN, side BKN."""
    series, event, side = ticker.split("-", 2)
    day = datetime.strptime(event[:7].title(), "%y%b%d").date()
    teams = event[7:]
    away, home = (ALIASES.get(t, t) for t in (teams[:3], teams[3:]))
    team = ALIASES.get(side[:3], side[:3])
    return {"series": series, "date": str(day), "away_team": away, "home_team": home, "team": team}


def depth(series_list=SERIES) -> pd.DataFrame:
    rows = []
    for s in series_list:
        markets = list_markets(s)
        months = pd.Series([m["close_time"][:7] for m in markets]).value_counts().sort_index()
        print(f"{s}: {len(markets)} markets, earliest close {min(m['close_time'] for m in markets)}")
        print("  " + ", ".join(f"{k}: {v}" for k, v in months.items()))
        rows += [{"series": s, "month": k, "markets": v} for k, v in months.items()]
    return pd.DataFrame(rows)


def _cutoff() -> pd.Timestamp:
    return pd.Timestamp(get_json(BASE + "/historical/cutoff")["market_settled_ts"])


def candles(market: dict, cutoff: pd.Timestamp) -> pd.DataFrame:
    """1-minute candles over the 30 hours before close, cached per ticker.

    A candle's ts is the end of its minute, so a quote is only visible once the minute is over.
    """
    path = CACHE / "candles" / f"{market['ticker']}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    close = pd.Timestamp(market["close_time"])
    start = max(pd.Timestamp(market["open_time"]), close - WINDOW)
    historical = close < cutoff
    url = (f"{BASE}/historical/markets/{market['ticker']}/candlesticks" if historical else
           f"{BASE}/series/{market['ticker'].split('-')[0]}/markets/{market['ticker']}/candlesticks")
    rows = []
    t = start
    while t < close:
        end = min(t + CHUNK, close)
        page = get_json(url, {"start_ts": int(t.timestamp()), "end_ts": int(end.timestamp()), "period_interval": 1})
        for c in page.get("candlesticks", []):
            bid, ask = c["yes_bid"]["close"], c["yes_ask"]["close"]
            if bid is None and ask is None:
                continue
            rows.append({"venue": "kalshi", "market_ticker": market["ticker"],
                         "ts": pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC"),
                         "bid": float(bid) if bid is not None else float("nan"),
                         "ask": float(ask) if ask is not None else float("nan"),
                         "volume": float(c.get("volume") or 0)})
        t = end
    frame = pd.DataFrame(rows, columns=["venue", "market_ticker", "ts", "bid", "ask", "volume"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def market_rows(markets: list[dict], games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    lookup = player_lookup(players)
    rows = []
    for m in markets:
        info = parse_ticker(m["ticker"])
        player_id, line = None, None
        if SERIES.get(info["series"]) == "pts":
            player_id = lookup.get(name_key(m.get("yes_sub_title", "").split(":")[0]))
            line = m.get("floor_strike")
        rows.append({"venue": "kalshi", "market_ticker": m["ticker"], "kind": SERIES[info["series"]],
                     "date": info["date"], "home_team": info["home_team"], "away_team": info["away_team"],
                     "team": info["team"], "player_id": player_id, "line": line, "title": m.get("title", "")})
    frame = pd.DataFrame(rows).merge(games[["game_id", "date", "home_team", "away_team"]],
                                     on=["date", "home_team", "away_team"], how="left")
    bad_game = frame[frame.game_id.isna()]
    if len(bad_game):
        codes = sorted(set(bad_game.home_team) | set(bad_game.away_team))
        print(f"  {len(bad_game)} markets with no game match; check ALIASES for codes {codes[:20]}")
    bad_player = frame[(frame.kind == "pts") & frame.player_id.isna()]
    if len(bad_player):
        print(f"  {len(bad_player)} prop markets with no player match, e.g. {bad_player.title.head(3).tolist()}")
    return frame.dropna(subset=["game_id"]).drop(columns=["date", "home_team", "away_team"])


def settlement_rows(markets: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"market_ticker": m["ticker"], "settled_at": pd.Timestamp(m["settlement_ts"]),
                          "outcome": 1 if m["result"] == "yes" else 0}
                         for m in markets if m.get("result") in ("yes", "no") and m.get("settlement_ts")])


def download(start: date, end: date, series_list=SERIES, workers=4):
    games, players = read_table("games"), read_table("players")
    cutoff = _cutoff()
    markets = []
    for s in series_list:
        found = [m for m in list_markets(s) if start <= date.fromisoformat(parse_ticker(m["ticker"])["date"]) <= end]
        print(f"{s}: {len(found)} markets between {start} and {end}")
        markets += found
    with ThreadPoolExecutor(workers) as pool:
        prices = list(pool.map(lambda m: candles(m, cutoff), markets))
    write_table("markets", market_rows(markets, games, players))
    write_table("prices", pd.concat(prices, ignore_index=True))
    write_table("settlements", settlement_rows(markets))
    print(f"wrote {len(markets)} markets and {sum(map(len, prices))} price rows")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("depth")
    d.add_argument("--series", nargs="+", default=list(SERIES))
    dl = sub.add_parser("download")
    dl.add_argument("--start", type=date.fromisoformat, required=True)
    dl.add_argument("--end", type=date.fromisoformat, required=True)
    dl.add_argument("--series", nargs="+", default=list(SERIES))
    dl.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "depth":
        depth(args.series)
    else:
        download(args.start, args.end, args.series, args.workers)


if __name__ == "__main__":
    main()
