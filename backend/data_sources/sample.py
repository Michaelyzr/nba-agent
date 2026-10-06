"""D6: copy a few replay nights into data/sample/ so the demo runs from a fresh clone.

    python -m data_sources.sample                          # the 3 test-period nights with the most news
    python -m data_sources.sample --nights 2026-02-10 2026-03-05

Keeps every game, box score, player and news row (the models need the history)
and only the chosen nights' markets, prices and settlements.
"""
import argparse

import pandas as pd

from data_sources import FROZEN, read_table, write_table
from nba_agent_paths import SAMPLE



def busiest_nights(games, news, markets, start="2026-02-01", end="2026-04-12", n=3) -> list:
    g = games[(games.date >= start) & (games.date <= end) & games.game_id.isin(markets.game_id)]
    per_night = news[news.game_id.isin(g.game_id)].merge(g[["game_id", "date"]], on="game_id").groupby("date").size()
    return sorted(per_night.nlargest(n).index)


def build(nights=None):
    t = {n: read_table(n, FROZEN) for n in ("games", "player_games", "players", "news", "markets", "prices",
                                             "settlements")}
    nights = nights or busiest_nights(t["games"], t["news"], t["markets"])
    ids = t["games"].loc[t["games"].date.isin(nights), "game_id"]
    markets = t["markets"][t["markets"].game_id.isin(ids)]
    for name in ("games", "player_games", "players", "news"):
        write_table(name, t[name], SAMPLE)
    write_table("markets", markets, SAMPLE)
    write_table("prices", t["prices"][t["prices"].market_ticker.isin(markets.market_ticker)], SAMPLE)
    write_table("settlements", t["settlements"][t["settlements"].market_ticker.isin(markets.market_ticker)], SAMPLE)
    size = sum(p.stat().st_size for p in SAMPLE.glob("*.parquet")) / 1e6
    print(f"sample nights {nights}: {len(ids)} games, {len(markets)} markets, {size:.1f} MB in {SAMPLE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", nargs="*", default=None)
    build(ap.parse_args().nights)


if __name__ == "__main__":
    main()
