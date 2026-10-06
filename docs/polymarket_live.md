# Polymarket NBA live data

`data_sources/polymarket.py` provides a public, read-only live NBA market client.
It does not need a wallet, API key, private key, or trading permission. The older
historical replay downloader in the same module remains available.

## Data flow

1. Gamma `GET /sports` identifies the current NBA series and primary tag.
2. Gamma `GET /events?series_id=...&active=true&closed=false` discovers events
   and their markets. If series discovery is unavailable or empty, the client
   uses exact NBA tag metadata. Its last fallback only accepts strict
   `nba-AAA-BBB-YYYY-MM-DD` slugs containing two valid NBA team codes.
3. CLOB `POST /prices`, `POST /midpoints`, and `POST /last-trades-prices` fetch
   batched, token-specific current pricing. Top-of-book failures do not crash a
   refresh; Gamma outcome prices remain visible and the quality result explains
   the degradation.

The client deliberately does not use `GET /markets?tag_slug=nba` for discovery:
that parameter has returned unrelated markets in observed API responses.

## Normalized interface

```python
from data_sources.polymarket import PolymarketClient

snapshot = PolymarketClient().fetch_live_nba()
if snapshot.quality.usable_for_future_analysis:
    for market in snapshot.markets:
        for outcome in market.outcomes:
            print(market.market_id, outcome.token_id, outcome.price)
```

`LiveNBASnapshot` contains normalized `MarketSnapshot` objects. Each market has
event/market identity, title and slug, start/end times, type, liquidity, volume,
flags, URL, and a generic tuple of outcomes. Each `OutcomeSnapshot` has its CLOB
token ID, current Gamma probability, best bid/ask, midpoint, last trade, and
spread. Downstream code never needs to parse raw Polymarket JSON and must not
assume outcomes are Yes/No.

`DataQualityStatus` reports `OK`, `WARNING`, or `UNAVAILABLE`, plus one of:

- `SOURCE DATA AVAILABLE`
- `SOURCE DATA STALE`
- `SOURCE DATA INVALID`
- `SOURCE DATA UNAVAILABLE`

Future model or recommendation code should run only when
`usable_for_future_analysis` is `True`.

## Run

Print a concise real-time snapshot (or a truthful empty/unavailable result):

```bash
python -m data_sources.polymarket
```

Useful options are `--limit 50`, `--json`, and `--save-history`. The historical
replay command still accepts `--start YYYY-MM-DD --end YYYY-MM-DD`.

Run the dashboard:

```bash
streamlit run app.py
```

Choose **NBA Polymarket Live Markets** in the sidebar. The page polls every
10–30 seconds, has a manual refresh button, shows UTC fetch time and quality,
supports event/outcome/type filters, and includes a developer table. API errors
produce a readable status and never cause sample/demo markets to appear.

## Snapshot history

When enabled in the UI (or with `--save-history`), the flattened one-row-per-
outcome adapter writes `data/live/polymarket_nba_snapshots.parquet`. Rows are
deduplicated by `fetched_at + market_id + token_id`, written atomically, and
limited to the latest 90 days. `data/live/` is generated data and is ignored by
Git. This schema is intentionally separate from the older replay
`markets.parquet`/`prices.parquet` pair, whose Yes-side format cannot represent
all live outcome types accurately.
