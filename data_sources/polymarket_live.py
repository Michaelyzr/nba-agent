"""Single real-time Polymarket entry point for UI and read-only agents.

Current quotes always come from the snapshot fetched during ``refresh``. Saved
parquet rows may extend a price series, but are never used as a fallback current
quote when the API or quality gate is unavailable.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_sources.polymarket import (LIVE_HISTORY, LiveNBASnapshot, PolymarketClient,
                                     write_snapshot_history)
from data_sources.polymarket_adapter import (PRICE_COLUMNS, PolymarketAgentTables,
                                             adapt_live_nba_snapshot)
from replay import AsOf


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class LivePolymarketView(AsOf):
    """Existing ``AsOf`` interface backed by the latest Polymarket refresh."""

    def __init__(self, tables: dict, now: pd.Timestamp, price_index: dict,
                 snapshot: LiveNBASnapshot, agent_tables: PolymarketAgentTables):
        super().__init__(tables, now, price_index)
        self.polymarket_snapshot = snapshot
        self.polymarket_quality = snapshot.quality
        self.polymarket_tables = agent_tables


@dataclass(frozen=True)
class PolymarketLiveContext:
    snapshot: LiveNBASnapshot
    agent_tables: PolymarketAgentTables
    history_path: Path
    history_written: bool = False
    history_error: str | None = None

    @property
    def now(self) -> pd.Timestamp:
        return _utc(self.snapshot.quality.fetched_at)

    @property
    def usable_for_live_analysis(self) -> bool:
        return self.agent_tables.usable_for_live_analysis

    def price_history(self) -> pd.DataFrame:
        """Real snapshots for current tokens, with this refresh always included."""
        current = self.agent_tables.prices.copy()
        if current.empty:
            # No cached quote is allowed to revive an unavailable refresh.
            return current
        frames = []
        if self.history_path.exists():
            try:
                saved = pd.read_parquet(self.history_path)
                mapping = self.agent_tables.markets[["market_ticker", "market_id", "token_id"]]
                saved = saved.merge(mapping, on=["market_id", "token_id"], how="inner")
                if not saved.empty:
                    history = pd.DataFrame({
                        "venue": "polymarket",
                        "market_ticker": saved.market_ticker,
                        "ts": pd.to_datetime(saved.fetched_at, utc=True),
                        "bid": saved.best_bid,
                        "ask": saved.best_ask,
                        "volume": float("nan"),
                        "probability": saved.price,
                        "midpoint": saved.midpoint,
                        "last_trade_price": saved.last_trade_price,
                        "spread": saved.spread,
                        "liquidity": saved.liquidity,
                        "total_volume": saved.total_volume,
                        "volume_24h": saved.volume_24h,
                    })
                    history = history[history.ts <= self.now].dropna(subset=["bid", "ask"])
                    frames.append(history)
            except (OSError, ValueError, KeyError, ImportError):
                # The current in-memory API snapshot remains authoritative.
                pass
        frames.append(current)
        prices = pd.concat(frames, ignore_index=True)
        prices["ts"] = pd.to_datetime(prices.ts, utc=True)
        return (prices.drop_duplicates(["market_ticker", "ts"], keep="last")
                .sort_values(["market_ticker", "ts"])
                .reset_index(drop=True)[PRICE_COLUMNS])

    def view(self, historical_tables: dict, news: pd.DataFrame | None = None) -> LivePolymarketView:
        """Build the view consumed by Forecaster/MarketAgent without old market quotes."""
        tables = dict(historical_tables)
        old_games = historical_tables.get("games", pd.DataFrame())
        tables["games"] = (pd.concat([old_games, self.agent_tables.games], ignore_index=True)
                           .drop_duplicates("game_id", keep="last"))
        tables["markets"] = self.agent_tables.markets.copy()
        tables["prices"] = self.price_history()
        tables["news"] = (news.copy() if news is not None else
                          historical_tables.get("news", pd.DataFrame(
                              columns=["news_id", "published_at", "game_id", "player_id", "status"])).copy())
        prices = tables["prices"].sort_values("ts")
        index = {ticker: rows.reset_index(drop=True)
                 for ticker, rows in prices.groupby("market_ticker")}
        return LivePolymarketView(tables, self.now, index, self.snapshot, self.agent_tables)

    def for_game(self, game: Any) -> dict:
        """Serializable current Polymarket block for a pregame-agent snapshot."""
        quality = self.snapshot.quality
        base = {
            "source": "polymarket",
            "is_live": True,
            "status": quality.status,
            "source_state": quality.source_state,
            "fetched_at": quality.fetched_at.isoformat(),
            "age_seconds": quality.age_seconds,
            "usable_for_future_analysis": False,
            "reasons": list(dict.fromkeys([*quality.reasons, *self.agent_tables.reasons])),
            "markets": [],
        }
        if not self.usable_for_live_analysis:
            return base

        games = self.agent_tables.games
        game_id = str(getattr(game, "game_id", ""))
        candidates = games[games.game_id.astype(str) == game_id]
        if candidates.empty:
            home, away = str(getattr(game, "home_team", "")), str(getattr(game, "away_team", ""))
            candidates = games[(games.home_team == home) & (games.away_team == away)]
            wanted_date = str(getattr(game, "date", ""))
            if wanted_date:
                candidates = candidates[candidates.date.astype(str) == wanted_date]
        if candidates.empty:
            base["reasons"].append("No live Polymarket moneyline matched this game")
            return base
        if len(candidates) > 1 and getattr(game, "tip_time", None) is not None:
            tip = _utc(game.tip_time)
            candidates = candidates.assign(distance=(pd.to_datetime(candidates.tip_time, utc=True) - tip).abs())
            candidates = candidates.nsmallest(1, "distance")
        selected = candidates.iloc[0]
        markets = self.agent_tables.markets[self.agent_tables.markets.game_id == selected.game_id]
        quotes = markets.merge(self.agent_tables.prices, on=["venue", "market_ticker"], how="left")
        fields = ["market_ticker", "market_id", "token_id", "team", "kind", "bid", "ask",
                  "probability", "midpoint", "last_trade_price", "spread", "liquidity",
                  "total_volume", "volume_24h", "ts", "polymarket_url"]
        base.update(
            usable_for_future_analysis=True,
            game_id=selected.game_id,
            event_id=selected.event_id,
            event_slug=selected.event_slug,
            event_title=selected.event_title,
            markets=quotes[fields].to_dict("records"),
        )
        return base


class PolymarketLiveProvider:
    """Fetch one authoritative live context; never falls back to cached prices."""

    def __init__(self, client: PolymarketClient | None = None, history_path: Path = LIVE_HISTORY):
        self.client = client or PolymarketClient()
        self.history_path = Path(history_path)

    def refresh(self, *, save_history: bool = False) -> PolymarketLiveContext:
        snapshot = self.client.fetch_live_nba()
        tables = adapt_live_nba_snapshot(snapshot)
        written, error = False, None
        if save_history and snapshot.markets:
            try:
                written = write_snapshot_history(snapshot, self.history_path) is not None
            except (OSError, ValueError, ImportError) as exc:
                error = str(exc)
        return PolymarketLiveContext(snapshot, tables, self.history_path, written, error)

    def for_game(self, game: Any, _now=None, *, save_history: bool = True) -> dict:
        return self.refresh(save_history=save_history).for_game(game)
