"""Adapt normalized live Polymarket moneylines to the existing agent table boundary.

This module is intentionally separate from :mod:`data_sources.polymarket`:
the public API client stays independent of replay and agent code.  The adapter
only prepares read-only ``games``, ``markets`` and ``prices`` frames.  It does
not create orders, estimate venue fees, or interpret non-moneyline markets as
game-winner contracts.
"""
from dataclasses import dataclass
from functools import lru_cache
import re

import pandas as pd

from data_sources import SCHEMAS, name_key
from data_sources.polymarket import DataQualityStatus, LiveNBASnapshot


_EVENT_SLUG = re.compile(
    r"^nba-(?P<away>[a-z]{3})-(?P<home>[a-z]{3})-(?P<date>\d{4}-\d{2}-\d{2})(?:-|$)",
    re.IGNORECASE,
)

GAME_COLUMNS = [*SCHEMAS["games"], "source", "event_id", "event_slug", "event_title"]
MARKET_COLUMNS = [
    *SCHEMAS["markets"], "event_id", "event_slug", "market_id", "token_id", "polymarket_url",
]
PRICE_COLUMNS = [
    *SCHEMAS["prices"], "probability", "midpoint", "last_trade_price", "spread", "liquidity",
    "total_volume", "volume_24h",
]


@dataclass(frozen=True)
class PolymarketAgentTables:
    """Legacy-shaped live tables plus an explicit suitability result.

    ``volume`` is deliberately missing/NaN because Polymarket's market-level
    cumulative volume is not the per-interval volume expected by replay sizing.
    The original values remain available in the extra columns.
    """

    games: pd.DataFrame
    markets: pd.DataFrame
    prices: pd.DataFrame
    quality: DataQualityStatus
    reasons: tuple[str, ...]
    skipped_markets: int = 0

    @property
    def usable_for_live_analysis(self) -> bool:
        return self.quality.usable_for_future_analysis and not self.markets.empty and not self.prices.empty

    @property
    def game_count(self) -> int:
        return int(self.games.game_id.nunique()) if not self.games.empty else 0

    @property
    def quote_count(self) -> int:
        return len(self.prices)


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _result(snapshot: LiveNBASnapshot, reasons=(), skipped=0, games=None, markets=None, prices=None):
    return PolymarketAgentTables(
        games if games is not None else _empty(GAME_COLUMNS),
        markets if markets is not None else _empty(MARKET_COLUMNS),
        prices if prices is not None else _empty(PRICE_COLUMNS),
        snapshot.quality,
        tuple(dict.fromkeys(str(reason) for reason in reasons if reason)),
        skipped,
    )


@lru_cache(maxsize=1)
def _team_directory() -> dict[str, dict]:
    """NBA team names/codes/IDs from nba_api's bundled static directory."""
    from nba_api.stats.static import teams

    directory: dict[str, dict] = {}
    for team in teams.get_teams():
        identity = {"code": str(team["abbreviation"]).upper(), "team_id": int(team["id"])}
        aliases = {
            team.get("abbreviation"), team.get("nickname"), team.get("full_name"),
            " ".join(part for part in (team.get("city"), team.get("nickname")) if part),
        }
        for alias in aliases:
            if alias:
                directory[name_key(str(alias))] = identity
    # Common feed wording that is not consistent across providers.
    for alias, code in {"Sixers": "PHI", "LA Clippers": "LAC"}.items():
        match = next((value for value in directory.values() if value["code"] == code), None)
        if match:
            directory[name_key(alias)] = match
    return directory


def _event_teams(slug: str | None) -> tuple[str, str, str] | None:
    match = _EVENT_SLUG.match(str(slug or ""))
    if not match:
        return None
    return match.group("away").upper(), match.group("home").upper(), match.group("date")


def _outcome_team(outcome: str | None) -> dict | None:
    return _team_directory().get(name_key(str(outcome or "")))


def adapt_live_nba_snapshot(snapshot: LiveNBASnapshot) -> PolymarketAgentTables:
    """Convert safe, two-team moneylines into the project's table interface.

    Unsupported props/totals remain available in the live dashboard but are
    not sent to a game-winner model.  A moneyline is accepted only when both
    outcomes map exactly to the two teams encoded in the NBA event slug and
    every outcome has a CLOB token plus bid and ask.
    """
    if not snapshot.quality.usable_for_future_analysis:
        return _result(snapshot, ("Source quality gate did not allow live analysis",))

    game_rows: dict[str, dict] = {}
    market_rows: list[dict] = []
    price_rows: list[dict] = []
    reasons: list[str] = []
    skipped = 0

    for market in snapshot.markets:
        if str(market.market_type or "").lower() != "moneyline":
            skipped += 1
            continue
        event_teams = _event_teams(market.event_slug)
        if event_teams is None:
            skipped += 1
            reasons.append(f"Moneyline {market.market_id or '<unknown>'} has no strict NBA game slug")
            continue
        away, home, game_date = event_teams
        if market.market_start_time is None:
            skipped += 1
            reasons.append(f"Moneyline {market.market_id or '<unknown>'} has no start time")
            continue
        if not market.active or market.closed or market.accepting_orders is False:
            skipped += 1
            reasons.append(f"Moneyline {market.market_id or '<unknown>'} is not open")
            continue

        tip = pd.Timestamp(market.market_start_time)
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
        fetched_at = pd.Timestamp(market.fetched_at)
        fetched_at = (fetched_at.tz_localize("UTC") if fetched_at.tzinfo is None
                      else fetched_at.tz_convert("UTC"))
        if tip <= fetched_at:
            skipped += 1
            reasons.append(f"Moneyline {market.market_id or '<unknown>'} has already started")
            continue

        outcomes = []
        for outcome in market.outcomes:
            identity = _outcome_team(outcome.outcome)
            if (identity is None or identity["code"] not in {away, home} or not outcome.token_id
                    or outcome.best_bid is None or outcome.best_ask is None):
                outcomes = []
                break
            outcomes.append((outcome, identity))
        if len(outcomes) != 2 or {identity["code"] for _, identity in outcomes} != {away, home}:
            skipped += 1
            reasons.append(f"Moneyline {market.market_id or '<unknown>'} could not map both team outcomes and quotes")
            continue

        identities = {identity["code"]: identity for _, identity in outcomes}
        event_key = market.event_id or market.event_slug
        game_id = f"polymarket:{event_key}"
        game_rows[game_id] = {
            "game_id": game_id,
            "date": game_date,
            "tip_time": tip,
            "final_at": tip + pd.Timedelta(hours=3),
            "home_team_id": identities[home]["team_id"],
            "away_team_id": identities[away]["team_id"],
            "home_team": home,
            "away_team": away,
            "home_pts": float("nan"),
            "away_pts": float("nan"),
            "source": "polymarket",
            "event_id": market.event_id,
            "event_slug": market.event_slug,
            "event_title": market.event_title,
        }
        for outcome, identity in outcomes:
            ticker = f"PM-{market.market_id}-{identity['code']}"
            market_rows.append({
                "venue": "polymarket",
                "market_ticker": ticker,
                "kind": "game",
                "game_id": game_id,
                "team": identity["code"],
                "player_id": None,
                "line": None,
                "title": market.question or market.event_title or "",
                "event_id": market.event_id,
                "event_slug": market.event_slug,
                "market_id": market.market_id,
                "token_id": outcome.token_id,
                "polymarket_url": market.polymarket_url,
            })
            price_rows.append({
                "venue": "polymarket",
                "market_ticker": ticker,
                "ts": fetched_at,
                "bid": float(outcome.best_bid),
                "ask": float(outcome.best_ask),
                # Replay expects interval volume; do not substitute cumulative volume.
                "volume": float("nan"),
                "probability": outcome.price,
                "midpoint": outcome.midpoint,
                "last_trade_price": outcome.last_trade_price,
                "spread": outcome.spread,
                "liquidity": market.liquidity,
                "total_volume": market.total_volume,
                "volume_24h": market.volume_24h,
            })

    games = pd.DataFrame(game_rows.values(), columns=GAME_COLUMNS)
    markets = pd.DataFrame(market_rows, columns=MARKET_COLUMNS)
    prices = pd.DataFrame(price_rows, columns=PRICE_COLUMNS)
    if markets.empty:
        reasons.append("No supported two-team NBA moneyline with complete CLOB quotes was found")
    return _result(snapshot, reasons, skipped, games, markets, prices)
