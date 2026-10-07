"""Read-only Polymarket NBA data for live display and historical replay.

Live usage::

    python -m data_sources.polymarket

The live client discovers NBA metadata through Gamma, fetches per-outcome CLOB
quotes in batches, and returns normalized dataclasses. It never authenticates or
places orders. The older historical replay builder remains available::

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
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_sources import FROZEN, RAW, ROOT, get_json, read_table

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CACHE = RAW / "polymarket"
LIVE_HISTORY = ROOT / "data" / "live" / "polymarket_nba_snapshots.parquet"
HALF_SPREAD = 0.01
POLITE_DELAY = 0.25
WINDOW = pd.Timedelta(30, unit="h")
NBA_TEAM_CODES = {
    "atl", "bos", "bkn", "cha", "chi", "cle", "dal", "den", "det", "gsw",
    "hou", "ind", "lac", "lal", "mem", "mia", "mil", "min", "nop", "nyk",
    "okc", "orl", "phi", "phx", "por", "sac", "sas", "tor", "uta", "was",
}
STALE_AFTER_SECONDS = 90
_SAFE_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NBA_GAME_SLUG = re.compile(r"^nba-([a-z]{3})-([a-z]{3})-\d{4}-\d{2}-\d{2}(?:-|$)")


class PolymarketAPIError(RuntimeError):
    """A public Polymarket endpoint failed or returned an unexpected shape."""


@dataclass(frozen=True)
class OutcomeSnapshot:
    outcome: str | None
    token_id: str | None
    price: float | None
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    last_trade_price: float | None
    spread: float | None


@dataclass(frozen=True)
class MarketSnapshot:
    fetched_at: datetime
    source: str
    event_id: str | None
    event_slug: str | None
    event_title: str | None
    market_id: str | None
    market_slug: str | None
    question: str | None
    market_type: str | None
    market_start_time: datetime | None
    market_end_time: datetime | None
    outcomes: tuple[OutcomeSnapshot, ...]
    liquidity: float | None
    total_volume: float | None
    volume_24h: float | None
    active: bool
    closed: bool
    accepting_orders: bool | None
    polymarket_url: str | None


@dataclass(frozen=True)
class DataQualityStatus:
    status: str
    source_state: str
    fetched_at: datetime
    age_seconds: float
    reasons: tuple[str, ...]
    usable_for_future_analysis: bool


@dataclass(frozen=True)
class LiveNBASnapshot:
    markets: tuple[MarketSnapshot, ...]
    quality: DataQualityStatus
    message: str
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryResult:
    events: tuple[dict, ...]
    request_succeeded: bool
    warnings: tuple[str, ...] = ()


def _default_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        backoff_factor=0.35,
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "nba-agent/0.1"})
    return session


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return default if value is None else bool(value)


def _as_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _unique_messages(messages: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(message for message in messages if message))


def _looks_like_nba_game(event: dict) -> bool:
    match = _NBA_GAME_SLUG.match(str(event.get("slug") or "").lower())
    return bool(match and match.group(1) in NBA_TEAM_CODES and match.group(2) in NBA_TEAM_CODES)


def is_nba_event(event: dict, primary_tag_id: str | int | None = None) -> bool:
    """Accept exact NBA metadata, with a strict two-team slug as a last fallback."""
    tag_id = str(primary_tag_id) if primary_tag_id is not None else None
    for tag in event.get("tags") or []:
        if not isinstance(tag, dict):
            continue
        if str(tag.get("slug") or "").lower() == "nba":
            return True
        if tag_id and str(tag.get("id")) == tag_id:
            return True
    return _looks_like_nba_game(event)


def _market_url(event_slug: str | None) -> str | None:
    if not event_slug or not _SAFE_SLUG.fullmatch(event_slug):
        return None
    return f"https://polymarket.com/event/{quote(event_slug, safe='-')}"


def normalize_market(
    event: dict,
    market: dict,
    fetched_at: datetime,
    clob_quotes: dict[str, dict] | None = None,
) -> MarketSnapshot:
    """Convert one Gamma market plus optional CLOB quotes into stable data."""
    clob_quotes = clob_quotes or {}
    names = _as_list(market.get("outcomes"))
    token_ids = _as_list(market.get("clobTokenIds"))
    prices = _as_list(market.get("outcomePrices"))
    count = max(len(names), len(token_ids), len(prices))
    outcomes = []
    for index in range(count):
        name = str(names[index]) if index < len(names) and names[index] is not None else None
        token_id = str(token_ids[index]) if index < len(token_ids) and token_ids[index] not in (None, "") else None
        gamma_price = _as_float(prices[index]) if index < len(prices) else None
        live = clob_quotes.get(token_id, {}) if token_id else {}
        best_bid = _as_float(live.get("best_bid"))
        best_ask = _as_float(live.get("best_ask"))
        midpoint = _as_float(live.get("midpoint"))
        last_trade = _as_float(live.get("last_trade_price"))

        # Gamma exposes these fields only for the first outcome. They are an
        # explicit degraded fallback when CLOB pricing is unavailable.
        if index == 0:
            best_bid = best_bid if best_bid is not None else _as_float(market.get("bestBid"))
            best_ask = best_ask if best_ask is not None else _as_float(market.get("bestAsk"))
            last_trade = last_trade if last_trade is not None else _as_float(market.get("lastTradePrice"))
        if midpoint is None and best_bid is not None and best_ask is not None:
            midpoint = (best_bid + best_ask) / 2
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
        price = gamma_price if gamma_price is not None else midpoint
        outcomes.append(OutcomeSnapshot(name, token_id, price, best_bid, best_ask, midpoint, last_trade, spread))

    event_slug = str(event.get("slug")) if event.get("slug") else None
    start = market.get("gameStartTime") or market.get("startDate") or event.get("startDate")
    end = market.get("endDate") or event.get("endDate")
    accepting = market.get("acceptingOrders")
    return MarketSnapshot(
        fetched_at=fetched_at,
        source="polymarket",
        event_id=str(event.get("id")) if event.get("id") is not None else None,
        event_slug=event_slug,
        event_title=str(event.get("title")) if event.get("title") else None,
        market_id=str(market.get("id")) if market.get("id") is not None else None,
        market_slug=str(market.get("slug")) if market.get("slug") else None,
        question=str(market.get("question")) if market.get("question") else None,
        market_type=str(market.get("sportsMarketType") or market.get("marketType") or "other"),
        market_start_time=_as_utc(start),
        market_end_time=_as_utc(end),
        outcomes=tuple(outcomes),
        liquidity=_as_float(market.get("liquidityNum") or market.get("liquidity")),
        total_volume=_as_float(market.get("volumeNum") or market.get("volume")),
        volume_24h=_as_float(market.get("volume24hr") or market.get("volume24hrClob")),
        active=_as_bool(market.get("active"), _as_bool(event.get("active"))),
        closed=_as_bool(market.get("closed"), _as_bool(event.get("closed"))),
        accepting_orders=_as_bool(accepting) if accepting is not None else None,
        polymarket_url=_market_url(event_slug),
    )


def validate_data_quality(
    markets: Iterable[MarketSnapshot],
    fetched_at: datetime,
    request_succeeded: bool,
    errors: Iterable[str] = (),
    *,
    now: datetime | None = None,
    stale_after_seconds: int = STALE_AFTER_SECONDS,
) -> DataQualityStatus:
    markets = tuple(markets)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    age = max(0.0, (now.astimezone(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds())
    reasons = list(errors)
    fatal = False

    if not request_succeeded:
        reasons.insert(0, "Polymarket API request failed")
        return DataQualityStatus(
            "UNAVAILABLE", "SOURCE DATA UNAVAILABLE", fetched_at, age,
            _unique_messages(reasons), False,
        )
    if not markets:
        reasons.append("No active NBA markets found")
        return DataQualityStatus(
            "WARNING", "SOURCE DATA AVAILABLE", fetched_at, age,
            _unique_messages(reasons), False,
        )

    token_ids: list[str] = []
    any_top_of_book = False
    incomplete_top_of_book = 0
    for market in markets:
        if not market.event_id or not market.market_id:
            reasons.append("One or more markets are missing required event or market IDs")
            fatal = True
        if not market.outcomes:
            reasons.append(f"Market {market.market_id or '<unknown>'} has no outcomes")
            fatal = True
        for outcome in market.outcomes:
            if not outcome.outcome:
                reasons.append(f"Market {market.market_id or '<unknown>'} has an unnamed outcome")
                fatal = True
            if not outcome.token_id:
                reasons.append(f"Market {market.market_id or '<unknown>'} has a missing CLOB token ID")
                fatal = True
            else:
                token_ids.append(outcome.token_id)
            if outcome.price is None or not 0 <= outcome.price <= 1:
                reasons.append(f"Market {market.market_id or '<unknown>'} has an invalid outcome price")
                fatal = True
            for label, value in (("bid", outcome.best_bid), ("ask", outcome.best_ask),
                                 ("midpoint", outcome.midpoint), ("last trade", outcome.last_trade_price)):
                if value is not None and not 0 <= value <= 1:
                    reasons.append(f"Market {market.market_id or '<unknown>'} has an invalid {label}")
                    fatal = True
            if outcome.best_bid is not None or outcome.best_ask is not None:
                any_top_of_book = True
            if outcome.best_bid is None or outcome.best_ask is None:
                incomplete_top_of_book += 1
            if (outcome.best_bid is not None and outcome.best_ask is not None
                    and outcome.best_bid > outcome.best_ask):
                reasons.append(f"Market {market.market_id or '<unknown>'} has bid above ask")
                fatal = True

    if len(token_ids) != len(set(token_ids)):
        reasons.append("Duplicate CLOB outcome token IDs detected")
        fatal = True
    if not any_top_of_book:
        reasons.append("CLOB top-of-book pricing is unavailable")
    elif incomplete_top_of_book:
        reasons.append(f"CLOB top-of-book pricing is incomplete for {incomplete_top_of_book} outcomes")
    stale = age > stale_after_seconds
    if stale:
        reasons.append(f"Snapshot is stale ({age:.0f} seconds old)")
    reasons = list(_unique_messages(reasons))
    source_state = "SOURCE DATA INVALID" if fatal else "SOURCE DATA STALE" if stale else "SOURCE DATA AVAILABLE"
    return DataQualityStatus(
        "OK" if not reasons else "WARNING",
        source_state,
        fetched_at,
        age,
        tuple(reasons),
        not fatal and not stale,
    )


class PolymarketClient:
    """Public, read-only Gamma discovery and batched CLOB pricing client."""

    def __init__(
        self,
        session: Any | None = None,
        *,
        timeout: float = 10,
        gamma_base: str = GAMMA,
        clob_base: str = CLOB,
        now: Callable[[], datetime] | None = None,
        batch_size: int = 100,
    ):
        self.session = session or _default_session()
        self.timeout = timeout
        self.gamma_base = gamma_base.rstrip("/")
        self.clob_base = clob_base.rstrip("/")
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.batch_size = max(1, batch_size)

    def _get_gamma(self, path: str, params: dict | None = None) -> Any:
        try:
            response = self.session.get(self.gamma_base + path, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise PolymarketAPIError(f"Gamma {path} failed: {exc}") from exc

    def _post_clob(self, path: str, payload: list[dict]) -> Any:
        try:
            response = self.session.post(self.clob_base + path, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise PolymarketAPIError(f"CLOB {path} failed: {exc}") from exc

    def _event_pages(self, params: dict, max_pages: int = 10) -> list[dict]:
        rows: list[dict] = []
        limit = 100
        for page in range(max_pages):
            payload = self._get_gamma("/events", {**params, "limit": limit, "offset": page * limit})
            if not isinstance(payload, list):
                raise PolymarketAPIError("Gamma /events returned a non-list response")
            rows.extend(item for item in payload if isinstance(item, dict))
            if len(payload) < limit:
                break
        return rows

    def discover_nba_events(self) -> DiscoveryResult:
        warnings: list[str] = []
        sport: dict | None = None
        try:
            sports = self._get_gamma("/sports")
            if not isinstance(sports, list):
                raise PolymarketAPIError("Gamma /sports returned a non-list response")
            sport = next((row for row in sports if isinstance(row, dict)
                          and (str(row.get("sport") or "").lower() == "nba"
                               or str(row.get("name") or "").lower() == "nba")), None)
        except PolymarketAPIError as exc:
            warnings.append(str(exc))

        request_succeeded = False
        if sport and sport.get("series") is not None:
            try:
                events = self._event_pages({
                    "series_id": str(sport["series"]), "active": "true", "closed": "false",
                    "order": "endDate", "ascending": "true",
                })
                request_succeeded = True
                filtered = [event for event in events if is_nba_event(event, sport.get("primaryTagId"))]
                if filtered:
                    return DiscoveryResult(tuple(_deduplicate_events(filtered)), True, _unique_messages(warnings))
            except PolymarketAPIError as exc:
                warnings.append(str(exc))

        # Exact NBA tag metadata is authoritative and also covers NBA futures if
        # no game-series events are active.
        try:
            events = self._event_pages({
                "tag_slug": "nba", "active": "true", "closed": "false",
                "order": "endDate", "ascending": "true",
            })
            request_succeeded = True
            filtered = [event for event in events if is_nba_event(event, sport.get("primaryTagId") if sport else None)]
            if filtered:
                return DiscoveryResult(tuple(_deduplicate_events(filtered)), True, _unique_messages(warnings))
        except PolymarketAPIError as exc:
            warnings.append(str(exc))

        # Last resort: scan active events, but accept only strict nba-AAA-BBB-date
        # slugs with two current NBA team codes.
        try:
            events = self._event_pages({"active": "true", "closed": "false"})
            request_succeeded = True
            filtered = [event for event in events if _looks_like_nba_game(event)]
            return DiscoveryResult(tuple(_deduplicate_events(filtered)), True, _unique_messages(warnings))
        except PolymarketAPIError as exc:
            warnings.append(str(exc))
        return DiscoveryResult((), request_succeeded, _unique_messages(warnings))

    def clob_quotes(self, token_ids: Iterable[str]) -> tuple[dict[str, dict], tuple[str, ...]]:
        tokens = list(dict.fromkeys(str(token) for token in token_ids if token))
        quotes = {token: {} for token in tokens}
        errors: list[str] = []
        for start in range(0, len(tokens), self.batch_size):
            chunk = tokens[start:start + self.batch_size]
            token_payload = [{"token_id": token} for token in chunk]
            try:
                sides = [{"token_id": token, "side": side} for token in chunk for side in ("BUY", "SELL")]
                prices = self._post_clob("/prices", sides)
                if not isinstance(prices, dict):
                    raise PolymarketAPIError("CLOB /prices returned a non-object response")
                for token in chunk:
                    row = prices.get(token) or {}
                    quotes[token]["best_bid"] = _as_float(row.get("BUY"))
                    quotes[token]["best_ask"] = _as_float(row.get("SELL"))
            except PolymarketAPIError as exc:
                errors.append(str(exc))
            try:
                midpoints = self._post_clob("/midpoints", token_payload)
                if not isinstance(midpoints, dict):
                    raise PolymarketAPIError("CLOB /midpoints returned a non-object response")
                for token in chunk:
                    quotes[token]["midpoint"] = _as_float(midpoints.get(token))
            except PolymarketAPIError as exc:
                errors.append(str(exc))
            try:
                trades = self._post_clob("/last-trades-prices", token_payload)
                if not isinstance(trades, list):
                    raise PolymarketAPIError("CLOB /last-trades-prices returned a non-list response")
                for row in trades:
                    if isinstance(row, dict) and str(row.get("token_id")) in quotes:
                        quotes[str(row["token_id"])]["last_trade_price"] = _as_float(row.get("price"))
            except PolymarketAPIError as exc:
                errors.append(str(exc))
        return quotes, _unique_messages(errors)

    def fetch_live_nba(self) -> LiveNBASnapshot:
        fetched_at = self.now()
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        else:
            fetched_at = fetched_at.astimezone(timezone.utc)
        discovery = self.discover_nba_events()
        errors = list(discovery.warnings)
        if not discovery.events:
            quality = validate_data_quality(
                (), fetched_at, discovery.request_succeeded, errors, now=fetched_at,
            )
            message = ("No active NBA markets are currently available."
                       if discovery.request_succeeded else "Polymarket data temporarily unavailable.")
            return LiveNBASnapshot((), quality, message, _unique_messages(errors))

        market_pairs = []
        token_ids: list[str] = []
        for event in discovery.events:
            for market in event.get("markets") or []:
                if not isinstance(market, dict):
                    continue
                active = _as_bool(market.get("active"), _as_bool(event.get("active")))
                closed = _as_bool(market.get("closed"), _as_bool(event.get("closed")))
                if not active or closed:
                    continue
                market_pairs.append((event, market))
                token_ids.extend(str(token) for token in _as_list(market.get("clobTokenIds")) if token)

        if not market_pairs:
            quality = validate_data_quality(
                (), fetched_at, discovery.request_succeeded, errors, now=fetched_at,
            )
            return LiveNBASnapshot(
                (), quality, "No active NBA markets are currently available.", _unique_messages(errors),
            )

        quotes, clob_errors = self.clob_quotes(token_ids)
        errors.extend(clob_errors)
        markets = tuple(normalize_market(event, market, fetched_at, quotes) for event, market in market_pairs)
        quality = validate_data_quality(
            markets, fetched_at, discovery.request_succeeded, errors, now=fetched_at,
        )
        message = (f"Fetched {len(markets)} active NBA markets across "
                   f"{len({market.event_id for market in markets})} events.")
        return LiveNBASnapshot(markets, quality, message, _unique_messages(errors))


def _deduplicate_events(events: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for event in events:
        key = str(event.get("id") or event.get("slug") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def snapshot_frame(snapshot: LiveNBASnapshot) -> pd.DataFrame:
    """Explicit one-row-per-outcome adapter for parquet history and models."""
    rows = []
    for market in snapshot.markets:
        for outcome in market.outcomes:
            rows.append({
                "fetched_at": market.fetched_at,
                "source": market.source,
                "event_id": market.event_id,
                "event_slug": market.event_slug,
                "event_title": market.event_title,
                "market_id": market.market_id,
                "market_slug": market.market_slug,
                "question": market.question,
                "market_type": market.market_type,
                "market_start_time": market.market_start_time,
                "market_end_time": market.market_end_time,
                "outcome": outcome.outcome,
                "token_id": outcome.token_id,
                "price": outcome.price,
                "best_bid": outcome.best_bid,
                "best_ask": outcome.best_ask,
                "midpoint": outcome.midpoint,
                "last_trade_price": outcome.last_trade_price,
                "spread": outcome.spread,
                "liquidity": market.liquidity,
                "total_volume": market.total_volume,
                "volume_24h": market.volume_24h,
                "active": market.active,
                "closed": market.closed,
                "accepting_orders": market.accepting_orders,
                "polymarket_url": market.polymarket_url,
            })
    return pd.DataFrame(rows)


def write_snapshot_history(
    snapshot: LiveNBASnapshot,
    path: Path = LIVE_HISTORY,
    *,
    retention_days: int = 90,
) -> Path | None:
    """Append a real live snapshot atomically, deduplicated by its natural key."""
    new = snapshot_frame(snapshot)
    if new.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    frame = pd.concat([old, new], ignore_index=True)
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True)
    frame = frame.drop_duplicates(["fetched_at", "market_id", "token_id"], keep="last")
    if retention_days > 0:
        cutoff = frame["fetched_at"].max() - pd.Timedelta(days=retention_days)
        frame = frame[frame["fetched_at"] >= cutoff]
    frame = frame.sort_values(["fetched_at", "market_id", "token_id"], na_position="last")
    temp = path.with_suffix(".tmp.parquet")
    try:
        frame.to_parquet(temp, index=False)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return path


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


def events_by_slug(start: date, end: date) -> list[dict]:
    """One small Gamma lookup per known game (nba-<away>-<home>-<ET date>).

    The tag_slug listing returns every prop market and pages of 10+ MB, so for a
    fixed game list this is far cheaper. The moneyline market shares the event
    slug, so /markets?slug= (fast) is used instead of /events (5+ s per call).
    """
    games = read_table("games")
    games = games[(games.date >= str(start)) & (games.date <= str(end))]
    folder = CACHE / "events"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for g in games.itertuples():
        slug = f"nba-{g.away_team.lower()}-{g.home_team.lower()}-{g.date}"
        path = folder / f"{slug}.json"
        if path.exists():
            event = json.loads(path.read_text())
        else:
            time.sleep(POLITE_DELAY)
            page = get_json(f"{GAMMA}/markets", {"slug": slug, "closed": "true"})
            event = {"slug": slug, "markets": [m for m in page or []
                                               if m.get("sportsMarketType", "moneyline") == "moneyline"]}
            path.write_text(json.dumps(event))
        if event["markets"]:
            out.append(event)
    return out


def parse_slug(slug: str) -> dict:
    """nba-was-bkn-2026-02-07 -> away WAS, home BKN, date 2026-02-07 (Eastern)."""
    _, away, home, *day = slug.split("-")
    return {"away_team": away.upper(), "home_team": home.upper(), "date": "-".join(day)}


def history(token: str, tip: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / f"{token[:40]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    time.sleep(POLITE_DELAY)
    page = get_json(f"{CLOB}/prices-history", {"market": token, "fidelity": 1,
                                                "startTs": int((tip - WINDOW).timestamp()),
                                                "endTs": int((tip + pd.Timedelta(hours=4)).timestamp())})
    frame = pd.DataFrame(page.get("history", []), columns=["t", "p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def build(start: date, end: date, by_slug: bool = False):
    """by_slug: look games up one by one and fetch only the first outcome's
    history; the second outcome's traded price is taken as 1 - p (the two
    tokens are complementary on the CLOB)."""
    games = read_table("games")
    codes = nickname_codes()
    markets, prices, settlements = [], [], []
    for e in (events_by_slug(start, end) if by_slug else events(start, end)):
        info = parse_slug(e["slug"])
        for m in e.get("markets", []):
            if m.get("sportsMarketType") != "moneyline" or not m.get("gameStartTime"):
                continue
            tip = pd.Timestamp(m["gameStartTime"])
            outcomes, tokens = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])
            finals = json.loads(m.get("outcomePrices") or "[]")
            first = None
            for i, (name, token) in enumerate(zip(outcomes, tokens)):
                team = codes.get(name.lower())
                if team is None:
                    continue
                ticker = f"PM-{e['slug']}-{team}"
                markets.append({"venue": "polymarket", "market_ticker": ticker, "kind": "game", **info,
                                "team": team, "player_id": None, "line": None, "title": m.get("question", "")})
                if by_slug and first is not None:
                    h = first.assign(p=1 - first.p)
                else:
                    h = first = history(token, tip)
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


def _print_live_snapshot(snapshot: LiveNBASnapshot, limit: int = 20) -> None:
    quality = snapshot.quality
    print(f"Polymarket NBA live | {quality.status} | {quality.source_state}")
    print(f"fetched_at={quality.fetched_at.isoformat()} markets={len(snapshot.markets)}")
    print(snapshot.message)
    for reason in quality.reasons:
        print(f"warning: {reason}")
    for market in snapshot.markets[:max(0, limit)]:
        start = market.market_start_time.isoformat() if market.market_start_time else "unknown start"
        print(f"\n{market.event_title or market.event_slug} | {market.question} | {start}")
        for outcome in market.outcomes:
            price = f"{outcome.price:.1%}" if outcome.price is not None else "unavailable"
            bid = f"{outcome.best_bid:.3f}" if outcome.best_bid is not None else "-"
            ask = f"{outcome.best_ask:.3f}" if outcome.best_ask is not None else "-"
            print(f"  {outcome.outcome or '<unnamed>'}: {price} (bid {bid}, ask {ask})")
    if len(snapshot.markets) > limit:
        print(f"\n... {len(snapshot.markets) - limit} additional markets not shown; use --limit to change this.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat,
                    help="historical replay start date; requires --end")
    ap.add_argument("--end", type=date.fromisoformat,
                    help="historical replay end date; requires --start")
    ap.add_argument("--by-slug", action="store_true",
                    help="historical: look up each known game by slug instead of listing all NBA events")
    ap.add_argument("--limit", type=int, default=20, help="maximum live markets to print")
    ap.add_argument("--timeout", type=float, default=10, help="per-request timeout in seconds")
    ap.add_argument("--json", action="store_true", help="print the normalized live result as JSON")
    ap.add_argument("--save-history", action="store_true", help="append this live refresh to data/live")
    args = ap.parse_args()
    if bool(args.start) != bool(args.end):
        ap.error("--start and --end must be provided together")
    if args.start and args.end:
        build(args.start, args.end, args.by_slug)
        return

    snapshot = PolymarketClient(timeout=args.timeout).fetch_live_nba()
    if args.save_history and snapshot.markets:
        write_snapshot_history(snapshot)
    if args.json:
        print(json.dumps(snapshot.to_dict(), default=str, indent=2))
    else:
        _print_live_snapshot(snapshot, args.limit)


if __name__ == "__main__":
    main()
