"""Unauthenticated read-only Polymarket order book adapter."""
from datetime import datetime, timezone
import math

import httpx


class MarketDataError(Exception):
    """The upstream returned a malformed public order book."""


def _parse_book(book, token_id):
    if str(book.get("asset_id", "")) != token_id:
        raise ValueError("Order book token does not match the requested token")
    levels = {}
    for side in ("bids", "asks"):
        values = [(float(v["price"]), float(v["size"])) for v in book.get(side, [])]
        if any(not math.isfinite(p) or not math.isfinite(s) or not 0 < p < 1 or s < 0 for p, s in values):
            raise ValueError("Invalid order book levels")
        levels[side] = [(p, s) for p, s in values if s > 0]
    if not levels["bids"] or not levels["asks"]:
        raise ValueError("The market does not have both bid and ask quotes")
    bid = max(p for p, _ in levels["bids"])
    ask = min(p for p, _ in levels["asks"])
    if bid > ask:
        raise ValueError("Crossed order book")
    observed = datetime.fromtimestamp(float(book["timestamp"]) / 1000, tz=timezone.utc)
    return {
        "bid_price": bid, "ask_price": ask,
        "market_probability": (bid + ask) / 2,
        # Only depth at the best ask supports entry at the displayed price.
        "liquidity": sum(p * s for p, s in levels["asks"] if p == ask),
        "observed_at": observed, "raw_data": book,
    }


def parse_book(book, token_id):
    try:
        return _parse_book(book, token_id)
    except (KeyError, TypeError, OverflowError, OSError) as exc:
        raise ValueError("Order book fields are missing or invalid") from exc


async def fetch_book(token_id):
    async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
        response = await client.get("https://clob.polymarket.com/book", params={"token_id": token_id})
        response.raise_for_status()
        try:
            return parse_book(response.json(), token_id)
        except ValueError as exc:
            raise MarketDataError("Upstream order book is invalid") from exc
