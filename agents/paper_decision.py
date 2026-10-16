"""One paper decision from the published edge rule.

No quotes are fetched and no order, sceptic, or model call is made.
The fee is the published Kalshi taker fee on the executable price:
``0.07 * ask * (1 - ask)``. A paper fill requires the edge in cents to be
strictly greater than the chosen gap. The stake is clamped to the $50
order cap. The $100/game and $300/day caps are published limits for a
full session; this function does not keep a running total.
"""
from __future__ import annotations

import math

ORDER_CAP = 50.0
GAME_CAP = 100.0
DAY_CAP = 300.0
FEE_RATE = 0.07
DEFAULT_GAP_CENTS = 4.0
DEFAULT_STAKE = 20.0


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def paper_decision(p, ask, gap_cents=DEFAULT_GAP_CENTS, stake=DEFAULT_STAKE) -> dict:
    """Return a PAPER FILL or a PASS. ``edge = p - ask - 0.07*ask*(1-ask)``."""
    home = _finite(p)
    price = _finite(ask)
    if home is None or price is None:
        return _pass("Home win probability or executable price is missing.")
    if home < 0.02 or home > 0.98:
        return _pass("Home win probability must be between 0.02 and 0.98.")
    if not 0 < price < 1:
        return _pass("Executable price must be greater than 0 and less than 1.")

    gap = _finite(gap_cents)
    if gap is None:
        return _pass("Enter an edge gap in cents.")
    amount = _finite(stake)
    if amount is None or amount <= 0:
        return _pass("Stake must be greater than $0.")

    fee = FEE_RATE * price * (1 - price)
    edge = home - price - fee
    edge_cents = edge * 100
    fill_stake = min(amount, ORDER_CAP)
    if edge_cents > gap:
        return {"status": "PAPER FILL", "reason": _edge_text(edge_cents, gap, cleared=True),
                "edge": edge, "edge_cents": edge_cents, "stake": fill_stake, "fee": fee}
    return {"status": "PASS", "reason": _edge_text(edge_cents, gap, cleared=False),
            "edge": edge, "edge_cents": edge_cents, "stake": None, "fee": fee}


def _edge_text(edge_cents: float, gap: float, cleared: bool) -> str:
    relation = "above" if cleared else "not above"
    return f"Edge {edge_cents:.2f}¢ is {relation} the {_gap_text(gap)}¢ gap."


def _gap_text(gap: float) -> str:
    rounded = round(gap, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _pass(reason: str) -> dict:
    return {"status": "PASS", "reason": reason, "edge": None, "edge_cents": None,
            "stake": None, "fee": None}
