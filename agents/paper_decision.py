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


_RULE = None
_TAIL = "Trading agent paper decision. No sceptic, no LLM, and no live order."


def _published_rule():
    """MarketAgent.analyse numbers when that module imports; otherwise the same published constants.

    The dashboard decision does not construct the LangGraph, call an LLM, or start the sceptic.
    """
    global _RULE
    if _RULE is not None:
        return _RULE
    try:
        from agents.graph import DEFAULT_MIN_EDGE, clip, fee_per_contract
        _RULE = (float(DEFAULT_MIN_EDGE), clip, fee_per_contract)
    except Exception:
        def clip(p: float) -> float:
            return float(min(max(float(p), 0.02), 0.98))

        def fee_per_contract(price: float) -> float:
            return FEE_RATE * price * (1.0 - price)

        _RULE = (DEFAULT_GAP_CENTS / 100.0, clip, fee_per_contract)
    return _RULE


def _optional(value):
    if value is None or value == "":
        return None
    return _finite(value)


def _unit_price(value):
    number = _optional(value)
    if number is None or not 0 < number < 1:
        return None
    return number


def _blank(**extra) -> dict:
    row = {"status": "PASS", "reason": "", "edge": None, "edge_cents": None, "stake": None,
           "fee": None, "anchor": None, "m4_shift": None, "p_used": None, "ask_used": None,
           "forecast_source": None, "cap": None, "llm": False, "sceptic": False, "order_sent": False}
    row.update(extra)
    return row


def _apply_cap(amount: float, spent_game: float, spent_day: float) -> tuple[float, str]:
    """Clamp one paper stake the way the published caps bind: order, then game, then day."""
    stake, cap = amount, "none"
    if stake > ORDER_CAP:
        stake, cap = ORDER_CAP, "$50 order cap"
    if stake > GAME_CAP - spent_game:
        stake, cap = GAME_CAP - spent_game, "$100 game cap"
    if stake > DAY_CAP - spent_day:
        stake, cap = DAY_CAP - spent_day, "$300 day cap"
    return stake, cap


def trading_agent_decision(p=None, ask=None, gap_cents=DEFAULT_GAP_CENTS, stake=DEFAULT_STAKE, *,
                           anchor=None, m4_shift=None, live_ask=None,
                           spent_game=0.0, spent_day=0.0) -> dict:
    """Paper decision for one game, using the published MarketAgent edge rule.

    Forecast is clip(Kalshi mid about 24h before tip + M4 news shift, 0.02, 0.98) when both
    inputs are present. If the anchor, the M4 shift, or a live quote is missing, the result
    says so and uses the form probability or the form ask. It does not invent a price.
    A fill requires the edge after ``0.07 * price * (1 - price)`` to be strictly greater
    than the gap (4 cents in the published agent). The stake starts at $20 and is clamped
    by the $50 order, $100 game, and $300 day caps. No sceptic, no LLM, and no order.
    """
    _min_edge, clip, fee_per_contract = _published_rule()
    anchor_n, shift_n = _optional(anchor), _optional(m4_shift)
    quote = _unit_price(live_ask)
    form_p, form_ask = _optional(p), _unit_price(ask)
    spent_g = _optional(spent_game)
    spent_d = _optional(spent_day)
    spent_g = 0.0 if spent_g is None or spent_g < 0 else spent_g
    spent_d = 0.0 if spent_d is None or spent_d < 0 else spent_d

    if anchor_n is not None and shift_n is not None:
        raw = anchor_n + shift_n
        home = float(clip(raw))
        forecast = (f"Anchor {anchor_n:.4f} plus M4 shift {shift_n:+.4f} = {raw:.4f}, "
                    f"clipped to {home:.4f}.")
        source = "anchor+m4"
    else:
        missing = []
        if anchor_n is None:
            missing.append("Kalshi 24h anchor is missing")
        if shift_n is None:
            missing.append("M4 news shift is missing")
        if form_p is None:
            home, source = None, "form"
            forecast = f"{'. '.join(missing)}. Form home-win probability is missing."
        else:
            home, source = form_p, "form"
            forecast = (f"{'. '.join(missing)}. Forecast uses the form home-win probability {form_p:.4f}.")

    if quote is not None:
        price = quote
        quote_line = f"Live quote ask {price:.4f}."
    elif form_ask is None:
        price = None
        quote_line = "Live quote is missing. Form executable price is missing."
    else:
        price = form_ask
        quote_line = f"Live quote is missing. Executable price is the form ask {form_ask:.4f}."

    base = _blank(anchor=anchor_n, m4_shift=shift_n, p_used=home, ask_used=price, forecast_source=source)

    def finish(status, stake_out, edge, edge_cents, fee, cap, detail):
        reason = f"{forecast} {quote_line} {detail} {_TAIL}"
        return {**base, "status": status, "reason": reason, "stake": stake_out, "edge": edge,
                "edge_cents": edge_cents, "fee": fee, "cap": cap}

    if home is None or price is None:
        return finish("PASS", None, None, None, None, None,
                      "Home win probability or executable price is missing.")
    if source == "form" and (home < 0.02 or home > 0.98):
        return finish("PASS", None, None, None, None, None,
                      "Home win probability must be between 0.02 and 0.98.")
    gap = _finite(gap_cents)
    if gap is None:
        return finish("PASS", None, None, None, None, None, "Enter an edge gap in cents.")
    amount = _finite(stake)
    if amount is None or amount <= 0:
        return finish("PASS", None, None, None, None, None, "Stake must be greater than $0.")

    fee = float(fee_per_contract(price))
    edge = home - price - fee
    edge_cents = edge * 100
    # MarketAgent.analyse keeps a candidate only when the probability gap is strictly above min_edge.
    cleared = edge > _min_edge if abs(gap - (_min_edge * 100)) < 1e-9 else edge_cents > gap
    relation = "above" if cleared else "not above"
    edge_line = f"Fee: {fee * 100:.2f}¢. Edge {edge_cents:.2f}¢ is {relation} the {_gap_text(gap)}¢ gap."
    if cleared:
        filled, cap = _apply_cap(amount, spent_g, spent_d)
        if filled <= 0:
            which = cap or "$100 game cap"
            return finish("PASS", None, edge, edge_cents, fee, which,
                          f"{edge_line} Cap: {which} leaves no stake.")
        stake_line = f"Stake ${filled:.2f}"
        if cap == "$50 order cap":
            stake_line += " (capped at $50)"
        elif cap == "$100 game cap":
            stake_line += " (capped at $100 per game)"
        elif cap == "$300 day cap":
            stake_line += " (capped at $300 per day)"
        stake_line += "."
        cap_line = "Cap: none ($50 order, $100 game and $300 day not binding)." if cap == "none" else f"Cap: {cap}."
        return finish("PAPER FILL", filled, edge, edge_cents, fee, cap, f"{edge_line} {stake_line} {cap_line}")
    return finish("PASS", None, edge, edge_cents, fee, "not applied", f"{edge_line} Cap: not applied.")
