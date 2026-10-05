"""Coach: explain a replayed game and teach how betting markets work, with paper money only.

    from agents.coach import snapshot, explain, lessons, paper_trade, feedback
    s = snapshot(forecaster, view, game, names)      # what is known at this decision time
    explain(s); lessons(s)                           # plain-language story and concept cards
    paper_trade(replay, game, now, s["sides"]["BOS"], stake=20)   # fills like the agent, settles at the end
    feedback(trades)                                 # habits: long shots, chasing moved prices, closing-line value

Every number comes from the as-of view (no future information) and the same
models, market anchor and fees as the trading agent. Backing a team means
buying YES on that team's game-winner contract.
"""
import pandas as pd

from agents.graph import BANNED, DEFAULT_MIN_EDGE, OUT_STATUSES, MarketAgent, clip, fee_per_contract, minutes_between
from replay import MAX_QUOTE_AGE, Order

LONG_SHOT = 0.35                 # side price at or below this counts as a long shot
CHASE_MOVE = 0.02                # price already moved this much toward the side since the anchor
NOTICE = ("Educational simulation on past games with paper money. Not betting advice. Real-money prediction "
          "markets are for adults only where legal, and most bets lose after fees. If betting stops being fun, "
          "contact your local problem-gambling support service.")


def _pts(x: float) -> str:
    return f"{x * 100:+.1f} points"


def snapshot(forecaster, view, game, names: dict | None = None) -> dict:
    """Model, market and news for one game at view.now, from the bettor's point of view (back a team)."""
    names = names or {}
    news = view.news(game.game_id).sort_values("published_at")
    latest = news.groupby("player_id").tail(1)
    out = [p for p, st in zip(latest.player_id, latest.status) if str(st).lower() in OUT_STATUSES]
    before = forecaster.win(view, game, [])["p_home"]
    after = forecaster.win(view, game, out)["p_home"]
    sides = {}
    for m in view.markets(game.game_id).itertuples():
        q = view.quote(m.market_ticker)
        if m.kind != "game" or q is None or view.now - q.ts > MAX_QUOTE_AGE or not 0 < q.ask < 1:
            continue
        home = m.team == game.home_team
        p_model = after if home else 1 - after
        shift = (after - before) * (1 if home else -1)
        anchor = MarketAgent._anchor_mid(view, m.market_ticker, game)
        p = clip(anchor + shift) if anchor is not None else p_model
        mid, fee = (q.bid + q.ask) / 2, fee_per_contract(q.ask)
        sides[m.team] = {"team": m.team, "ticker": m.market_ticker, "bid": float(q.bid), "ask": float(q.ask),
                         "mid": float(mid), "fee": fee, "breakeven": float(q.ask) + fee, "anchor": anchor,
                         "move": float(mid - anchor) if anchor is not None else 0.0, "p_model": p_model, "p": p,
                         "gap": p - float(q.ask) - fee}
    best = max(sides.values(), key=lambda s: s["gap"], default=None)
    return {"game_id": game.game_id, "home": game.home_team, "away": game.away_team, "as_of": view.now,
            "hours_to_tip": minutes_between(view.now, game.tip_time) / 60,
            "news": [{"player": names.get(r.player_id, str(r.player_id)), "status": str(r.status),
                      "published_at": r.published_at} for r in latest.itertuples()],
            "out": [names.get(p, str(p)) for p in out], "p_home_before": before, "p_home_after": after,
            "shift": after - before, "sides": sides, "best": best,
            "pick": best["team"] if best is not None and best["gap"] > DEFAULT_MIN_EDGE else None}


def explain(s: dict) -> list:
    """The game in plain language, in the order a careful bettor would think about it."""
    home, away = s["home"], s["away"]
    lines = [f"{away} at {home}, tip-off in {s['hours_to_tip']:.1f} hours."]
    if s["news"]:
        lines.append("News so far: " + "; ".join(f"{n['player']} is {n['status'].lower()}" for n in s["news"]) + ".")
    else:
        lines.append("No injury news yet.")
    pb, pa = s["p_home_before"], s["p_home_after"]
    lines.append(f"Our model gives {home} a {pa:.0%} chance to win"
                 + (f", {_pts(pa - pb)} from {pb:.0%} before the news." if abs(pa - pb) >= 0.005 else "."))
    h = s["sides"].get(home)
    if h is None:
        lines.append("There is no fresh market price for this game, so there is nothing to compare against.")
        return lines
    moved = "" if h["anchor"] is None else (
        f"; 24 hours before tip the mid was {h['anchor']:.0%}, " +
        (f"a move of {_pts(h['move'])}" if abs(h["move"]) >= 0.005 else "so the price has not moved"))
    lines.append(f"The market prices {home} at {h['bid']:.0%} to {h['ask']:.0%}{moved}.")
    lines.append(f"The market is usually better informed than our model, so we start from its earlier price "
                 f"and add only our news shift: {home} {h['p']:.0%}, {away} {1 - h['p']:.0%}.")
    b = s["best"]
    if s["pick"]:
        lines.append(f"Backing {b['team']} costs {b['ask']:.0%} plus a {b['fee'] * 100:.1f}-cent fee, so it breaks "
                     f"even at {b['breakeven']:.0%}. We estimate {b['p']:.0%}: an edge of {_pts(b['gap'])}, above our "
                     f"{DEFAULT_MIN_EDGE:.0%} threshold. On these numbers alone that is a candidate; the agent "
                     f"still checks its learned rules and risk limits before ordering.")
    else:
        lines.append(f"No team clears fees by our {DEFAULT_MIN_EDGE:.0%} threshold (best: {b['team']} at "
                     f"{_pts(b['gap'])}). Passing is the disciplined call.")
    return lines


def lessons(s: dict) -> list:
    """Concept cards triggered by what is happening in this game right now."""
    cards = []
    h = s["sides"].get(s["home"])
    if h:
        cards.append({"id": "price_is_probability", "title": "A price is a probability",
                      "body": f"A {s['home']} contract at {h['ask']:.0%} costs {h['ask'] * 100:.0f} cents and pays "
                              f"$1 if {s['home']} wins. Buying it only makes sense if you think {s['home']} wins "
                              f"more than {h['ask']:.0%} of the time."})
        cards.append({"id": "fees_and_spread", "title": "The spread and the fee move your break-even",
                      "body": f"You buy at the ask ({h['ask']:.0%}), not the mid ({h['mid']:.1%}), and pay a fee of "
                              f"{h['fee'] * 100:.1f} cents per contract (7% x price x (1 - price)). You need "
                              f"{h['breakeven']:.1%} just to break even. Small edges disappear into these costs."})
    if abs(s["shift"]) >= 0.01:
        who = ", ".join(s["out"]) or "the missing players"
        cards.append({"id": "injury_shift", "title": "How injury news moves a win probability",
                      "body": f"With {who} out, our model moves {s['home']} by {_pts(s['shift'])}. The model "
                              f"subtracts each missing player's usual minutes and points from his team: losing a "
                              f"starter matters far more than losing a bench player."})
    for side in s["sides"].values():
        if abs(side["move"]) >= CHASE_MOVE:
            shift = s["shift"] if side["team"] == s["home"] else -s["shift"]
            cards.append({"id": "priced_in", "title": "Is the news already in the price?",
                          "body": f"The {side['team']} price has moved {_pts(side['move'])} since 24 hours before "
                                  f"tip, while our model's news shift for {side['team']} is {_pts(shift)}. Markets "
                                  f"react within minutes. Buying after a big move ('chasing') usually means paying "
                                  f"for news everyone already knows."})
            break
    cheap = [x for x in s["sides"].values() if x["ask"] <= LONG_SHOT]
    if cheap:
        c = cheap[0]
        cards.append({"id": "long_shot", "title": "Long shots look cheap",
                      "body": f"{c['team']} at {c['ask']:.0%} pays about {1 / c['ask']:.1f} times your money, which "
                              f"feels attractive. But it loses about {1 - c['ask']:.0%} of the time if the price is "
                              f"right, and an over-confident model sees fake value here most often. Our learning "
                              f"agent taught itself to skip sides priced at 35 cents or less."})
    if s["hours_to_tip"] <= 1.5:
        cards.append({"id": "closing_line", "title": "The closing line is the scoreboard",
                      "body": "The price at tip-off reflects nearly everything public. Professionals judge a bet by "
                              "whether they bought below that closing price (closing-line value), not by whether "
                              "it won: one result is mostly luck, beating the close repeatedly is skill."})
    if s["sides"] and not s["pick"]:
        cards.append({"id": "passing", "title": "Passing is a position",
                      "body": "Most games offer no edge after fees. The agent passes on the large majority of "
                              "decision times; betting every game is the fastest way to lose to the spread and fees."})
    return cards


def paper_trade(rp, game, now, side: dict, stake: float, snap: dict | None = None) -> dict:
    """Back side["team"] with paper money: fill at the recorded ask plus fee (volume-capped), settle at the end."""
    order = Order(side["ticker"], "yes", side["p"], float(stake), "paper trade")
    fill, why = rp._fill(order, now, game.tip_time)
    base = {"game_id": game.game_id, "as_of": now, "team": side["team"], "stake": float(stake),
            "gap": side["gap"], "move": side["move"], "pick": (snap or {}).get("pick")}
    if fill is None:
        return {**base, "filled": False, "why": why}
    s = rp._settle({**fill, "market_ticker": side["ticker"], "side": "yes"}, game.tip_time)
    return {**base, "filled": True, "price": s["price"], "contracts": s["contracts"], "fee": s["fee"],
            "close_price": s["close_price"], "clv": s["clv"], "clv_dollars": s["clv"] * s["contracts"],
            "won": None if s["outcome"] is None else bool(s["outcome"] == 1), "pnl": s["pnl"],
            "long_shot": s["price"] <= LONG_SHOT, "chased": side["move"] >= CHASE_MOVE}


def feedback(trades: list) -> dict:
    """Totals and habit tips over a user's filled paper trades."""
    t = pd.DataFrame([x for x in trades if x.get("filled")])
    if t.empty:
        return {"trades": 0, "tips": ["No paper trades yet. Pick a game and a decision time, then make your call."]}
    out = {"trades": len(t), "pnl": float(t.pnl.sum()), "clv_dollars": float(t.clv_dollars.sum()),
           "mean_clv": float(t.clv.mean()), "long_shot_share": float(t.long_shot.mean()),
           "chased_share": float(t.chased.mean()), "negative_edge_share": float((t.gap < 0).mean())}
    tips = []
    if out["mean_clv"] < 0:
        tips.append(f"On average you paid {-out['mean_clv'] * 100:.1f} cents more than the closing price: the market "
                    f"usually knew first. Act on news earlier, or pass.")
    else:
        tips.append(f"You beat the closing price by {out['mean_clv'] * 100:.1f} cents on average. That is the "
                    f"sign of good timing; keep checking it over many trades.")
    if len(t) >= 3 and out["long_shot_share"] >= 0.5:
        tips.append(f"{out['long_shot_share']:.0%} of your trades were long shots (35 cents or less). Cheap contracts "
                    f"lose most of the time; make sure the edge is real, not just a big payout.")
    if len(t) >= 3 and out["chased_share"] >= 0.5:
        tips.append(f"{out['chased_share']:.0%} of your trades came after the price had already moved your way. "
                    f"That is chasing: the news was probably priced in.")
    if out["negative_edge_share"] > 0:
        tips.append(f"{out['negative_edge_share']:.0%} of your trades had a negative edge after fees by our "
                    f"estimate. Check the break-even price before you buy.")
    if len(t) < 20:
        tips.append(f"With {len(t)} trade{'s' if len(t) > 1 else ''}, profit or loss is mostly luck. Closing-line value tells you more.")
    out["tips"] = tips
    return out


def banned_words(texts) -> list:
    """Coach copy must never promise wins ("lock", "guaranteed", "risk-free")."""
    return [m.group(0) for t in texts for m in [BANNED.search(t)] if m]
