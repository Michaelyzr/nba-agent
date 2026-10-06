"""D4: chronological day-by-day replay over recorded market prices.

    python -m replay --start 2026-02-01 --end 2026-02-28 --policy record --name feb-record

For each game the replay asks the policy for orders at every news update in the
NEWS_WINDOW before tip-off and once at LEAD before tip. A policy only sees an
AsOf view: news published, games finished and quotes printed by that moment.
Orders go through a risk function, then fill at the recorded ask (or 1 - bid
for "no") plus fees, capped by recent volume. After each game settles the
replay records profit and closing-line value; after each day it calls
on_day_end, where the reviewer and gate plug in.

Policies, risk functions and day-end hooks are plain callables so the agents
subgroup can swap in forecast/api.py, policy/risk.py and the gate.
"""
import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import FROZEN, read_table
from nba_agent_paths import RUNS

LEAD = pd.Timedelta(minutes=60)
NEWS_WINDOW = pd.Timedelta(hours=6)
MAX_QUOTE_AGE = pd.Timedelta(minutes=5)
VOLUME_LOOKBACK = pd.Timedelta(minutes=60)
VOLUME_SHARE = 0.10
FALLBACK_MAX_CONTRACTS = 200


@dataclass
class Order:
    market_ticker: str
    side: str                     # "yes" or "no"
    p_model: float                # model probability that "yes" settles true
    stake: float                  # dollars
    reason: str
    citations: list = field(default_factory=list)
    channel: str = "platform"
    rules_applied: list = field(default_factory=list)
    llm: str = "none"


def kalshi_fee(contracts: int, price: float) -> float:
    """Kalshi taker fee: 7% of contracts x P x (1 - P), rounded up to the cent."""
    return math.ceil(0.07 * contracts * price * (1 - price) * 100 - 1e-9) / 100


def no_fee(contracts: int, price: float) -> float:
    return 0.0


class AsOf:
    """Read-only view of the frozen tables at one moment. Settlements are not reachable."""

    def __init__(self, tables: dict, now: pd.Timestamp, price_index: dict):
        self.now = now
        self._t = tables
        self._prices = price_index

    def games(self) -> pd.DataFrame:
        """Schedule for every game; scores only for games already final."""
        g = self._t["games"].copy()
        pending = g.final_at > self.now
        g.loc[pending, ["home_pts", "away_pts"]] = np.nan
        return g

    def player_games(self) -> pd.DataFrame:
        g = self._t["games"]
        done = g.loc[g.final_at <= self.now, "game_id"]
        return self._t["player_games"][self._t["player_games"].game_id.isin(done)]

    def news(self, game_id: str | None = None) -> pd.DataFrame:
        n = self._t["news"]
        n = n[n.published_at <= self.now]
        return n if game_id is None else n[n.game_id == game_id]

    def markets(self, game_id: str | None = None) -> pd.DataFrame:
        m = self._t["markets"]
        return m if game_id is None else m[m.game_id == game_id]

    def prices(self, ticker: str) -> pd.DataFrame:
        p = self._prices.get(ticker)
        if p is None:
            return pd.DataFrame(columns=["ts", "bid", "ask", "volume"])
        return p.iloc[: p.ts.searchsorted(self.now, side="right")]

    def quote(self, ticker: str):
        p = self.prices(ticker)
        return None if p.empty else p.iloc[-1]


def basic_risk(order: Order, ctx: dict, max_order=50.0, max_game=100.0, max_day=300.0):
    """Placeholder until policy/risk.py: caps, cited reason, channel permissions."""
    if order.channel in ("team", "media"):
        return False, "channel_cannot_order"
    if not order.reason.strip():
        return False, "missing_reason"
    if order.side not in ("yes", "no"):
        return False, "bad_side"
    if order.stake > max_order:
        return False, "over_order_cap"
    if ctx["spent_game"] + order.stake > max_game:
        return False, "over_game_cap"
    if ctx["spent_day"] + order.stake > max_day:
        return False, "over_day_cap"
    return True, "approved"


def cited_after_decision(decision: dict, news: pd.DataFrame) -> list:
    """Leakage check: cited news published after the decision time."""
    cited = news[news.news_id.isin(decision.get("citations", []))]
    return cited.loc[cited.published_at > decision["as_of"], "news_id"].tolist()


class Replay:
    def __init__(self, tables: dict, policy, risk=basic_risk, fee=kalshi_fee, on_day_end=None,
                 one_fill_per_market=True):
        self.t = tables
        self.policy = policy
        self.risk = risk
        self.fee = fee
        self.on_day_end = on_day_end
        self.one_fill_per_market = one_fill_per_market
        prices = tables["prices"].sort_values("ts")
        self.price_index = {k: v.reset_index(drop=True) for k, v in prices.groupby("market_ticker")}
        self.outcomes = dict(zip(tables["settlements"].market_ticker, tables["settlements"].outcome))
        self.decisions, self.fills = [], []

    def decision_times(self, game) -> list:
        tip = game.tip_time
        news = self.t["news"]
        times = news.loc[(news.game_id == game.game_id) & (news.published_at >= tip - NEWS_WINDOW)
                         & (news.published_at < tip), "published_at"]
        return sorted(set(times) | {tip - LEAD})

    def _fill(self, order: Order, now, tip):
        if now >= tip:
            return None, "post_tip"
        view = AsOf(self.t, now, self.price_index)
        q = view.quote(order.market_ticker)
        if q is None or now - q.ts > MAX_QUOTE_AGE:
            return None, "stale_quote"
        price = q.ask if order.side == "yes" else 1 - q.bid
        if not 0 < price < 1:
            return None, "no_quote"
        recent = view.prices(order.market_ticker)
        volume = recent.loc[recent.ts > now - VOLUME_LOOKBACK, "volume"].sum(min_count=1)
        cap = FALLBACK_MAX_CONTRACTS if pd.isna(volume) else int(volume * VOLUME_SHARE)
        contracts = min(math.floor(order.stake / price + 1e-9), cap)
        if contracts < 1:
            return None, "no_liquidity"
        return {"price": float(price), "contracts": contracts, "fee": self.fee(contracts, float(price)),
                "quote_ts": q.ts}, "filled"

    def _settle(self, fill: dict, tip):
        view = AsOf(self.t, tip, self.price_index)
        q = view.quote(fill["market_ticker"])
        close_yes = (q.bid + q.ask) / 2 if q is not None else np.nan
        close = close_yes if fill["side"] == "yes" else 1 - close_yes
        outcome = self.outcomes.get(fill["market_ticker"])
        won = None if outcome is None else (outcome == 1) == (fill["side"] == "yes")
        pnl = np.nan if won is None else fill["contracts"] * (float(won) - fill["price"]) - fill["fee"]
        return {**fill, "close_price": close, "clv": close - fill["price"], "outcome": outcome, "pnl": pnl}

    def run(self, start: str, end: str):
        games = self.t["games"]
        days = games[(games.date >= start) & (games.date <= end)].dropna(subset=["tip_time"])
        for day, today in days.groupby("date", sort=True):
            spent_day, spent_game, filled_markets, day_fills = 0.0, {}, set(), []
            events = sorted((t, g.game_id) for g in today.itertuples() for t in self.decision_times(g))
            by_id = {g.game_id: g for g in today.itertuples()}
            for now, game_id in events:
                game = by_id[game_id]
                view = AsOf(self.t, now, self.price_index)
                for order in self.policy(view, game, now) or []:
                    if self.one_fill_per_market and order.market_ticker in filled_markets:
                        continue
                    ctx = {"now": now, "tip_time": game.tip_time, "game_id": game_id,
                           "spent_game": spent_game.get(game_id, 0.0), "spent_day": spent_day}
                    ok, risk_result = self.risk(order, ctx)
                    fill, fill_result = self._fill(order, now, game.tip_time) if ok else (None, "blocked")
                    decision = {**asdict(order), "decision_id": f"{game_id}-{len(self.decisions)}",
                                "as_of": now, "game_id": game_id, "risk_result": risk_result,
                                "fill_result": fill_result}
                    self.decisions.append(decision)
                    if fill:
                        spent_game[game_id] = spent_game.get(game_id, 0.0) + order.stake
                        spent_day += order.stake
                        filled_markets.add(order.market_ticker)
                        day_fills.append(self._settle({**fill, "decision_id": decision["decision_id"],
                                                       "market_ticker": order.market_ticker, "side": order.side,
                                                       "p_model": order.p_model, "game_id": game_id,
                                                       "as_of": now}, game.tip_time))
            self.fills += day_fills
            if self.on_day_end:
                self.on_day_end(day, pd.DataFrame(day_fills), self)
        return pd.DataFrame(self.decisions), pd.DataFrame(self.fills)


def summary(fills: pd.DataFrame) -> dict:
    if fills.empty:
        return {"fills": 0}
    settled = fills.dropna(subset=["outcome"])
    p_side = np.where(settled.side == "yes", settled.p_model, 1 - settled.p_model)
    won = np.where(settled.side == "yes", settled.outcome == 1, settled.outcome == 0)
    return {"fills": len(fills), "staked": float((fills.price * fills.contracts).sum()),
            "pnl_after_fees": float(settled.pnl.sum()), "fees": float(fills.fee.sum()),
            "mean_clv": float(fills.clv.mean()), "brier": float(np.mean((p_side - won) ** 2)),
            "max_drawdown": float((settled.pnl.cumsum().cummax() - settled.pnl.cumsum()).max())}


def record_baseline(edge=0.05, stake=20.0, home_edge=0.03):
    """Smoke-test policy for game-winner markets: shrunk win rates to date, log5, home bump.

    Acts only at the fixed LEAD decision. Replace with forecast/api.py forecasts.
    """
    def policy(view: AsOf, game, now):
        if now != game.tip_time - LEAD:
            return []
        g = view.games().dropna(subset=["home_pts"])

        def rate(team_id):
            home, away = g[g.home_team_id == team_id], g[g.away_team_id == team_id]
            wins = (home.home_pts > home.away_pts).sum() + (away.away_pts > away.home_pts).sum()
            return (wins + 5) / (len(home) + len(away) + 10)

        h, a = rate(game.home_team_id), rate(game.away_team_id)
        p_home = min(max(h * (1 - a) / (h * (1 - a) + a * (1 - h)) + home_edge, 0.02), 0.98)
        orders = []
        for m in view.markets(game.game_id).itertuples():
            q = view.quote(m.market_ticker)
            if m.kind != "game" or q is None:
                continue
            p = p_home if m.team == game.home_team else 1 - p_home
            why = f"record baseline: {game.home_team} {h:.2f} vs {game.away_team} {a:.2f}, p({m.team})={p:.2f}"
            if p - q.ask > edge:
                orders.append(Order(m.market_ticker, "yes", p, stake, f"{why}, ask {q.ask:.2f}"))
            elif q.bid - p > edge:
                orders.append(Order(m.market_ticker, "no", p, stake, f"{why}, bid {q.bid:.2f}"))
        return orders

    return policy


POLICIES = {"record": record_baseline}


def load_tables(folder: Path = FROZEN) -> dict:
    tables = {n: read_table(n, folder) for n in ("games", "player_games", "markets", "prices", "settlements")}
    try:
        tables["news"] = read_table("news", folder)
    except FileNotFoundError:
        tables["news"] = pd.DataFrame(columns=["news_id", "published_at", "game_id", "player_id", "status"])
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--policy", choices=POLICIES, default="record")
    ap.add_argument("--name", default="run")
    args = ap.parse_args()
    decisions, fills = Replay(load_tables(), POLICIES[args.policy]()).run(args.start, args.end)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(out / "decisions.parquet", index=False)
    fills.to_parquet(out / "fills.parquet", index=False)
    print(f"{len(decisions)} decisions, {len(fills)} fills -> {out}")
    print(summary(fills))


if __name__ == "__main__":
    main()
