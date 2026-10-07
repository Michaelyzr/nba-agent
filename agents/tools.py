"""As-of tools for the LLM tool agent (agents/tool_agent.py).

Every tool is a pure function of the replay's AsOf view at the decision time
`now`: quotes printed, news published and games finished by then. Nothing
here can reach settlements or later prices. Tools return small JSON-ready
dicts with code-computed numbers, so the LLM only chooses tools and argues;
it never supplies a number the trade depends on.

Times are relative (minutes ago, minutes to tip), so prompts contain no
timestamps. With anonymise=True team abbreviations become TEAM_A (home) and
TEAM_B (away) and player names become "Player <id>".
"""
import math

import pandas as pd

from agents.graph import ANCHOR_LEAD, DEFAULT_MIN_EDGE, OUT_STATUSES, fee_per_contract
from replay import MAX_QUOTE_AGE, NEWS_WINDOW

MAX_HISTORY_HOURS = 24.0
ROTATION_GAMES = 10
ROTATION_MIN_MINUTES = 10.0

TOOL_SPECS = {
    "get_quote": ("team", "Current bid, ask, mid, spread, quote age and last-hour volume of the team's win contract."),
    "get_price_history": ("team, hours (0.5-24, default 6)",
                          "Mid-price summary over the last `hours`: start, now, min, max, change and six checkpoints."),
    "get_anchor": ("team", "Mid price 24 hours before tip-off (the anchored base rate) and the move since then."),
    "get_news": ("", "Inactive-list entries for this game published so far (latest status per player)."),
    "get_rotation": ("team", "Rotation players over the team's last 10 games: usual minutes and points."),
    "m4_win_prob": ("out (list of player_id)",
                    "M4 win model: each team's win probability before any absences and after the listed players are "
                    "out, and the shift between them."),
    "m6_predicted_move": ("", "M6 neural impact model: predicted move of the home mid between now and tip-off."),
    "fee": ("price", "Kalshi taker fee per contract at this price (0.07 x price x (1 - price))."),
    "breakeven": ("price", "Win probability needed to break even when buying at this price, after the fee."),
    "edge": ("team, estimate_p",
             "Cheapest way to back the team now (yes on its contract or no on the opponent's), the all-in cost and the "
             "gap after fees between estimate_p and that cost."),
}


def r4(x):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), 4)


def numbers(obj) -> list:
    """Every numeric leaf of a tool output (bools excluded)."""
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)] if math.isfinite(float(obj)) else []
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in numbers(x)]
    if isinstance(obj, (list, tuple)):
        return [v for x in obj for v in numbers(x)]
    return []


def lookup(obj, path: str):
    """Value at a dotted path such as "after.BOS" or "checkpoints.0.mid"; None if absent."""
    for part in str(path).split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return None
    return obj


class AsOfTools:
    def __init__(self, view, game, now, forecaster=None, impact=None, players=None, anonymise=False,
                 min_edge=DEFAULT_MIN_EDGE):
        if pd.Timestamp(view.now) != pd.Timestamp(now):
            raise ValueError("tools must be built on the view for the decision time")
        self.view, self.game, self.now = view, game, pd.Timestamp(now)
        self.forecaster, self.impact, self.min_edge = forecaster, impact, min_edge
        self.players = players or {}
        self.anonymise = anonymise
        self.labels = ({game.home_team: "TEAM_A", game.away_team: "TEAM_B"} if anonymise
                       else {game.home_team: game.home_team, game.away_team: game.away_team})
        self.home, self.away = self.labels[game.home_team], self.labels[game.away_team]
        markets = view.markets(game.game_id)
        markets = markets[markets.kind == "game"]
        self.tickers = {self.labels[m.team]: m.market_ticker for m in markets.itertuples() if m.team in self.labels}
        self.team_ids = {self.home: game.home_team_id, self.away: game.away_team_id}
        self.label_by_id = {v: k for k, v in self.team_ids.items()}

    # ---------------- helpers ----------------

    @property
    def teams(self) -> list:
        return [t for t in (self.home, self.away) if t in self.tickers]

    def other(self, team):
        return self.away if team == self.home else self.home

    def _team(self, team) -> str:
        t = str(team).strip().upper()
        if t not in self.tickers:
            raise ValueError(f"unknown team {team!r}; use one of {self.teams}")
        return t

    def _prices(self, team) -> pd.DataFrame:
        p = self.view.prices(self.tickers[team])
        return p[p.ts <= self.now]

    def _fresh(self, team):
        p = self._prices(team)
        if p.empty or self.now - p.ts.iloc[-1] > MAX_QUOTE_AGE:
            return None
        return p.iloc[-1]

    def fresh_teams(self) -> list:
        return [t for t in self.teams if self._fresh(t) is not None]

    def minutes_to_tip(self) -> float:
        return (self.game.tip_time - self.now).total_seconds() / 60

    def player_name(self, pid) -> str:
        return f"Player {int(pid)}" if self.anonymise else self.players.get(int(pid), f"Player {int(pid)}")

    @staticmethod
    def _mid(row) -> float:
        return float((row.bid + row.ask) / 2)

    def _mid_at(self, p: pd.DataFrame, t):
        before = p[p.ts <= t]
        return self._mid(before.iloc[-1]) if len(before) else None

    # ---------------- tools ----------------

    def get_quote(self, team):
        team = self._team(team)
        p = self._prices(team)
        if p.empty:
            return {"team": team, "error": "no quote yet"}
        q = p.iloc[-1]
        volume = p.loc[p.ts > self.now - pd.Timedelta(hours=1), "volume"].sum()
        return {"team": team, "bid": r4(q.bid), "ask": r4(q.ask), "mid": r4(self._mid(q)), "spread": r4(q.ask - q.bid),
                "age_minutes": r4((self.now - q.ts).total_seconds() / 60),
                "fresh": bool(self.now - q.ts <= MAX_QUOTE_AGE), "volume_last_hour": r4(volume)}

    def get_price_history(self, team, hours=6):
        team = self._team(team)
        hours = min(max(float(hours), 0.5), MAX_HISTORY_HOURS)
        p = self._prices(team)
        start = self.now - pd.Timedelta(hours=hours)
        window = p[p.ts > start]
        if p.empty:
            return {"team": team, "hours": hours, "error": "no quotes yet"}
        start_mid = self._mid_at(p, start)
        if start_mid is None:
            start_mid = self._mid(window.iloc[0])
        mids = (window.bid + window.ask) / 2 if len(window) else pd.Series([start_mid])
        checkpoints = []
        for k in range(5, -1, -1):
            t = self.now - pd.Timedelta(hours=hours * k / 6)
            m = self._mid_at(p, t)
            if m is not None:
                checkpoints.append({"minutes_ago": r4(hours * 60 * k / 6), "mid": r4(m)})
        now_mid = self._mid(p.iloc[-1])
        return {"team": team, "hours": hours, "start_mid": r4(start_mid), "mid_now": r4(now_mid),
                "change": r4(now_mid - start_mid), "min_mid": r4(min(mids.min(), start_mid)),
                "max_mid": r4(max(mids.max(), start_mid)), "volume": r4(window.volume.sum()),
                "updates": int(len(window)), "checkpoints": checkpoints}

    def get_anchor(self, team):
        team = self._team(team)
        p = self._prices(team)
        if p.empty:
            return {"team": team, "error": "no quotes yet"}
        early = p[p.ts <= self.game.tip_time - ANCHOR_LEAD]
        row = early.iloc[-1] if len(early) else p.iloc[0]
        anchor, now_mid = self._mid(row), self._mid(p.iloc[-1])
        return {"team": team, "anchor_mid": r4(anchor),
                "anchor_hours_before_tip": r4((self.game.tip_time - row.ts).total_seconds() / 3600),
                "mid_now": r4(now_mid), "move_since_anchor": r4(now_mid - anchor)}

    def news_rows(self) -> pd.DataFrame:
        n = self.view.news(self.game.game_id)
        n = n[(n.published_at <= self.now) & (n.published_at >= self.game.tip_time - NEWS_WINDOW)]
        return n.sort_values("published_at").groupby("player_id").tail(1)

    def get_news(self):
        rows = []
        for r in self.news_rows().itertuples():
            team = self.label_by_id.get(getattr(r, "team_id", None))
            status = str(r.status).lower()
            text = (f"{self.player_name(r.player_id)} ({team or 'unknown team'}) {status}" if self.anonymise
                    else str(getattr(r, "text", "") or f"{self.player_name(r.player_id)} {status}"))
            rows.append({"news_id": r.news_id, "minutes_ago": r4((self.now - r.published_at).total_seconds() / 60),
                         "team": team, "player_id": int(r.player_id), "player": self.player_name(r.player_id),
                         "status": status, "text": text})
        return {"count": len(rows), "items": rows,
                "note": "inactive lists; out and doubtful players are not expected to play"}

    def out_from_news(self) -> list:
        n = self.news_rows()
        return [int(p) for p, s in zip(n.player_id, n.status) if str(s).lower() in OUT_STATUSES]

    def _recent_player_games(self, team) -> pd.DataFrame:
        pg = self.view.player_games()
        tid = self.team_ids[team]
        rows = pg[pg.team_id == tid]
        if rows.empty:
            return rows
        order = self.view.games().set_index("game_id").tip_time
        recent = (rows.drop_duplicates("game_id").assign(tip=lambda d: d.game_id.map(order))
                  .sort_values("tip").game_id.tail(ROTATION_GAMES))
        return rows[rows.game_id.isin(recent)]

    def get_rotation(self, team):
        team = self._team(team)
        rows = self._recent_player_games(team)
        if rows.empty:
            return {"team": team, "players": []}
        g = rows.groupby("player_id").agg(games=("game_id", "nunique"), min=("min", "mean"), pts=("pts", "mean"))
        g = g[g["min"] >= ROTATION_MIN_MINUTES].sort_values("min", ascending=False).head(10)
        return {"team": team, "games_considered": int(rows.game_id.nunique()),
                "players": [{"player_id": int(pid), "player": self.player_name(pid), "games": int(r.games),
                             "minutes": r4(r["min"]), "points": r4(r.pts)} for pid, r in g.iterrows()]}

    def known_players(self) -> set:
        ids = set(int(p) for p in self.news_rows().player_id)
        for team in self.teams:
            ids |= set(int(p) for p in self._recent_player_games(team).player_id)
        return ids

    def m4_win_prob(self, out=()):
        if self.forecaster is None:
            return {"error": "M4 is not loaded"}
        asked = []
        for x in (out or []):
            try:
                asked.append(int(x))
            except (TypeError, ValueError):
                continue
        known = self.known_players()
        used = sorted(p for p in set(asked) if p in known)
        before = self.forecaster.win(self.view, self.game, out=[])["p_home"]
        after = self.forecaster.win(self.view, self.game, out=used)["p_home"]
        return {"before": {self.home: r4(before), self.away: r4(1 - before)},
                "after": {self.home: r4(after), self.away: r4(1 - after)},
                "shift": {self.home: r4(after - before), self.away: r4(before - after)},
                "out_used": used, "ignored": sorted(set(asked) - set(used))}

    def m6_predicted_move(self):
        if self.impact is None:
            return {"error": "M6 is not loaded"}
        pred = self.impact.predict(self.view, self.game, self.now)
        if pred is None:
            return {"error": "no M6 prediction (no fresh home quote)"}
        return {"home": self.home, "home_mid_now": r4(pred["mid"]), "predicted_move_home": r4(pred["move"]),
                "move_q10": r4(pred["q10"]), "move_q90": r4(pred["q90"]),
                "home_mid_at_tip_estimate": r4(pred["mid"] + pred["move"])}

    def fee(self, price):
        price = float(price)
        if not 0 < price < 1:
            raise ValueError("price must be between 0 and 1")
        return {"price": r4(price), "fee_per_contract": r4(fee_per_contract(price))}

    def breakeven(self, price):
        price = float(price)
        if not 0 < price < 1:
            raise ValueError("price must be between 0 and 1")
        fee = fee_per_contract(price)
        return {"price": r4(price), "fee_per_contract": r4(fee), "breakeven_probability": r4(price + fee)}

    def edge(self, team, estimate_p):
        out = self.route(team, estimate_p)
        out.pop("ticker", None)
        return out

    def route(self, team, estimate_p):
        """edge() plus the market ticker to order on; tickers stay out of prompts (they contain dates and teams)."""
        team, p = self._team(team), float(estimate_p)
        if not 0 <= p <= 1:
            raise ValueError("estimate_p must be a probability")
        routes = []
        q = self._fresh(team)
        if q is not None and 0 < q.ask < 1:
            routes.append(("yes", self.tickers[team], float(q.ask)))
        opp = self.other(team)
        q = self._fresh(opp) if opp in self.tickers else None
        if q is not None and 0 < 1 - q.bid < 1:
            routes.append(("no", self.tickers[opp], float(1 - q.bid)))
        if not routes:
            return {"team": team, "error": "no fresh quote to trade on"}
        side, ticker, price = min(routes, key=lambda r: r[2] + fee_per_contract(r[2]))
        fee = fee_per_contract(price)
        gap = p - price - fee
        return {"team": team, "estimate_p": r4(p), "route": f"{side} on {self.labels_of(ticker)} contract",
                "side": side, "ticker": ticker, "price": r4(price), "fee": r4(fee), "all_in_cost": r4(price + fee),
                "gap_after_fees": r4(gap), "min_edge": self.min_edge, "clears_min_edge": bool(gap > self.min_edge)}

    def labels_of(self, ticker) -> str:
        return next((t for t, k in self.tickers.items() if k == ticker), "?")

    # ---------------- dispatch ----------------

    def call(self, name: str, args: dict | None = None) -> dict:
        args = dict(args or {})
        fn = getattr(self, name, None) if name in TOOL_SPECS else None
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except (ValueError, KeyError) as exc:
            return {"error": str(exc)}
