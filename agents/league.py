"""League: practise on real historical Kalshi NBA markets with play money, ranked on skill, not luck.

    from agents import league
    lg = league.create("friday", rp, period="test", n_games=10)     # one seeded slate, the same for everyone
    league.add_bots(lg, rp, agent.policy, plain_policy(forecaster)) # house bots decide on the same slate
    league.join(lg, "alice")
    info = league.decision_info(forecaster, rp, lg, 0, names)       # as-of only: prices, news, Coach snapshot
    league.pick(lg, "alice", 0, "home", 20, rp, info["snapshot"])    # ask + fee, settles at the final result
    league.leaderboard(lg, games)                                   # ranked by mean CLV, P&L alongside, badge
    league.save(lg); league.load("friday")

A round is a fixed slate of real decision points (one per game) from the test or
play-off holdout period, chosen with a seed derived from the league name. At
each one the player sees exactly what the Coach sees at that moment and backs
the home team, backs the away team or passes. Fills and settlement go through
the replay (coach.paper_trade), so a player's trade costs what the agent's would.

Ranking is by mean closing-line value (CLV) per contract, among players with at
least MIN_TRADES trades. One night's P&L is mostly luck; beating the closing
price repeatedly is skill. The badge uses the day-clustered bootstrap from
evaluation/stats.py: "skill" when the 95% CI of mean CLV is above 0, "costs"
when it is below 0, otherwise "too early to tell".

Leagues are JSON files in league_data/ (git-ignored). Play money only.
"""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agents.coach import NOTICE as COACH_NOTICE, explain, lessons, paper_trade, snapshot
from data_sources import ROOT
from evaluation.stats import clv_positive_test, day_table
from replay import LEAD, MAX_QUOTE_AGE, AsOf

DATA = ROOT / "league_data"
BANKROLL = 1000.0                # play-money dollars at the start of a season
STAKE_MIN, STAKE_MAX = 5.0, 50.0  # per pick; the max matches the agent's $50 order cap
N_GAMES = 10
MIN_TRADES = 5                   # trades needed to be ranked
PERIODS = {"test": ("2026-02-01", "2026-04-12"), "holdout": ("2026-04-13", "2026-06-14")}
CHOICES = ("home", "away", "pass")
BOTS = ("Never trade", "Agent", "Raw model")
NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
NOTICE = ("Play money only: no real money is deposited, bet or paid out, and nothing here links to a betting "
          "account. " + COACH_NOTICE)
BADGES = {"skill": "skill", "costs": "costs", "early": "too early to tell"}


def _clean(x):
    """JSON-safe copy: timestamps as ISO strings, numpy scalars as Python, NaN as None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _check_name(name: str, what: str) -> str:
    if not NAME.match(name or ""):
        raise ValueError(f"{what} must be 1-40 letters, digits, '-' or '_'")
    return name


def seed_for(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _game(rp, game_id):
    return next(g for g in rp.t["games"][rp.t["games"].game_id == game_id].itertuples())


def _tradeable(rp, game, now) -> bool:
    """Both teams' game-winner markets have a fresh two-sided quote at `now`."""
    view = AsOf(rp.t, now, rp.price_index)
    m = view.markets(game.game_id)
    m = m[m.kind == "game"]
    quotes = [view.quote(t) for t in m.market_ticker]
    return len(quotes) == 2 and all(q is not None and now - q.ts <= MAX_QUOTE_AGE and 0 < q.ask < 1 for q in quotes)


def choose_slate(rp, period: str = "test", n_games: int = N_GAMES, seed: int = 0) -> list:
    """n_games seeded (game, decision time) pairs from the period, in chronological order.

    A game's decision time is drawn from its injury-news times when it has any (the moments worth practising),
    otherwise it is the fixed LEAD decision before tip-off.
    """
    start, end = PERIODS[period]
    g = rp.t["games"]
    g = g[(g.date >= start) & (g.date <= end) & g.game_id.isin(rp.t["markets"].game_id)].dropna(subset=["tip_time"])
    rng = np.random.default_rng(seed)
    slate = []
    for i in rng.permutation(len(g)):
        game = next(g.iloc[[i]].itertuples())
        times = [t for t in rp.decision_times(game) if _tradeable(rp, game, t)]
        times = [t for t in times if t != game.tip_time - LEAD] or times
        if times:
            slate.append({"game_id": game.game_id, "as_of": pd.Timestamp(times[rng.integers(len(times))]).isoformat()})
        if len(slate) == n_games:
            break
    return sorted(slate, key=lambda s: s["as_of"])


def create(name: str, rp, period: str = "test", n_games: int = N_GAMES, bankroll: float = BANKROLL,
           slate: list | None = None) -> dict:
    _check_name(name, "League name")
    seed = seed_for(name)
    return _clean({"name": name, "period": period, "seed": seed, "bankroll": float(bankroll),
                   "created": pd.Timestamp.now(tz="UTC"), "notice": NOTICE,
                   "slate": slate if slate is not None else choose_slate(rp, period, n_games, seed),
                   "players": {}, "bots": {}})


def join(lg: dict, user: str) -> dict:
    _check_name(user, "Username")
    if user in BOTS:
        raise ValueError("That name belongs to a house bot")
    lg["players"].setdefault(user, {"joined": pd.Timestamp.now(tz="UTC").isoformat(), "picks": {}})
    return lg


def decision_info(forecaster, rp, lg: dict, idx: int, names: dict | None = None) -> dict:
    """Everything shown at slate decision idx, all from the as-of view: no later prices, news or the score."""
    item = lg["slate"][idx]
    now = pd.Timestamp(item["as_of"])
    view = AsOf(rp.t, now, rp.price_index)
    public = view.games()                                     # scores hidden until the game is final
    game = next(public[public.game_id == item["game_id"]].itertuples())
    s = snapshot(forecaster, view, game, names)
    m = view.markets(game.game_id)
    return {"index": idx, "game": game, "now": now, "view": view, "snapshot": s, "explain": explain(s),
            "lessons": lessons(s), "news": view.news(game.game_id),
            "prices": {r.team: view.prices(r.market_ticker) for r in m[m.kind == "game"].itertuples()}}


def _filled(picks: dict) -> list:
    return [p for p in picks.values() if p.get("filled")]


def balance(lg: dict, user: str) -> float:
    return lg["bankroll"] + sum(p["pnl"] or 0.0 for p in _filled(lg["players"][user]["picks"]))


def stake_cap(lg: dict, user: str) -> float:
    return max(0.0, min(STAKE_MAX, balance(lg, user)))


def next_index(lg: dict, user: str):
    picks = lg["players"][user]["picks"]
    return next((i for i in range(len(lg["slate"])) if str(i) not in picks), None)


def pick(lg: dict, user: str, idx: int, choice: str, stake: float, rp, snap: dict) -> dict:
    """Record a user's call at slate decision idx; fills and settles exactly like coach.paper_trade."""
    if user not in lg["players"]:
        raise ValueError(f"{user} has not joined {lg['name']}")
    if idx != next_index(lg, user):
        raise ValueError("Play the slate in order, one call per decision")
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}")
    item = lg["slate"][idx]
    now = pd.Timestamp(item["as_of"])
    if snap["game_id"] != item["game_id"] or pd.Timestamp(snap["as_of"]) != now:
        raise ValueError("snapshot is not for this decision")
    base = {"index": idx, "game_id": item["game_id"], "as_of": now, "choice": choice, "pick": snap.get("pick")}
    if choice == "pass":
        trade = {**base, "filled": False, "why": "pass", "stake": 0.0, "team": None}
    else:
        if not STAKE_MIN <= stake <= stake_cap(lg, user):
            raise ValueError(f"stake must be between ${STAKE_MIN:.0f} and ${stake_cap(lg, user):.0f}")
        team = snap[choice]
        side = snap["sides"].get(team)
        trade = ({**base, "filled": False, "why": "no_quote", "stake": float(stake), "team": team} if side is None
                 else {**paper_trade(rp, _game(rp, item["game_id"]), now, side, stake, snap), **base})
    lg["players"][user]["picks"][str(idx)] = trade = _clean(trade)
    return trade


def bot_picks(rp, slate: list, policy) -> dict:
    """A replay policy (agent or raw model) decides at each slate point; its first order fills and settles."""
    teams = dict(zip(rp.t["markets"].market_ticker, rp.t["markets"].team))
    out = {}
    for i, item in enumerate(slate):
        game, now = _game(rp, item["game_id"]), pd.Timestamp(item["as_of"])
        orders = (policy(AsOf(rp.t, now, rp.price_index), game, now) or []) if policy else []
        base = {"index": i, "game_id": game.game_id, "as_of": now}
        if not orders:
            out[str(i)] = _clean({**base, "filled": False, "why": "pass", "choice": "pass", "team": None})
            continue
        o = orders[0]
        fill, why = rp._fill(o, now, game.tip_time)
        team = teams.get(o.market_ticker)
        if o.side == "no":
            team = game.away_team if team == game.home_team else game.home_team
        choice = "home" if team == game.home_team else "away"
        if fill is None:
            out[str(i)] = _clean({**base, "filled": False, "why": why, "choice": choice, "team": team})
            continue
        s = rp._settle({**fill, "market_ticker": o.market_ticker, "side": o.side}, game.tip_time)
        won = None if s["outcome"] is None else bool((s["outcome"] == 1) == (o.side == "yes"))
        out[str(i)] = _clean({**base, "filled": True, "choice": choice, "team": team, "stake": o.stake,
                              "price": s["price"], "contracts": s["contracts"], "fee": s["fee"],
                              "close_price": s["close_price"], "clv": s["clv"],
                              "clv_dollars": s["clv"] * s["contracts"], "won": won, "pnl": s["pnl"]})
    return out


def add_bots(lg: dict, rp, agent_policy=None, raw_policy=None) -> dict:
    """House bots on the league's slate: never trade, the anchored agent and the raw model."""
    for name, policy in zip(BOTS, (None, agent_policy, raw_policy)):
        if name == "Never trade" or policy is not None:
            lg["bots"][name] = {"picks": bot_picks(rp, lg["slate"], policy)}
    return lg


def badge(trades: list, games: pd.DataFrame, min_trades: int = MIN_TRADES) -> dict:
    """skill / costs / too early to tell, from the day-clustered bootstrap CI of mean CLV per contract."""
    if len(trades) < min_trades:
        return {"badge": BADGES["early"] if trades else "no trades", "ci_low": None, "ci_high": None}
    fills = pd.DataFrame(trades)[["game_id", "clv", "contracts", "pnl", "price"]].astype(
        {"clv": float, "contracts": float, "pnl": float, "price": float})
    days = games.loc[games.game_id.isin(fills.game_id), "date"]
    r = clv_positive_test(day_table(fills, games, days))
    label = "skill" if r["ci_low"] > 0 else "costs" if r["ci_high"] < 0 else "early"
    return {"badge": BADGES[label], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}


def summary(picks: dict, games: pd.DataFrame, min_trades: int = MIN_TRADES) -> dict:
    """Decisions, trades, pass rate, mean CLV per contract, CLV $, P&L, fees, beat-the-close share, badge."""
    made = list(picks.values())
    t = _filled(picks)
    clv = [p["clv"] for p in t if p.get("clv") is not None]
    return {"decisions": len(made), "trades": len(t),
            "pass_rate": sum(p.get("choice") == "pass" for p in made) / len(made) if made else None,
            "mean_clv": float(np.mean(clv)) if clv else None,
            "clv_dollars": float(sum(p.get("clv_dollars") or 0.0 for p in t)),
            "pnl": float(sum(p.get("pnl") or 0.0 for p in t)), "fees": float(sum(p.get("fee") or 0.0 for p in t)),
            "beat_close": float(np.mean([c > 0 for c in clv])) if clv else None,
            **badge(t, games, min_trades)}


def leaderboard(lg: dict, games: pd.DataFrame, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    """Players and house bots ranked by mean CLV per contract (ranked only with min_trades); P&L alongside."""
    rows = [{"name": n, "kind": kind, **summary(p["picks"], games, min_trades)}
            for kind, group in (("player", lg["players"]), ("bot", lg["bots"])) for n, p in group.items()]
    if not rows:
        return pd.DataFrame(columns=["rank", "name", "kind"])
    df = pd.DataFrame(rows)
    df["ranked"] = df.trades >= min_trades
    df = df.assign(_clv=pd.to_numeric(df.mean_clv, errors="coerce")).sort_values(
        ["ranked", "_clv", "pnl"], ascending=[False, False, False], na_position="last").drop(columns="_clv")
    df.insert(0, "rank", [str(i + 1) if r else "-" for i, r in enumerate(df.ranked)])
    return df.reset_index(drop=True)


def compare(lg: dict, user: str, games: pd.DataFrame) -> pd.DataFrame:
    """The user against each house bot on only the decisions the user has played."""
    played = lg["players"][user]["picks"]
    rows = [{"name": user, **summary(played, games)}]
    for name, b in lg["bots"].items():
        rows.append({"name": name, **summary({k: v for k, v in b["picks"].items() if k in played}, games)})
    return pd.DataFrame(rows)


def path_for(name: str, folder: Path = DATA) -> Path:
    return Path(folder) / f"{_check_name(name, 'League name')}.json"


def save(lg: dict, folder: Path = DATA) -> Path:
    p = path_for(lg["name"], folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(lg), indent=1))
    tmp.replace(p)
    return p


def load(name: str, folder: Path = DATA) -> dict:
    return json.loads(path_for(name, folder).read_text())


def list_leagues(folder: Path = DATA) -> list:
    return sorted(p.stem for p in Path(folder).glob("*.json")) if Path(folder).exists() else []
