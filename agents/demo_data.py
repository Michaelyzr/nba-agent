"""Synthetic news, markets, prices and settlements on top of the prototype season, for running the agent loop.

The market prices a game from win rates plus noise, and reacts to injury news
MARKET_LAG after it is published. Agents that act on news before the price
moves should earn positive closing-line value; trades without news should not.
These are invented numbers for developing the loop; never report them as results.
"""
import numpy as np
import pandas as pd

from agents.graph import OUT_MINUTE_VALUE, clip, log5_home, win_rate
from forecast.dev_data import synthetic_tables

MARKET_NOISE = 0.05
MARKET_LAG = pd.Timedelta(minutes=10)
MARKET_REACTION = pd.Timedelta(minutes=20)
HALF_SPREAD = 0.01
ROTATION_MIN = 20.0
ROTATION_STALE = pd.Timedelta(days=14)


def absences(player_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Rotation players (ROTATION_MIN+ over their last 10 games for the team) missing from each game.

    A player whose latest game was for another team has been traded, not ruled out.
    """
    rows = player_games.merge(games[["game_id", "tip_time"]], on="game_id").sort_values("tip_time")
    recent, last_tip, last_team, out = {}, {}, {}, []
    for (game_id, tip), game in rows.groupby(["game_id", "tip_time"], sort=True):
        for team_id, team_rows in game.groupby("team_id"):
            present = set(team_rows.player_id)
            for p, mins in recent.get(team_id, {}).items():
                usual = float(np.mean(mins))
                if (p not in present and last_team[p] == team_id and usual >= ROTATION_MIN
                        and tip - last_tip[p] <= ROTATION_STALE):
                    out.append({"game_id": game_id, "player_id": p, "team_id": team_id, "usual_min": usual})
            team_recent = recent.setdefault(team_id, {})
            for p, m in zip(team_rows.player_id, team_rows["min"]):
                team_recent[p] = (team_recent.get(p, []) + [m])[-10:]
                last_tip[p], last_team[p] = tip, team_id
    return pd.DataFrame(out, columns=["game_id", "player_id", "team_id", "usual_min"])


def synthetic_replay_tables(start: str, end: str, seed: int = 0) -> dict:
    player_games, games = synthetic_tables()
    rng = np.random.default_rng(seed)
    window = games[(games.date >= start) & (games.date <= end)]
    missing = absences(player_games, games)
    missing = missing[missing.game_id.isin(window.game_id)]

    news, markets, prices, settlements = [], [], [], []
    for g in window.itertuples():
        done = games[games.final_at <= g.tip_time - pd.Timedelta(hours=7)]
        p_pre = clip(log5_home(win_rate(done, g.home_team_id), win_rate(done, g.away_team_id))
                     + rng.normal(0, MARKET_NOISE))
        shift, react = 0.0, None
        for m in missing[missing.game_id == g.game_id].itertuples():
            published = g.tip_time - pd.Timedelta(minutes=int(rng.integers(90, 300)))
            news.append({"news_id": f"syn-{g.game_id}-{m.player_id}", "published_at": published,
                         "game_id": g.game_id, "player_id": m.player_id, "status": "out", "source": "synthetic",
                         "url": "", "text": f"Player {m.player_id} listed out"})
            shift += OUT_MINUTE_VALUE * m.usual_min * (-1 if m.team_id == g.home_team_id else 1)
            react = published if react is None else min(react, published)
        p_post = clip(p_pre + shift)

        ts = pd.date_range(g.tip_time - pd.Timedelta(hours=7), g.tip_time + pd.Timedelta(hours=1), freq="1min")
        if react is None:
            p_home = np.full(len(ts), p_pre)
        else:
            start_move = react + MARKET_LAG
            frac = np.clip((ts - start_move) / MARKET_REACTION, 0, 1)
            p_home = p_pre + (p_post - p_pre) * np.asarray(frac, dtype=float)
        home_won = int(g.home_pts > g.away_pts)
        for team, p, won in ((g.home_team, p_home, home_won), (g.away_team, 1 - p_home, 1 - home_won)):
            ticker = f"SYN-{g.game_id}-{team}"
            markets.append({"venue": "synthetic", "market_ticker": ticker, "kind": "game", "game_id": g.game_id,
                            "team": team, "player_id": None, "line": None, "title": f"{team} to win"})
            prices.append(pd.DataFrame({"venue": "synthetic", "market_ticker": ticker, "ts": ts,
                                        "bid": np.round(np.clip(p - HALF_SPREAD, 0.01, 0.99), 2),
                                        "ask": np.round(np.clip(p + HALF_SPREAD, 0.01, 0.99), 2),
                                        "volume": 500.0}))
            settlements.append({"market_ticker": ticker, "settled_at": g.final_at, "outcome": won})

    return {"games": games, "player_games": player_games,
            "news": pd.DataFrame(news, columns=["news_id", "published_at", "game_id", "player_id", "status",
                                                "source", "url", "text"]),
            "markets": pd.DataFrame(markets), "prices": pd.concat(prices, ignore_index=True),
            "settlements": pd.DataFrame(settlements)}
