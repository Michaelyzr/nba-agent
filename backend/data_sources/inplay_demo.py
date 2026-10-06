"""Explicitly synthetic in-game scenarios; historical tables supply rosters only."""
import pandas as pd

from data_sources.inplay_types import event_id, utc
from forecast.history import History
from replay import AsOf


def synthetic_game(tables, players, game):
    view = AsOf(tables, utc(game.tip_time), {})
    history = History(view.player_games(), view.games())
    home = history.rotation(game.home_team_id, game.tip_time)
    away = history.rotation(game.away_team_id, game.tip_time)
    if not home or not away:
        raise ValueError("synthetic in-play demo needs historical rotation players for both teams")
    hp, ap = max(home, key=lambda p: home[p][0]), max(away, key=lambda p: away[p][0])
    names = dict(zip(players.player_id, players.player_name))
    tip = utc(game.tip_time)
    scores, events = [], []
    scenarios = [(0, 1, 720, 0, 0, 0), (2400, 2, 180, 40, 42, 15),
                 (2405, 2, 180, 40, 42, 15), (2410, 2, 180, 40, 42, 15),
                 (2420, 2, 180, 40, 42, 15), (6000, 4, 240, 96, 94, 30),
                 (6005, 4, 240, 96, 94, 30), (7800, 4, 0, 113, 111, 34)]
    for offset, period, clock, hs, aws, played in scenarios:
        now = tip + pd.Timedelta(seconds=offset)
        scores.append({"game_id": str(game.game_id), "phase": "final" if offset == 7800 else "live",
                       "home_score": hs, "away_score": aws, "period": period, "clock_seconds": clock,
                       "observed_at": now, "updated_at": now, "source": "synthetic_inplay_demo",
                       "player_minutes": {str(hp): played, str(ap): played},
                       "player_teams": {str(hp): int(game.home_team_id), str(ap): int(game.away_team_id)},
                       "roster": [hp, ap]})
    for offset, pid, status, text in [(2405, hp, "left_injured", "left for the locker room with an ankle injury"),
                                     (2410, hp, "injury_out", "will not return"),
                                     (2420, hp, "returned", "returned to the game (synthetic correction)"),
                                     (6005, ap, "ejected", "was ejected")]:
        now = tip + pd.Timedelta(seconds=offset)
        events.append({"game_id": str(game.game_id), "event_id": event_id(game.game_id, offset, pid),
                       "player_id": int(pid), "status": status, "source": "synthetic_inplay_demo",
                       "published_at": now, "observed_at": now, "url": "", "text": str(names.get(pid, pid)) + " " + text})
    return pd.DataFrame(scores), pd.DataFrame(events)
