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


def synthetic_full_game(tables, players, game):
    """Minute samples across Q1–Q4, plus explicit incident and boundary samples.

    Scores interpolate synthetic quarter totals, not actual NBA play-by-play.
    Incident observations retain the same score/clock as their pre-event sample.
    """
    scores, original = synthetic_game(tables, players, game)
    base = scores.iloc[0].to_dict()
    hp = int(original.iloc[0].player_id)
    ap = int(original.iloc[-1].player_id)
    names = dict(zip(players.player_id, players.player_name))
    totals = [(0, 0), (26, 24), (52, 50), (79, 80), (113, 111)]
    incidents = {16: [(1, hp, "left_injured", "left for the locker room with an ankle injury", "因伤离场")],
                 18: [(1, hp, "injury_out", "will not return", "确认无法回归")],
                 27: [(1, hp, "returned", "returned to the game (synthetic correction)", "确认回归／更正")],
                 43: [(1, ap, "ejected", "was ejected", "对手被驱逐"),
                      (3, ap, "returned", "has returned to the game", "次级回归误报被保留")]}
    tip = utc(game.tip_time)
    states, events = [], []

    def add(period, minute_in_quarter, delay=0, label=None):
        elapsed = (period - 1) * 12 + minute_in_quarter
        clock = 720 - minute_in_quarter * 60
        wall = elapsed * 95 + (period - 1) * 150 + (900 if period >= 3 else 0) + delay
        now = tip + pd.Timedelta(seconds=wall)
        previous, end = totals[period - 1], totals[period]
        hs, aws = [round(a + (b - a) * minute_in_quarter / 12) for a, b in zip(previous, end)]
        home_active = elapsed - max(0, min(elapsed, 27) - 16)
        state = {**base, "period": period, "clock_seconds": clock,
                 "phase": "final" if elapsed == 48 else "live", "home_score": hs, "away_score": aws,
                 "observed_at": now, "updated_at": now,
                 "player_minutes": {str(hp): round(home_active * .75, 3), str(ap): round(min(elapsed, 43) * .7, 3)},
                 "demo_label": label or ("终场停止" if elapsed == 48 else "赛中独立基准" if elapsed == 0 else f"Q{period} 比分与时钟更新")}
        states.append(state)
        return now

    for period in range(1, 5):
        for minute in range(13):
            elapsed = (period - 1) * 12 + minute
            if period == 4 and minute == 12:
                add(4, 12 - 1 / 60, label="最后一秒预测")
            add(period, minute)
            for delay, pid, status, text, label in incidents.get(elapsed, []):
                now = add(period, minute, delay, label)
                events.append({"game_id": str(game.game_id), "event_id": event_id(game.game_id, elapsed, delay, pid),
                               "player_id": pid, "status": status, "source": "synthetic_inplay_demo",
                               "published_at": now, "observed_at": now, "url": "",
                               "text": str(names[pid]) + " " + text,
                               "demo_conflict": label == "次级回归误报被保留"})
    return pd.DataFrame(states), pd.DataFrame(events)
