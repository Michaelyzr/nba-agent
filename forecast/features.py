"""Shared, leakage-safe features for one player in one game (used by M1b, M2 and M3).

Every feature for game g uses only that player's and team's games that tipped
off before g. The same code builds training rows and prediction rows:
prediction targets are appended with empty outcomes, so the rolling windows
are computed identically.

Training rows take "teammates out" from who actually missed the game (known
only afterwards); prediction rows take it from the `out` list the caller
passes (from the injury report). Report this mismatch as a limitation.
"""
import numpy as np
import pandas as pd

ROLL_COLS = ["min", "pts", "fga", "fta", "started"]
WINDOWS = (5, 10)
ROTATION_MIN = 15.0              # usual minutes to count as a rotation player
ROTATION_STALE = pd.Timedelta(days=14)
TARGETS = ("pts", "min")

FEATURES = (
    [f"{c}_avg{w}" for w in WINDOWS for c in ROLL_COLS]
    + ["pts_per_min_avg10", "games_prior", "rest_days", "back_to_back", "home",
       "teammates_out_min", "opp_allowed_avg10"]
)


def _rows(player_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "tip_time", "home_team_id", "away_team_id"]]
    rows = player_games.merge(g, on="game_id", how="inner")
    rows["home"] = (rows.team_id == rows.home_team_id).astype(int)
    rows["opp_team_id"] = np.where(rows.home == 1, rows.away_team_id, rows.home_team_id)
    if "started" not in rows or rows["started"].isna().all():
        rows["started"] = np.nan
    rows["started"] = rows["started"].astype(float)
    return rows.drop(columns=["home_team_id", "away_team_id"])


def _history(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.sort_values(["player_id", "tip_time"]).reset_index(drop=True)
    by = rows.groupby("player_id", sort=False)
    for c in ROLL_COLS:
        prior = by[c].shift(1)
        for w in WINDOWS:
            rows[f"{c}_avg{w}"] = prior.groupby(rows.player_id).rolling(w, min_periods=1).mean().reset_index(
                level=0, drop=True)
    pts10 = by["pts"].shift(1).groupby(rows.player_id).rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    min10 = by["min"].shift(1).groupby(rows.player_id).rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    rows["pts_per_min_avg10"] = pts10 / min10.where(min10 > 0)
    rows["games_prior"] = by.cumcount()
    rest = (rows.tip_time - by.tip_time.shift(1)).dt.total_seconds() / 86400
    rows["rest_days"] = rest.clip(upper=10)
    rows["back_to_back"] = (rest < 1.5).astype(int)
    return rows


def _opponent_defence(games: pd.DataFrame) -> pd.DataFrame:
    """Points each team allowed over its previous 10 games, keyed by (game_id, team)."""
    done = games.dropna(subset=["home_pts"])
    sides = pd.concat([
        pd.DataFrame({"game_id": done.game_id, "tip_time": done.tip_time, "team": done.home_team_id,
                      "allowed": done.away_pts}),
        pd.DataFrame({"game_id": done.game_id, "tip_time": done.tip_time, "team": done.away_team_id,
                      "allowed": done.home_pts})]).sort_values(["team", "tip_time"])
    sides["opp_allowed_avg10"] = sides.groupby("team").allowed.transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    upcoming = games[games.home_pts.isna()]
    last = sides.groupby("team").tail(10).groupby("team").allowed.mean()
    future = pd.concat([
        pd.DataFrame({"game_id": upcoming.game_id, "team": upcoming.home_team_id}),
        pd.DataFrame({"game_id": upcoming.game_id, "team": upcoming.away_team_id})])
    future["opp_allowed_avg10"] = future.team.map(last)
    return pd.concat([sides[["game_id", "team", "opp_allowed_avg10"]], future], ignore_index=True)


def _actual_absences(rows: pd.DataFrame) -> pd.Series:
    """Usual minutes of rotation players who missed each team-game (training rows).

    A rotation player averaged ROTATION_MIN+ over his last 10 games for this team
    and played for it within ROTATION_STALE, so traded players drop out.
    """
    out_min = {}
    for team_id, team_rows in rows[rows["min"].notna()].groupby("team_id"):
        recent, last_tip = {}, {}
        for (game_id, tip), game_rows in team_rows.groupby(["game_id", "tip_time"], sort=True):
            present = set(game_rows.player_id)
            out_min[(game_id, team_id)] = float(sum(
                np.mean(m) for p, m in recent.items()
                if p not in present and np.mean(m) >= ROTATION_MIN and tip - last_tip[p] <= ROTATION_STALE))
            for p, m in zip(game_rows.player_id, game_rows["min"]):
                recent[p] = (recent.get(p, []) + [m])[-10:]
                last_tip[p] = tip
    return pd.Series(out_min, dtype=float)


def _usual_minutes(rows: pd.DataFrame, player_id, before: pd.Timestamp) -> float:
    hist = rows[(rows.player_id == player_id) & (rows.tip_time < before) & rows["min"].notna()]
    return float(hist["min"].tail(10).mean()) if len(hist) else 0.0


def make_features(player_games: pd.DataFrame, games: pd.DataFrame, targets: pd.DataFrame | None = None):
    """Feature rows for every played game (training), or only for `targets` (prediction).

    targets: columns game_id, player_id, team_id and optionally out (list of player_ids
    expected to miss the game). Their outcomes are unknown and stay NaN.
    """
    played = player_games[player_games["min"] > 0]
    rows = _rows(played, games)
    if targets is not None:
        t = targets.copy()
        t["out"] = t.get("out", pd.Series([[]] * len(t), index=t.index))
        t_rows = _rows(t.drop(columns=["out"]).assign(**{c: np.nan for c in ["min", "pts", "fga", "fta"]}), games)
        t_rows["_target"] = True
        rows = pd.concat([rows.assign(_target=False), t_rows], ignore_index=True)
    rows = _history(rows)

    if targets is None:
        absences = _actual_absences(rows)
        rows["teammates_out_min"] = [absences.get((g, t), 0.0) for g, t in zip(rows.game_id, rows.team_id)]
    else:
        history = rows[~rows._target]
        out_by_key = {(r.game_id, r.player_id): list(r.out) for r in t.itertuples()}
        rows = rows[rows._target].drop(columns="_target")
        rows["teammates_out_min"] = [
            sum(_usual_minutes(history, p, tip) for p in out_by_key.get((g, pid), []))
            for g, pid, tip in zip(rows.game_id, rows.player_id, rows.tip_time)]

    defence = _opponent_defence(games).rename(columns={"team": "opp_team_id"})
    rows = rows.merge(defence, on=["game_id", "opp_team_id"], how="left")
    return rows.sort_values(["tip_time", "game_id", "player_id"]).reset_index(drop=True)
