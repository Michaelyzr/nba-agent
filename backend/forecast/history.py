"""Leakage-safe lookups over finished games, shared by M2, M3, M4 and forecast/api.py.

Every query takes a tip-off time and only sees games that tipped off strictly
before it, so the same index serves training (built once on all games) and
prediction (built from an as-of view). Rotation rules match features.py.
"""
import numpy as np
import pandas as pd

ROTATION_MIN = 15.0                    # usual minutes to count as a rotation player
ROTATION_STALE = pd.Timedelta(days=14)
SEQ_COLS = ["min", "pts", "fga", "fta", "started", "home", "rest_days"]


def _ns(tip) -> np.datetime64:
    """A tip-off time as naive UTC nanoseconds, the dtype the index arrays use."""
    t = pd.Timestamp(tip)
    return np.datetime64((t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t).to_datetime64(), "ns")


class History:
    def __init__(self, player_games: pd.DataFrame, games: pd.DataFrame):
        done = games.dropna(subset=["home_pts"])
        g = done[["game_id", "tip_time", "home_team_id", "away_team_id", "home_pts", "away_pts"]]
        rows = player_games[player_games["min"] > 0].merge(g, on="game_id", how="inner")
        rows["home"] = (rows.team_id == rows.home_team_id).astype(float)
        started = rows["started"] if "started" in rows else pd.Series(np.nan, index=rows.index)
        rows["started"] = pd.to_numeric(started.astype(object).where(started.notna()), errors="coerce").fillna(0.5)
        rows = rows.sort_values(["player_id", "tip_time"])
        rows["rest_days"] = (rows.groupby("player_id").tip_time.diff().dt.total_seconds() / 86400).clip(upper=10).fillna(10)
        self.players = {}
        for pid, r in rows.groupby("player_id", sort=False):
            self.players[pid] = {"tip": r.tip_time.to_numpy("datetime64[ns]"), "team": r.team_id.to_numpy(),
                                 "seq": r[SEQ_COLS].to_numpy(float), "game": r.game_id.to_numpy()}
        sides = pd.concat([
            pd.DataFrame({"team": g.home_team_id, "game_id": g.game_id, "tip": g.tip_time,
                          "margin": g.home_pts - g.away_pts, "home": 1}),
            pd.DataFrame({"team": g.away_team_id, "game_id": g.game_id, "tip": g.tip_time,
                          "margin": g.away_pts - g.home_pts, "home": 0})]).sort_values(["team", "tip"])
        self.teams = {t: {"tip": s.tip.to_numpy("datetime64[ns]"), "game": s.game_id.to_numpy(),
                          "margin": s.margin.to_numpy(float)} for t, s in sides.groupby("team")}
        self.box = {gid: dict(zip(r.player_id, r["min"])) for gid, r in rows.groupby("game_id")}
        self.points = {gid: dict(zip(r.player_id, r.pts)) for gid, r in rows.groupby("game_id")}

    @staticmethod
    def _before(tips, tip) -> int:
        return int(np.searchsorted(tips, _ns(tip), side="left"))

    # ---------------- teams ----------------

    def team_games(self, team, tip, n=None) -> list:
        t = self.teams.get(team)
        if t is None:
            return []
        k = self._before(t["tip"], tip)
        return list(t["game"][max(0, k - n) if n else 0:k])

    def team_margins(self, team, tip, n=None) -> np.ndarray:
        t = self.teams.get(team)
        if t is None:
            return np.array([])
        k = self._before(t["tip"], tip)
        return t["margin"][max(0, k - n) if n else 0:k]

    def team_rest_days(self, team, tip) -> float:
        t = self.teams.get(team)
        k = 0 if t is None else self._before(t["tip"], tip)
        if k == 0:
            return 10.0
        gap = (_ns(tip) - t["tip"][k - 1]) / np.timedelta64(1, "D")
        return float(min(gap, 10.0))

    # ---------------- players ----------------

    def player_rows(self, pid, tip, n=None) -> np.ndarray:
        """Last n played games before tip, columns SEQ_COLS, oldest first."""
        p = self.players.get(pid)
        if p is None:
            return np.zeros((0, len(SEQ_COLS)))
        k = self._before(p["tip"], tip)
        return p["seq"][max(0, k - n) if n else 0:k]

    def last_played(self, pid, tip):
        """(team_id, tip) of the player's latest game before tip, or (None, None)."""
        p = self.players.get(pid)
        k = 0 if p is None else self._before(p["tip"], tip)
        return (None, None) if k == 0 else (p["team"][k - 1], pd.Timestamp(p["tip"][k - 1], tz="UTC"))

    def roster(self, team, tip, games=10) -> list:
        """Players who appeared for the team in its last `games` games and have not moved since."""
        seen = {p for gid in self.team_games(team, tip, games) for p in self.box.get(gid, {})}
        return sorted(p for p in seen if self.last_played(p, tip)[0] == team)

    def usual(self, pid, tip, n=10) -> tuple:
        """(mean minutes, mean points) over the player's last n played games."""
        r = self.player_rows(pid, tip, n)
        return (float(r[:, 0].mean()), float(r[:, 1].mean())) if len(r) else (0.0, 0.0)

    def rotation(self, team, tip) -> dict:
        """Rotation players likely available to play: {player_id: (usual minutes, usual points)}."""
        out = {}
        for p in self.roster(team, tip):
            last_team, last_tip = self.last_played(p, tip)
            minutes, points = self.usual(p, tip)
            if minutes >= ROTATION_MIN and pd.Timestamp(tip) - last_tip <= ROTATION_STALE:
                out[p] = (minutes, points)
        return out
