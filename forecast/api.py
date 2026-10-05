"""M5: one forecast entry point for agents, replay, evaluation and the demo.

    from forecast.api import Forecaster
    f = Forecaster.load()                       # trained models from models/
    f.win(view, game, out=[pid])                # home win probability, before/after "player X out"
    f.player(view, game, pid, "pts", lines=[22.5], out=[teammate])   # README section 5 format
    MarketAgent(forecaster=f)                   # P(yes) for every game-winner market

Every call reads only the as-of view it is given. The history index is
rebuilt once per Eastern date from games already final at that moment, so a
forecast never sees a game that finished after the decision.
"""
import math
import pickle
from pathlib import Path

import pandas as pd

from forecast.baselines import QUANTILES, _qcols, prob_at_least
from forecast.features import make_features
from forecast.history import History
from forecast.play import OUT_P_PLAY, roster_rows

MODELS = Path(__file__).resolve().parent.parent / "models"
P_MIN, P_MAX = 0.02, 0.98


def _load(path: Path):
    return pickle.loads(path.read_bytes()) if path.exists() else None


class Forecaster:
    _cache: dict = {}

    def __init__(self, win=None, play=None, points=None):
        self.win_model, self.play_model, self.points_model = win, play, points

    @classmethod
    def load(cls, folder: Path = MODELS):
        points = _load(folder / "m2" / "gru.pkl") or _load(folder / "m1" / "gbm.pkl")
        f = cls(_load(folder / "m4" / "win.pkl"), _load(folder / "m3" / "play.pkl"), points)
        if f.win_model is None:
            raise FileNotFoundError(f"no win model in {folder}/m4; run python -m forecast.win first")
        return f

    @property
    def name(self) -> str:
        return "+".join(m.name for m in (self.win_model, self.play_model, self.points_model) if m is not None)

    def history(self, view) -> History:
        day = view.now.tz_convert("America/New_York").date()
        key = (id(view._t), day)
        if key not in self._cache:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = History(view.player_games(), view.games())
        return self._cache[key]

    # ---------------- games ----------------

    def win(self, view, game, out=()) -> dict:
        from forecast.win import game_features
        h = self.history(view)
        f = game_features(h, game.home_team_id, game.away_team_id, game.tip_time, out)
        p = float(self.win_model.predict(pd.DataFrame([f]))[0])
        return {"game_id": game.game_id, "as_of": view.now.isoformat(), "model": self.win_model.name,
                "p_home": min(max(p, P_MIN), P_MAX), "missing": f["missing"], "overrides": {"out": list(out)}}

    def __call__(self, view, game, markets: pd.DataFrame, overrides: dict) -> dict:
        """MarketAgent interface: P(yes) for each game-winner market."""
        p_home = self.win(view, game, overrides.get("out", []))["p_home"]
        return {m.market_ticker: p_home if m.team == game.home_team else 1 - p_home
                for m in markets.itertuples() if m.kind == "game"}

    # ---------------- players ----------------

    def play(self, view, game, team_id, out=()) -> dict:
        """P(play) for each player on the team's recent roster; official 'out' wins over the model."""
        h = self.history(view)
        rows = roster_rows(h, team_id, game.tip_time)
        if rows.empty:
            return {}
        p = self.play_model.predict(rows) if self.play_model is not None else rows.played_rate10.to_numpy()
        return {pid: (OUT_P_PLAY if pid in set(out) else float(min(max(v, P_MIN), P_MAX)))
                for pid, v in zip(rows.player_id, p)}

    def players(self, view, game, team_id, out=(), lines: dict | None = None) -> list:
        """Section 5 forecasts (points and minutes) for the team's roster, given who is out."""
        h = self.history(view)
        p_play = self.play(view, game, team_id, out)
        ids = [p for p in p_play if p not in set(out)]
        if not ids or self.points_model is None:
            return []
        targets = pd.DataFrame({"game_id": game.game_id, "player_id": ids, "team_id": team_id,
                                "out": [[o for o in out if o != p] for p in ids]})
        rows = make_features(view.player_games(), view.games(), targets)
        pred = (self.points_model.predict(rows, h) if self.points_model.name == "gru"
                else self.points_model.predict(rows))
        result = []
        for i, row in rows.iterrows():
            for target in ("pts", "min"):
                q = [float(pred.loc[i, c]) for c in _qcols(target)]
                wanted = (lines or {}).get(row.player_id, []) if target == "pts" else []
                result.append({
                    "game_id": game.game_id, "player_id": int(row.player_id), "as_of": view.now.isoformat(),
                    "model": self.points_model.name, "target": target,
                    "quantiles": {f"p{int(p * 100)}": round(v, 2) for p, v in zip(QUANTILES, q)},
                    "p_over": {str(x): round(p_play[row.player_id] * prob_at_least(q, math.floor(x) + 1), 4)
                               for x in wanted},
                    "p_play": round(p_play[row.player_id], 4), "overrides": {"out": list(out)}})
        return result

    def player(self, view, game, player_id, target="pts", lines=(), out=()) -> dict | None:
        team_id, _ = self.history(view).last_played(player_id, game.tip_time)
        if team_id is None:
            return None
        if player_id in set(out):
            return {"game_id": game.game_id, "player_id": int(player_id), "as_of": view.now.isoformat(),
                    "model": self.name, "target": target, "quantiles": {}, "p_over": {str(x): 0.0 for x in lines},
                    "p_play": OUT_P_PLAY, "overrides": {"out": list(out)}}
        for f in self.players(view, game, team_id, out, {player_id: list(lines)}):
            if f["player_id"] == int(player_id) and f["target"] == target:
                return f
        return None
