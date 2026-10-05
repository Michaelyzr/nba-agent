"""M3: will a rostered player play tonight, from his availability history alone.

    python -m forecast.play --source frozen --split 2026-02-01

One row per (team game, player on the recent roster). The label is whether he
logged minutes. News is not a feature: forecast/api.py sets p_play to
OUT_P_PLAY when an official status says out, and uses this model otherwise, so
the model covers the hours before the injury report. Held-out Brier score is
reported against two simple rules.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast.history import History

PLAY_FEATURES = ["played_rate3", "played_rate10", "missed_last", "miss_streak", "min_avg5", "min_avg10",
                 "games_prior", "team_rest_days", "days_since_played"]
OUT_P_PLAY = 0.02


def roster_rows(h: History, team, tip) -> pd.DataFrame:
    """Availability features for every player on the team's recent roster, as of tip."""
    recent = h.team_games(team, tip, 10)
    rows = []
    for p in h.roster(team, tip):
        played = [gid for gid in recent if p in h.box.get(gid, {})]
        flags = [gid in played for gid in recent]
        streak = 0
        for f in reversed(flags):
            if f:
                break
            streak += 1
        r = h.player_rows(p, tip)
        _, last_tip = h.last_played(p, tip)
        rows.append({"player_id": p, "team_id": team,
                     "played_rate3": float(np.mean(flags[-3:])), "played_rate10": float(np.mean(flags)),
                     "missed_last": float(not flags[-1]), "miss_streak": float(min(streak, 10)),
                     "min_avg5": float(r[-5:, 0].mean()), "min_avg10": float(r[-10:, 0].mean()),
                     "games_prior": float(min(len(r), 82)), "team_rest_days": h.team_rest_days(team, tip),
                     "days_since_played": min((pd.Timestamp(tip) - last_tip).total_seconds() / 86400, 30.0)})
    return pd.DataFrame(rows, columns=["player_id", "team_id"] + PLAY_FEATURES)


def training_rows(h: History, games: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for g in games.dropna(subset=["home_pts"]).itertuples():
        played = h.box.get(g.game_id, {})
        for team in (g.home_team_id, g.away_team_id):
            rows = roster_rows(h, team, g.tip_time)
            if len(rows):
                parts.append(rows.assign(game_id=g.game_id, tip_time=g.tip_time,
                                         played=rows.player_id.map(lambda p: float(p in played))))
    return pd.concat(parts, ignore_index=True)


class PlayModel:
    name = "play-logit"

    def fit(self, rows: pd.DataFrame):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        self.model.fit(rows[PLAY_FEATURES], rows.played)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if rows.empty:
            return np.array([])
        return self.model.predict_proba(rows[PLAY_FEATURES])[:, 1]


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def evaluate(model: PlayModel, test: pd.DataFrame) -> pd.DataFrame:
    y = test.played.to_numpy()
    rules = {"model": model.predict(test), "played_rate10": test.played_rate10.clip(0.02, 0.98),
             "played_last_game": np.where(test.missed_last == 1, 0.3, 0.95)}
    return pd.DataFrame([{"predictor": k, "brier": brier(p, y), "rows": len(y), "base_rate": float(y.mean())}
                         for k, p in rules.items()])


def main():
    from forecast.baselines import MODEL_DIR, load_source
    from forecast.play import PlayModel, training_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--split", default="2026-02-01")
    args = ap.parse_args()
    player_games, games = load_source(args.source)
    rows = training_rows(History(player_games, games), games)
    split = pd.Timestamp(args.split, tz="UTC")
    train, test = rows[rows.tip_time < split], rows[rows.tip_time >= split]
    model = PlayModel().fit(train)
    print(f"{len(train)} training rows, {len(test)} test rows")
    print(evaluate(model, test).round(4).to_string(index=False))
    path = MODEL_DIR.parent / "m3" / "play.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
