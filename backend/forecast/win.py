"""M4: home win probability from team ratings adjusted for who is out.

    python -m forecast.win --source frozen --split 2026-02-01

Each team's rating is its recent point margin (exponentially weighted, shrunk
toward zero). Missing rotation players subtract their usual minutes and
points. Training takes "out" from who actually missed the game; prediction
takes it from the injury news the caller passes, so the same features give a
before-the-news and an after-the-news probability. Logistic regression on
home-minus-away differences; the intercept is home advantage.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from forecast.history import History

HALF_LIFE = 10.0                 # games
SHRINK = 3.0                     # pseudo-games at zero margin
WIN_FEATURES = ["rating_diff", "margin10_diff", "rest_diff", "b2b_diff", "missing_min_diff", "missing_pts_diff"]


def team_side(h: History, team, tip, out: set, availability=None) -> dict:
    m = h.team_margins(team, tip, 40)
    w = 0.5 ** (np.arange(len(m))[::-1] / HALF_LIFE)
    rest = h.team_rest_days(team, tip)
    rotation = h.rotation(team, tip)
    # availability maps player ids to expected lost share (0 = full, 1 = absent).
    lost = availability or {}
    missing = [(v, 1.0 if p in out else lost.get(p, 0.0)) for p, v in rotation.items()]
    return {"rating": float((w * m).sum() / (w.sum() + SHRINK)), "margin10": float(m[-10:].mean()) if len(m) else 0.0,
            "rest": rest, "b2b": float(rest < 1.5), "missing_min": float(sum(v[0] * share for v, share in missing)),
            "missing_pts": float(sum(v[1] * share for v, share in missing)), "rotation": rotation}


def game_features(h: History, home_team_id, away_team_id, tip, out=(), availability=None) -> dict:
    out = set(out)
    home = team_side(h, home_team_id, tip, out, availability)
    away = team_side(h, away_team_id, tip, out, availability)
    row = {f"{k}_diff": home[k] - away[k] for k in ("rating", "margin10", "rest", "b2b", "missing_min", "missing_pts")}
    lost = availability or {}
    row["missing"] = {"home": [p for p in home["rotation"] if p in out or lost.get(p, 0) > 0],
                      "away": [p for p in away["rotation"] if p in out or lost.get(p, 0) > 0]}
    return row


def training_rows(h: History, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in games.dropna(subset=["home_pts"]).itertuples():
        played = h.box.get(g.game_id, {})
        out = {p for team in (g.home_team_id, g.away_team_id) for p in h.rotation(team, g.tip_time) if p not in played}
        f = game_features(h, g.home_team_id, g.away_team_id, g.tip_time, out)
        f.pop("missing")
        rows.append({**f, "game_id": g.game_id, "tip_time": g.tip_time, "home_win": float(g.home_pts > g.away_pts)})
    return pd.DataFrame(rows)


class WinModel:
    name = "win-logit"

    def fit(self, rows: pd.DataFrame):
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(rows[WIN_FEATURES], rows.home_win)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(rows[WIN_FEATURES])[:, 1]

    def p_home(self, h: History, game, out=()) -> float:
        f = game_features(h, game.home_team_id, game.away_team_id, game.tip_time, out)
        return float(self.predict(pd.DataFrame([f]))[0])

    def coefficients(self) -> dict:
        return {"home_intercept": float(self.model.intercept_[0]),
                **dict(zip(WIN_FEATURES, map(float, self.model.coef_[0])))}


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def evaluate(model: WinModel, test: pd.DataFrame) -> pd.DataFrame:
    y = test.home_win.to_numpy()
    no_injury = test.assign(missing_min_diff=0.0, missing_pts_diff=0.0)
    preds = {"model": model.predict(test), "model_ignoring_absences": model.predict(no_injury),
             "home_rate": np.full(len(y), 0.55)}
    out = [{"predictor": k, "brier": brier(p, y), "accuracy": float(np.mean((p > 0.5) == y)), "games": len(y)}
           for k, p in preds.items()]
    return pd.DataFrame(out)


def main():
    from forecast.baselines import MODEL_DIR, load_source
    from forecast.win import WinModel, training_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--split", default="2026-02-01")
    args = ap.parse_args()
    player_games, games = load_source(args.source)
    rows = training_rows(History(player_games, games), games)
    split = pd.Timestamp(args.split, tz="UTC")
    train, test = rows[rows.tip_time < split], rows[rows.tip_time >= split]
    model = WinModel().fit(train)
    print(f"{len(train)} training games, {len(test)} test games")
    print(evaluate(model, test).round(4).to_string(index=False))
    print({k: round(v, 4) for k, v in model.coefficients().items()})
    path = MODEL_DIR.parent / "m4" / "win.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
