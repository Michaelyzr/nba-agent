"""Deterministic Elo baseline. No LLM probabilities or future-score features."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class GameRecord:
    id: str
    home: str
    away: str
    tipoff: datetime
    season: str
    home_score: int | None = None
    away_score: int | None = None
    status: str = "scheduled"
    season_type: str | None = "Regular Season"

    @property
    def eligible_result(self):
        return (
            self.status == "final"
            and self.season_type in {"Regular Season", "Playoffs"}
            and self.home_score is not None and self.away_score is not None
            and self.home_score >= 0 and self.away_score >= 0
            and self.home_score != self.away_score
            and self.home != self.away
        )

    @property
    def available_at(self):
        # Existing schema has no completion time. Conservative proxy, disclosed
        # in outputs; replace with actual publication times before production.
        return self.tipoff + timedelta(hours=6)


class EloEngine:
    version = "elo-v1"

    def __init__(self, k=20.0, home_advantage=65.0, retention=0.75):
        self.k = k
        self.home_advantage = home_advantage
        self.retention = retention
        self.ratings = defaultdict(lambda: 1500.0)
        self.seasons = {}
        self.history = defaultdict(list)

    def rating(self, team, season):
        value = self.ratings[team]
        if team in self.seasons and self.seasons[team] != season:
            value = 1500 + (value - 1500) * self.retention
        return value

    @staticmethod
    def probability(difference):
        return 1 / (1 + 10 ** (-difference / 400))

    def predict(self, game):
        home_elo = self.rating(game.home, game.season)
        away_elo = self.rating(game.away, game.season)
        p = self.probability(home_elo - away_elo + self.home_advantage)
        neutral = self.probability(home_elo - away_elo)
        features = {"home_elo": round(home_elo, 2), "away_elo": round(away_elo, 2)}
        for label, team in [("home", game.home), ("away", game.away)]:
            past = self.history[team]
            features[f"{label}_history_games"] = len(past)
            features[f"{label}_last10_win_rate"] = (
                sum(won for _, won in past[-10:]) / len(past[-10:]) if past else None
            )
            features[f"{label}_rest_days"] = (
                max(0, (game.tipoff.date() - past[-1][0].date()).days - 1) if past else None
            )
        return {
            "home_probability": p, "away_probability": 1 - p,
            "model_version": self.version, "calibrated": False,
            "features": features,
            "factors": [
                {"name": "球队 Elo 实力差", "value": neutral - 0.5},
                {"name": "主场基线", "value": p - neutral},
            ],
            "limitations": [
                "Elo 参数尚未训练、概率尚未校准；仅用于研究与模拟。",
                "赛果可用时间暂按开赛后 6 小时估计。",
                "近期胜率与休息天数仅展示；伤病、阵容和球员影响尚未纳入概率。",
            ],
        }

    def update(self, game):
        if not game.eligible_result:
            return
        home = self.rating(game.home, game.season)
        away = self.rating(game.away, game.season)
        expected = self.probability(home - away + self.home_advantage)
        won = int(game.home_score > game.away_score)
        delta = self.k * (won - expected)
        self.ratings[game.home] = home + delta
        self.ratings[game.away] = away - delta
        self.seasons[game.home] = self.seasons[game.away] = game.season
        self.history[game.home].append((game.tipoff, won))
        self.history[game.away].append((game.tipoff, 1 - won))


def forecast(history, target, as_of, **parameters):
    engine = EloEngine(**parameters)
    for game in sorted(history, key=lambda g: (g.available_at, g.id)):
        if game.id != target.id and game.available_at <= as_of and game.tipoff < target.tipoff:
            engine.update(game)
    return engine.predict(target)


def walk_forward(games, *, min_history=10, **parameters):
    """Predict at each tipoff; consume scores only when their result is available."""
    games = sorted((g for g in games if g.eligible_result), key=lambda g: (g.tipoff, g.id))
    outcomes = sorted(games, key=lambda g: (g.available_at, g.id))
    engine = EloEngine(**parameters)
    cursor = 0
    rows = []
    for game in games:
        while cursor < len(outcomes) and outcomes[cursor].available_at <= game.tipoff:
            engine.update(outcomes[cursor])
            cursor += 1
        prediction = engine.predict(game)
        f = prediction["features"]
        if min(f["home_history_games"], f["away_history_games"]) < min_history:
            continue
        rows.append({
            "game_id": game.id, "tipoff_time": game.tipoff.isoformat(),
            "home_probability": prediction["home_probability"],
            "home_win": int(game.home_score > game.away_score),
        })
    from app.intelligence.metrics import summarize
    return summarize(rows, len(games))
