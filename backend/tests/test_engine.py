from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from app.intelligence.data import normalize_results
from app.intelligence.engine import EloEngine, GameRecord, forecast, walk_forward
from app.intelligence.market import parse_book
from app.intelligence.metrics import summarize
from app.intelligence.risk import evaluate
from app.nba.games_service import NBAGamesService
from app.schemas.intelligence import MarketRequest, PaperRequest

NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)


def game(i, **kwargs):
    return GameRecord(str(i), "A", "B", NOW - timedelta(days=40-i), "2025-26",
                      **{"home_score": 110, "away_score": 100, "status": "final", **kwargs})


class EngineTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        result = EloEngine().predict(game(1))
        self.assertAlmostEqual(result["home_probability"] + result["away_probability"], 1)
        self.assertFalse(result["calibrated"])

    def test_future_result_cannot_change_forecast(self):
        target = replace(game(44), home_score=None, away_score=None, status="scheduled")
        history = [game(i) for i in range(20)]
        future = replace(game(45), home_score=1, away_score=200)
        a = forecast(history, target, NOW)
        b = forecast(history + [future], target, NOW)
        self.assertEqual(a, b)

    def test_result_not_available_before_six_hours(self):
        target = replace(game(44), status="scheduled")
        recent = replace(game(1), tipoff=NOW - timedelta(hours=5))
        result = forecast([recent], target, NOW)
        self.assertEqual(result["features"]["home_history_games"], 0)

    def test_preseason_and_missing_scores_excluded(self):
        engine = EloEngine()
        engine.update(replace(game(1), season_type="Pre Season"))
        engine.update(replace(game(2), home_score=None))
        engine.update(replace(game(3), home_score=100))
        self.assertEqual(engine.ratings["A"], 1500)

    def test_offseason_shrink(self):
        engine = EloEngine(retention=.5)
        engine.update(game(1))
        old = engine.ratings["A"]
        self.assertAlmostEqual(engine.rating("A", "2026-27"), 1500 + (old - 1500) * .5)

    def test_backtest_is_order_independent(self):
        games = [replace(game(i), home_score=110 if i % 2 else 90) for i in range(30)]
        self.assertEqual(walk_forward(games, min_history=5), walk_forward(list(reversed(games)), min_history=5))

    def test_target_outcome_never_enters_its_own_prediction(self):
        games = [game(i) for i in range(20)]
        a = walk_forward(games, min_history=3)
        games[-1] = replace(games[-1], home_score=1)
        b = walk_forward(games, min_history=3)
        self.assertEqual(a["rows"][-1]["home_probability"], b["rows"][-1]["home_probability"])

    def test_same_time_games_do_not_leak_results(self):
        games = [replace(game(i), tipoff=NOW) for i in range(10)]
        self.assertEqual(walk_forward(games, min_history=1)["sample_size"], 0)

    def test_metrics_and_auc_ties(self):
        result = summarize([{"home_probability": .5, "home_win": y} for y in [0, 1]], 2)
        self.assertAlmostEqual(result["metrics"]["brier_score"], .25)
        self.assertAlmostEqual(result["metrics"]["roc_auc"], .5)
        self.assertAlmostEqual(result["metrics"]["calibration_error"], 0)


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(min_team_history=10, market_max_age_seconds=900,
            market_min_liquidity=1000, market_max_spread=.05, signal_min_edge=.05, paper_cost_buffer=.01)
        self.game = SimpleNamespace(status="scheduled", season_type="Regular Season", tipoff_time=NOW + timedelta(days=1))
        self.quote = SimpleNamespace(provider="polymarket", observed_at=NOW, ask_price=.56, bid_price=.54, liquidity=2000)

    def decision(self):
        return evaluate(probability=.65, snapshot=self.quote, history_count=20, game=self.game, now=NOW, settings=self.settings)

    def test_uses_ask_and_cost_buffer(self):
        result = self.decision()
        self.assertEqual(result["status"], "paper_signal")
        self.assertAlmostEqual(result["effective_price"], .57)
        self.assertAlmostEqual(result["edge"], .08)

    def test_stale_quotes_rejected(self):
        self.quote.observed_at = NOW - timedelta(hours=1)
        self.assertEqual(self.decision()["status"], "blocked")

    def test_future_quotes_rejected(self):
        self.quote.observed_at = NOW + timedelta(seconds=1)
        self.assertEqual(self.decision()["status"], "blocked")

    def test_manual_quotes_are_research_only(self):
        self.quote.provider = "manual"
        self.assertEqual(self.decision()["status"], "blocked")

    def test_wide_spread_and_low_depth_rejected(self):
        self.quote.bid_price = .1
        self.quote.liquidity = 1
        self.assertEqual(len(self.decision()["reasons"]), 2)

    def test_started_game_rejected(self):
        self.game.tipoff_time = NOW
        self.assertEqual(self.decision()["status"], "blocked")

    def test_nonfinite_price_rejected(self):
        self.quote.ask_price = float('nan')
        self.assertEqual(self.decision()["status"], "blocked")


class DataTests(unittest.TestCase):
    def test_book_unsorted_levels_and_best_ask_depth(self):
        book = {"asset_id": "123", "timestamp": str(int(NOW.timestamp() * 1000)),
            "bids": [{"price": ".51", "size": "50"}, {"price": ".54", "size": "10"}],
            "asks": [{"price": ".70", "size": "10000"}, {"price": ".56", "size": "20"}]}
        result = parse_book(book, "123")
        self.assertAlmostEqual(result["ask_price"], .56)
        self.assertAlmostEqual(result["liquidity"], .56 * 20)
        with self.assertRaises(ValueError): parse_book(book, "456")

    def test_market_input_validation(self):
        with self.assertRaises(ValidationError): MarketRequest(selection="home", bid_price=.7, ask_price=.6)
        with self.assertRaises(ValidationError): MarketRequest(selection="home", provider="polymarket", token_id="123")
        with self.assertRaises(ValidationError): MarketRequest(selection="away", bid_price=float('nan'), ask_price=.6)
        with self.assertRaises(ValidationError): MarketRequest(selection="home", bid_price=.5, ask_price=.6, observed_at="2026-10-04T12:00:00")

    def test_malformed_book_is_a_readable_error(self):
        with self.assertRaises(ValueError):
            parse_book({"asset_id": "123", "bids": [{"price": ".5"}], "asks": []}, "123")

    def test_result_pairs_and_timezone(self):
        rows = [{"GAME_ID": "22500001", "TEAM_ID": 1, "MATCHUP": "A vs. B", "GAME_DATE": "2025-12-01", "PTS": 110},
                {"GAME_ID": "22500001", "TEAM_ID": 2, "MATCHUP": "B @ A", "GAME_DATE": "2025-12-01", "PTS": 100}]
        result = normalize_results(rows)
        self.assertEqual(result[0]["nba_game_id"], "0022500001")
        self.assertEqual(result[0]["tipoff_proxy"], datetime(2025, 12, 2, 5, tzinfo=timezone.utc))
        self.assertEqual(normalize_results(rows[:1]), [])

    def test_unknown_status_and_season_are_not_regular_season(self):
        self.assertEqual(NBAGamesService._normalize_status(None, None), "unknown")
        self.assertEqual(NBAGamesService._normalize_season_type({"gameId": "0012600001"}), "Pre Season")
        self.assertEqual(NBAGamesService._normalize_season_type({"gameId": "0022600001"}), "Regular Season")
        self.assertEqual(NBAGamesService._normalize_season_type({"gameId": "bad"}), "Unknown")
