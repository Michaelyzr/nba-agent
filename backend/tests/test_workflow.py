from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import unittest

import httpx
from sqlalchemy import func, select

from app.db.session import get_db
from app.intelligence import service
from app.main import app
from app.models.intelligence import GamePrediction, GameSignal, PaperPosition
from app.models.market import MarketSnapshot
from app.schemas.intelligence import MarketRequest, PaperRequest
from tests.support import TestDB, seed


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = TestDB()
        self.game = seed(self.db)
        async def dependency():
            try: yield self.db
            except Exception:
                await self.db.rollback()
                raise
        app.dependency_overrides[get_db] = dependency
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        app.dependency_overrides.clear()
        self.db.close()

    async def quote(self):
        # Synthetic quote in isolated memory only, never sent to a real exchange.
        q = MarketSnapshot(game_id=self.game.id, provider="polymarket", market_type="game_winner",
            contract_id="123", selection="home", market_probability=.5, bid_price=.49, ask_price=.51,
            liquidity=2000, observed_at=datetime.now(timezone.utc))
        self.db.add(q)
        await self.db.commit()
        return q

    async def signal(self):
        await self.quote()
        result = await service.run_game(self.db, self.game.id)
        signal = next(s for s in result["signals"] if s["selection"] == "home")
        self.assertEqual(signal["status"], "paper_signal")
        return signal

    async def test_run_persists_prediction_and_both_audit_decisions(self):
        response = await self.client.post(f"/api/v1/intelligence/games/{self.game.id}/run")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(await self.db.scalar(select(func.count()).select_from(GamePrediction)), 1)
        self.assertEqual(await self.db.scalar(select(func.count()).select_from(GameSignal)), 2)
        self.assertEqual(response.json()["data"]["signals"][0]["status"], "blocked")
        listed = await self.client.get("/api/v1/intelligence/games")
        self.assertIsNotNone(listed.json()["data"][0]["prediction"])

    async def test_paper_open_duplicate_rejection_and_settlement(self):
        signal = await self.signal()
        response = await self.client.post("/api/v1/intelligence/paper", json={"signal_id": signal["id"], "stake": 25})
        self.assertEqual(response.status_code, 200, response.text)
        duplicate = await self.client.post("/api/v1/intelligence/paper", json={"signal_id": signal["id"], "stake": 25})
        self.assertEqual(duplicate.status_code, 400)
        self.game.status = "final"
        self.game.home_score, self.game.away_score = 110, 100
        self.game.tipoff_time = datetime.now(timezone.utc) - timedelta(days=1)
        await self.db.commit()
        settled = await service.settle_paper(self.db)
        self.assertEqual(settled["settled"], 1)
        p = await self.db.scalar(select(PaperPosition))
        self.assertAlmostEqual(p.pnl, 25 / .52 - 25)
        self.assertEqual((await service.settle_paper(self.db))["settled"], 0)

    async def test_stake_limit_rejected(self):
        signal = await self.signal()
        with self.assertRaisesRegex(ValueError, "单笔"):
            await service.open_paper(self.db, PaperRequest(signal_id=signal["id"], stake=101))
        self.assertEqual(await self.db.scalar(select(func.count()).select_from(PaperPosition)), 0)

    async def test_prediction_expiry_rejected(self):
        signal = await self.signal()
        prediction = await self.db.scalar(select(GamePrediction))
        prediction.as_of = datetime.now(timezone.utc) - timedelta(hours=1)
        await self.db.commit()
        with self.assertRaisesRegex(ValueError, "预测过期"):
            await service.open_paper(self.db, PaperRequest(signal_id=signal["id"], stake=25))

    async def test_total_exposure_rejected(self):
        signal = await self.signal()
        from uuid import UUID
        self.db.add(PaperPosition(game_id=self.game.id, signal_id=UUID(signal["id"]), selection="away",
                                  stake=500, entry_price=.5, shares=1000, status="open"))
        await self.db.commit()
        with self.assertRaisesRegex(ValueError, "总模拟敞口"):
            await service.open_paper(self.db, PaperRequest(signal_id=signal["id"], stake=25))

    async def test_order_size_cannot_exceed_best_ask_depth(self):
        signal = await self.signal()
        quote = await self.db.scalar(select(MarketSnapshot))
        quote.liquidity = 10
        await self.db.commit()
        from app.core.config import settings
        with patch.object(settings, "market_min_liquidity", 1):
            with self.assertRaisesRegex(ValueError, "档位深度"):
                await service.open_paper(self.db, PaperRequest(signal_id=signal["id"], stake=25))

    async def test_changed_quote_rejected(self):
        signal = await self.signal()
        await self.quote()
        with self.assertRaisesRegex(ValueError, "报价已变化"):
            await service.open_paper(self.db, PaperRequest(signal_id=signal["id"], stake=25))

    async def test_backtest_and_season_filter(self):
        result = await service.backtest(self.db, "2025-26")
        self.assertEqual(result["sample_size"], 30)
        self.assertEqual(result["skipped"], 10)
        future = await service.backtest(self.db, "2026-27")
        self.assertEqual(future["sample_size"], 0)

    async def test_manual_quotes_do_not_allow_paper_signals(self):
        request = MarketRequest(selection="home", bid_price=.49, ask_price=.51, liquidity=2000)
        await service.save_market(self.db, self.game.id, request)
        result = await service.run_game(self.db, self.game.id)
        self.assertEqual(result["signals"][0]["status"], "blocked")

    async def test_missing_and_invalid_game_ids(self):
        invalid = await self.client.post("/api/v1/intelligence/games/not-a-uuid/run")
        self.assertEqual(invalid.status_code, 422)
        missing = await self.client.post("/api/v1/intelligence/games/11111111-1111-1111-1111-111111111111/run")
        self.assertEqual(missing.status_code, 400)

    async def test_started_game_rejected_without_predictions(self):
        self.game.tipoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        await self.db.commit()
        with self.assertRaisesRegex(ValueError, "尚未开赛"):
            await service.run_game(self.db, self.game.id)
        self.assertEqual(await self.db.scalar(select(func.count()).select_from(GamePrediction)), 0)

    async def test_external_order_book_mock_is_persisted(self):
        async def book(token_id):
            return {"bid_price": .49, "ask_price": .51, "market_probability": .5,
                    "liquidity": 2000, "observed_at": datetime.now(timezone.utc), "raw_data": {"asset_id": token_id}}
        with patch("app.intelligence.service.fetch_book", book):
            response = await self.client.post(f"/api/v1/intelligence/games/{self.game.id}/markets", json={
                "selection": "home", "provider": "polymarket", "token_id": "123", "mapping_confirmed": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"]["provider"], "polymarket")
