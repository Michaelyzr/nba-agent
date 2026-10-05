from datetime import datetime, timezone
from math import sqrt
from statistics import mean
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.forecast import Forecast
from app.models.game import Game
from app.models.player import Player
from app.models.stats import PlayerGameStat


class PredictionService:
    """
    NBA Baseline Prediction Service。

    当前负责：
    1. 找到球员下一场比赛
    2. 读取最近 N 场比赛
    3. 构造历史特征
    4. 预测 minutes
    5. 预测 points
    6. 写入 forecasts 表

    后续：
    - Rolling Average
    - EWMA
    - Gradient Boosting
    - GRU
    - Injury Adjustment
    - News Adjustment
    """

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _std(
        values: list[float],
    ) -> float:
        """
        简单 population standard deviation。
        """

        if len(values) <= 1:
            return 0.0

        avg = mean(values)

        variance = sum(
            (value - avg) ** 2
            for value in values
        ) / len(values)

        return sqrt(variance)

    @staticmethod
    def _weighted_average(
        values: list[float],
    ) -> float:
        """
        越新的比赛权重越高。

        假设输入顺序：
        oldest -> newest

        例如 5 场：
        权重 = 1,2,3,4,5
        """

        if not values:
            return 0.0

        weights = list(
            range(1, len(values) + 1)
        )

        weighted_sum = sum(
            value * weight
            for value, weight in zip(
                values,
                weights,
            )
        )

        return weighted_sum / sum(weights)

    # ========================================================
    # Player
    # ========================================================

    async def get_player_by_nba_id(
        self,
        session: AsyncSession,
        *,
        nba_player_id: int,
    ) -> Player:
        """
        根据 NBA player ID 获取数据库球员。
        """

        result = await session.execute(
            select(Player).where(
                Player.nba_player_id
                == nba_player_id
            )
        )

        player = result.scalar_one_or_none()

        if player is None:
            raise ValueError(
                f"Player {nba_player_id} "
                "does not exist in database."
            )

        return player

    # ========================================================
    # Next Game
    # ========================================================

    async def get_next_game(
        self,
        session: AsyncSession,
        *,
        player: Player,
    ) -> Game:
        """
        找当前球员球队的下一场 scheduled game。
        """

        if player.current_team_id is None:
            raise ValueError(
                f"{player.full_name} "
                "does not have a current team."
            )

        now = datetime.now(
            timezone.utc
        )

        result = await session.execute(
            select(Game)
            .where(
                Game.tipoff_time > now,
                Game.status == "scheduled",
                or_(
                    Game.home_team_id
                    == player.current_team_id,
                    Game.away_team_id
                    == player.current_team_id,
                ),
            )
            .order_by(
                Game.tipoff_time.asc()
            )
            .limit(1)
        )

        game = result.scalar_one_or_none()

        if game is None:
            raise ValueError(
                f"No upcoming game found for "
                f"{player.full_name}."
            )

        return game

    # ========================================================
    # Historical Stats
    # ========================================================

    async def get_recent_stats(
        self,
        session: AsyncSession,
        *,
        player_id: UUID,
        before_game: Game,
        limit: int = 10,
    ) -> list[PlayerGameStat]:
        """
        获取目标比赛之前最近 N 场比赛。

        注意：
        必须限定 Game.tipoff_time < target game，
        防止 future leakage。
        """

        result = await session.execute(
            select(PlayerGameStat)
            .join(
                Game,
                PlayerGameStat.game_id
                == Game.id,
            )
            .where(
                PlayerGameStat.player_id
                == player_id,
                Game.tipoff_time
                < before_game.tipoff_time,
                Game.status == "final",
            )
            .order_by(
                Game.tipoff_time.desc()
            )
            .limit(limit)
        )

        # SQL 查询得到 newest -> oldest
        stats = list(
            result.scalars().all()
        )

        # 预测函数希望 oldest -> newest
        stats.reverse()

        return stats

    # ========================================================
    # Build Features
    # ========================================================

    def build_features(
        self,
        stats: list[PlayerGameStat],
    ) -> dict:
        """
        从历史比赛生成 baseline features。
        """

        valid_minutes = [
            float(stat.minutes)
            for stat in stats
            if stat.minutes is not None
        ]

        valid_points = [
            float(stat.points)
            for stat in stats
            if stat.points is not None
        ]

        valid_rebounds = [
            float(stat.rebounds)
            for stat in stats
            if stat.rebounds is not None
        ]

        valid_assists = [
            float(stat.assists)
            for stat in stats
            if stat.assists is not None
        ]

        if not valid_minutes:
            raise ValueError(
                "Player has no usable minutes history."
            )

        if not valid_points:
            raise ValueError(
                "Player has no usable points history."
            )

        return {
            "games_used": len(stats),

            "minutes": {
                "mean": mean(valid_minutes),
                "weighted_mean": (
                    self._weighted_average(
                        valid_minutes
                    )
                ),
                "std": self._std(
                    valid_minutes
                ),
                "last": valid_minutes[-1],
            },

            "points": {
                "mean": mean(valid_points),
                "weighted_mean": (
                    self._weighted_average(
                        valid_points
                    )
                ),
                "std": self._std(
                    valid_points
                ),
                "last": valid_points[-1],
            },

            "rebounds_mean": (
                mean(valid_rebounds)
                if valid_rebounds
                else None
            ),

            "assists_mean": (
                mean(valid_assists)
                if valid_assists
                else None
            ),
        }

    # ========================================================
    # Forecast
    # ========================================================

    async def create_player_forecast(
        self,
        session: AsyncSession,
        *,
        nba_player_id: int,
        history_games: int = 10,
    ) -> dict:
        """
        为球员下一场比赛创建：

        1. minutes forecast
        2. points forecast
        """

        # ----------------------------------------------------
        # Player
        # ----------------------------------------------------

        player = await self.get_player_by_nba_id(
            session,
            nba_player_id=nba_player_id,
        )

        # ----------------------------------------------------
        # Next Game
        # ----------------------------------------------------

        next_game = await self.get_next_game(
            session,
            player=player,
        )

        # ----------------------------------------------------
        # Historical Stats
        # ----------------------------------------------------

        stats = await self.get_recent_stats(
            session,
            player_id=player.id,
            before_game=next_game,
            limit=history_games,
        )

        if len(stats) < 3:
            raise ValueError(
                f"Not enough historical games for "
                f"{player.full_name}. "
                f"Found {len(stats)}, need at least 3."
            )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = self.build_features(
            stats
        )

        # ----------------------------------------------------
        # Baseline Predictions
        # ----------------------------------------------------

        predicted_minutes = (
            features["minutes"][
                "weighted_mean"
            ]
        )

        predicted_points = (
            features["points"][
                "weighted_mean"
            ]
        )

        minutes_std = (
            features["minutes"]["std"]
        )

        points_std = (
            features["points"]["std"]
        )

        # ----------------------------------------------------
        # Confidence
        #
        # 目前先根据历史样本数量给基础 confidence。
        # 后续会替换为 calibration-based confidence。
        # ----------------------------------------------------

        confidence = min(
            1.0,
            len(stats) / history_games,
        )

        # ----------------------------------------------------
        # Minutes Forecast
        # ----------------------------------------------------

        minutes_forecast = Forecast(
            game_id=next_game.id,
            player_id=player.id,

            forecast_type="minutes",

            predicted_value=(
                predicted_minutes
            ),

            distribution={
                "mean": predicted_minutes,
                "std": minutes_std,
                "lower": max(
                    0.0,
                    predicted_minutes
                    - minutes_std,
                ),
                "upper": min(
                    48.0,
                    predicted_minutes
                    + minutes_std,
                ),
            },

            confidence=confidence,

            model_name=(
                "rolling_weighted_baseline"
            ),

            model_version="1.0",

            features=features,

            explanation=(
                f"Minutes forecast based on "
                f"{len(stats)} recent games "
                f"using recency-weighted average."
            ),
        )

        # ----------------------------------------------------
        # Points Forecast
        # ----------------------------------------------------

        points_forecast = Forecast(
            game_id=next_game.id,
            player_id=player.id,

            forecast_type="points",

            predicted_value=(
                predicted_points
            ),

            distribution={
                "mean": predicted_points,
                "std": points_std,
                "lower": max(
                    0.0,
                    predicted_points
                    - points_std,
                ),
                "upper": (
                    predicted_points
                    + points_std
                ),
            },

            confidence=confidence,

            model_name=(
                "rolling_weighted_baseline"
            ),

            model_version="1.0",

            features=features,

            explanation=(
                f"Points forecast based on "
                f"{len(stats)} recent games "
                f"using recency-weighted average."
            ),
        )

        session.add(
            minutes_forecast
        )

        session.add(
            points_forecast
        )

        await session.commit()

        await session.refresh(
            minutes_forecast
        )

        await session.refresh(
            points_forecast
        )

        return {
            "player": {
                "id": str(player.id),
                "nba_player_id": (
                    player.nba_player_id
                ),
                "name": (
                    player.full_name
                ),
            },

            "game": {
                "id": str(next_game.id),
                "nba_game_id": (
                    next_game.nba_game_id
                ),
                "tipoff_time": (
                    next_game
                    .tipoff_time
                    .isoformat()
                ),
            },

            "history_games": len(stats),

            "forecast": {
                "minutes": {
                    "forecast_id": str(
                        minutes_forecast.id
                    ),
                    "value": round(
                        predicted_minutes,
                        2,
                    ),
                    "std": round(
                        minutes_std,
                        2,
                    ),
                },

                "points": {
                    "forecast_id": str(
                        points_forecast.id
                    ),
                    "value": round(
                        predicted_points,
                        2,
                    ),
                    "std": round(
                        points_std,
                        2,
                    ),
                },

                "confidence": round(
                    confidence,
                    3,
                ),
            },

            "model": {
                "name": (
                    "rolling_weighted_baseline"
                ),
                "version": "1.0",
            },
        }


prediction_service = PredictionService()