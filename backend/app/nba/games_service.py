import asyncio
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nba_api.stats.endpoints.scheduleleaguev2 import ScheduleLeagueV2

from app.models.game import Game
from app.models.team import Team


class NBAGamesService:
    """
    NBA 比赛数据同步服务。

    负责：
    1. 从 NBA API 获取指定赛季完整赛程
    2. 匹配主队 / 客队
    3. 写入 PostgreSQL games 表
    4. 已存在比赛则更新状态和比分
    5. 支持不同 NBA 赛季

    例如：
    - 2025-26
    - 2026-27
    """

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:
        """
        安全转换为 int。

        None / NaN / 空数据 -> None
        """

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_game_id(
        value: Any,
    ) -> str:
        """
        NBA GAME_ID 通常类似：

        0012600009

        Pandas 有时可能把它处理为数字，
        导致前导 0 丢失。

        这里统一恢复成 10 位字符串。
        """

        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass

        game_id = str(value).strip()

        # 例如：
        # 12600009.0
        # ->
        # 12600009
        if game_id.endswith(".0"):
            game_id = game_id[:-2]

        # 补前导 0
        if game_id.isdigit():
            game_id = game_id.zfill(10)

        return game_id

    @classmethod
    def _normalize_status(
        cls,
        game_status: Any,
        status_text: Any,
    ) -> str:
        """
        NBA gameStatus：

        1 = scheduled
        2 = live
        3 = final

        转换成系统统一状态：
        - scheduled
        - live
        - final
        - postponed
        """

        text = str(
            status_text or ""
        ).lower()

        # ----------------------------------------------------
        # Postponed
        # ----------------------------------------------------

        if "postpon" in text:
            return "postponed"

        # ----------------------------------------------------
        # NBA status code
        # ----------------------------------------------------

        status_code = cls._safe_int(
            game_status
        )

        if status_code == 1:
            return "scheduled"

        if status_code == 2:
            return "live"

        if status_code == 3:
            return "final"

        return "unknown"

    @staticmethod
    def _normalize_season_type(
        row: dict,
    ) -> str:
        """
        根据 NBA Schedule 数据中的标签
        尽量判断比赛类型。

        返回：
        - Pre Season
        - Regular Season
        - Playoffs
        - All Star
        """

        texts = [
            row.get("gameLabel"),
            row.get("gameSubLabel"),
            row.get("gameSubtype"),
        ]

        combined_text = " ".join(
            str(value)
            for value in texts
            if value is not None
            and not (
                isinstance(value, float)
                and pd.isna(value)
            )
        ).lower()

        if (
            "preseason" in combined_text
            or "pre season" in combined_text
        ):
            return "Pre Season"

        if "playoff" in combined_text or "nba finals" in combined_text:
            return "Playoffs"

        if "all-star" in combined_text:
            return "All Star"

        game_id = NBAGamesService._normalize_game_id(row.get("gameId"))
        return {"001": "Pre Season", "002": "Regular Season", "003": "All Star",
                "004": "Playoffs", "005": "Play In"}.get((game_id or "")[:3], "Unknown")

    # ========================================================
    # NBA API
    # ========================================================

    async def _fetch_schedule(
        self,
        *,
        season: str,
    ):
        """
        获取指定 NBA 赛季赛程。

        nba_api 是同步 HTTP library，
        所以放到 asyncio.to_thread() 中，
        防止阻塞 FastAPI async event loop。
        """

        def fetch():
            endpoint = ScheduleLeagueV2(
                league_id="00",
                season=season,
                timeout=60,
            )

            return (
                endpoint
                .season_games
                .get_data_frame()
            )

        dataframe = await asyncio.to_thread(
            fetch
        )

        return dataframe

    # ========================================================
    # Sync Games
    # ========================================================

    async def sync_games(
        self,
        session: AsyncSession,
        *,
        season: str,
    ) -> dict:
        """
        将指定赛季赛程同步进入 PostgreSQL。

        逻辑：

        NBA API
            ↓
        ScheduleLeagueV2
            ↓
        Match Teams
            ↓
        Existing Game?
          /       \\
        No         Yes
        ↓           ↓
      INSERT      UPDATE
        """

        # ====================================================
        # Fetch NBA Schedule
        # ====================================================

        dataframe = await self._fetch_schedule(
            season=season,
        )

        # ====================================================
        # Load Teams
        # ====================================================

        team_result = await session.execute(
            select(Team)
        )

        teams = team_result.scalars().all()

        # NBA Team ID -> Team ORM object
        team_map = {
            team.nba_team_id: team
            for team in teams
        }

        # ====================================================
        # Load Existing Games
        # ====================================================

        game_result = await session.execute(
            select(Game)
        )

        existing_games = (
            game_result.scalars().all()
        )

        # NBA Game ID -> Game ORM object
        game_map = {
            game.nba_game_id: game
            for game in existing_games
        }

        # ====================================================
        # Counters
        # ====================================================

        created_count = 0
        updated_count = 0
        skipped_count = 0

        unmatched_team_count = 0

        # ====================================================
        # Process Schedule Rows
        # ====================================================

        rows = dataframe.to_dict(
            orient="records"
        )

        for row in rows:

            # ------------------------------------------------
            # NBA Game ID
            # ------------------------------------------------

            nba_game_id = (
                self._normalize_game_id(
                    row.get("gameId")
                )
            )

            if not nba_game_id:
                skipped_count += 1
                continue

            # ------------------------------------------------
            # Team IDs
            # ------------------------------------------------

            home_nba_team_id = (
                self._safe_int(
                    row.get(
                        "homeTeam_teamId"
                    )
                )
            )

            away_nba_team_id = (
                self._safe_int(
                    row.get(
                        "awayTeam_teamId"
                    )
                )
            )

            home_team = team_map.get(
                home_nba_team_id
            )

            away_team = team_map.get(
                away_nba_team_id
            )

            # 如果球队无法匹配，
            # 说明 teams 表可能还没有同步完整。
            if (
                home_team is None
                or away_team is None
            ):
                unmatched_team_count += 1
                skipped_count += 1
                continue

            # ------------------------------------------------
            # Tipoff Time
            # ------------------------------------------------

            raw_tipoff = row.get(
                "gameDateTimeUTC"
            )

            if raw_tipoff is None:
                skipped_count += 1
                continue

            try:
                if pd.isna(raw_tipoff):
                    skipped_count += 1
                    continue
            except TypeError:
                pass

            try:
                tipoff_time: datetime = (
                    pd.to_datetime(
                        raw_tipoff,
                        utc=True,
                    )
                    .to_pydatetime()
                )
            except Exception:
                skipped_count += 1
                continue

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            status = self._normalize_status(
                row.get("gameStatus"),
                row.get("gameStatusText"),
            )

            # ------------------------------------------------
            # Season Type
            # ------------------------------------------------

            season_type = (
                self._normalize_season_type(
                    row
                )
            )

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            home_score = self._safe_int(
                row.get(
                    "homeTeam_score"
                )
            )

            away_score = self._safe_int(
                row.get(
                    "awayTeam_score"
                )
            )

            # ------------------------------------------------
            # Find Existing Game
            # ------------------------------------------------

            existing_game = game_map.get(
                nba_game_id
            )

            # =================================================
            # CREATE
            # =================================================

            if existing_game is None:

                game = Game(
                    nba_game_id=nba_game_id,

                    season=season,

                    season_type=(
                        season_type
                    ),

                    home_team_id=(
                        home_team.id
                    ),

                    away_team_id=(
                        away_team.id
                    ),

                    tipoff_time=(
                        tipoff_time
                    ),

                    status=status,

                    home_score=(
                        home_score
                    ),

                    away_score=(
                        away_score
                    ),
                )

                session.add(game)

                # 加入内存 map，
                # 防止同一次同步出现重复记录。
                game_map[
                    nba_game_id
                ] = game

                created_count += 1

            # =================================================
            # UPDATE
            # =================================================

            else:

                existing_game.season = (
                    season
                )

                existing_game.season_type = (
                    season_type
                )

                existing_game.home_team_id = (
                    home_team.id
                )

                existing_game.away_team_id = (
                    away_team.id
                )

                existing_game.tipoff_time = (
                    tipoff_time
                )
                existing_game.tipoff_time_estimated = False

                existing_game.status = (
                    status
                )

                existing_game.home_score = (
                    home_score
                )

                existing_game.away_score = (
                    away_score
                )

                updated_count += 1

        # ====================================================
        # Commit Transaction
        # ====================================================

        await session.commit()

        # ====================================================
        # Result
        # ====================================================

        return {
            "season": season,

            "rows_received": len(
                dataframe
            ),

            "created": (
                created_count
            ),

            "updated": (
                updated_count
            ),

            "skipped": (
                skipped_count
            ),

            "unmatched_teams": (
                unmatched_team_count
            ),
        }

    # ========================================================
    # Get Games
    # ========================================================

    async def get_games(
        self,
        session: AsyncSession,
        *,
        limit: int = 100,
        season: str | None = None,
    ) -> list[Game]:
        """
        查询 games 表。

        可以：
        - 获取所有比赛
        - 按 season 筛选
        """

        query = select(Game)

        # ----------------------------------------------------
        # Season Filter
        # ----------------------------------------------------

        if season is not None:
            query = query.where(
                Game.season == season
            )

        # ----------------------------------------------------
        # Sort + Limit
        # ----------------------------------------------------

        query = (
            query
            .order_by(
                Game.tipoff_time.asc()
            )
            .limit(limit)
        )

        result = await session.execute(
            query
        )

        return list(
            result.scalars().all()
        )


# ============================================================
# Singleton
# ============================================================

nba_games_service = NBAGamesService()
