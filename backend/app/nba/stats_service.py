import asyncio
from typing import Any

import pandas as pd

from nba_api.stats.endpoints import playergamelogs
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game import Game
from app.models.player import Player
from app.models.stats import PlayerGameStat
from app.models.team import Team


class NBAStatsService:
    """
    NBA Player Game Stats Service。

    负责：

    NBA PlayerGameLogs API
            ↓
    获取指定赛季 / 赛季类型的球员逐场数据
            ↓
    匹配：
        Game
        Player
        Team
            ↓
    写入 PostgreSQL player_game_stats

    支持：

    season:
        2025-26
        2026-27
        ...

    season_type:
        Pre Season
        Regular Season
        Playoffs
    """

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:
        """
        安全转换成 int。

        None / NaN / 空值 -> None
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

    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float | None:
        """
        安全转换成 float。

        None / NaN / 空值 -> None
        """

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        try:
            return float(value)

        except (TypeError, ValueError):
            return None

    # ========================================================

    @staticmethod
    def _normalize_game_id(
        value: Any,
    ) -> str:
        """
        统一 NBA GAME_ID。

        NBA GAME_ID 通常类似：

        0022500001

        Pandas 有可能把它变成：

        22500001
        或
        22500001.0

        所以这里统一恢复成 10 位字符串。
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
        # 22500001.0
        # ->
        # 22500001
        if game_id.endswith(".0"):
            game_id = game_id[:-2]

        # 补回前导 0
        if game_id.isdigit():
            game_id = game_id.zfill(10)

        return game_id

    # ========================================================

    @staticmethod
    def _true_shooting_pct(
        *,
        points: int | None,
        fga: int | None,
        fta: int | None,
    ) -> float | None:
        """
        计算 True Shooting Percentage。

        TS% = PTS /
              [2 × (FGA + 0.44 × FTA)]
        """

        if (
            points is None
            or fga is None
            or fta is None
        ):
            return None

        denominator = 2 * (
            fga + 0.44 * fta
        )

        if denominator <= 0:
            return None

        return points / denominator

    # ========================================================
    # NBA API
    # ========================================================

    async def _fetch_player_game_logs(
        self,
        *,
        season: str,
        season_type: str,
    ):
        """
        从 NBA API 获取指定赛季球员逐场比赛数据。

        nba_api 本身使用同步 HTTP，
        所以使用 asyncio.to_thread()
        避免阻塞 FastAPI event loop。
        """

        def fetch():
            endpoint = (
                playergamelogs.PlayerGameLogs(
                    season_nullable=season,
                    season_type_nullable=(
                        season_type
                    ),
                    timeout=60,
                )
            )

            dataframe = (
                endpoint
                .player_game_logs
                .get_data_frame()
            )

            return dataframe

        dataframe = await asyncio.to_thread(
            fetch
        )

        return dataframe

    # ========================================================
    # Sync Player Game Stats
    # ========================================================

    async def sync_player_game_stats(
        self,
        session: AsyncSession,
        *,
        season: str,
        season_type: str,
    ) -> dict:
        """
        同步指定赛季 / 类型的球员逐场数据。

        示例：

        2025-26 Regular Season

        或：

        2026-27 Pre Season
        """

        # ====================================================
        # Fetch NBA Data
        # ====================================================

        dataframe = (
            await self._fetch_player_game_logs(
                season=season,
                season_type=season_type,
            )
        )

        # 如果 NBA API 返回空数据
        if dataframe.empty:
            return {
                "season": season,
                "season_type": season_type,
                "rows_received": 0,
                "created": 0,
                "updated": 0,
                "skipped": 0,
                "unmatched_games": 0,
                "unmatched_players": 0,
                "unmatched_teams": 0,
            }

        # ====================================================
        # Load Games
        # ====================================================

        game_result = await session.execute(
            select(Game).where(
                Game.season == season
            )
        )

        games = game_result.scalars().all()

        # NBA GAME_ID -> Game
        game_map = {
            game.nba_game_id: game
            for game in games
        }

        # ====================================================
        # Load Players
        # ====================================================

        player_result = await session.execute(
            select(Player)
        )

        players = player_result.scalars().all()

        # NBA PLAYER_ID -> Player
        player_map = {
            player.nba_player_id: player
            for player in players
        }

        # ====================================================
        # Load Teams
        # ====================================================

        team_result = await session.execute(
            select(Team)
        )

        teams = team_result.scalars().all()

        # NBA TEAM_ID -> Team
        team_map = {
            team.nba_team_id: team
            for team in teams
        }

        # ====================================================
        # Load Existing Stats
        # ====================================================

        stat_result = await session.execute(
            select(PlayerGameStat)
            .join(
                Game,
                PlayerGameStat.game_id
                == Game.id,
            )
            .where(
                Game.season == season
            )
        )

        existing_stats = (
            stat_result.scalars().all()
        )

        # (game UUID, player UUID)
        # ->
        # PlayerGameStat
        stat_map = {
            (
                stat.game_id,
                stat.player_id,
            ): stat
            for stat in existing_stats
        }

        # ====================================================
        # Counters
        # ====================================================

        created_count = 0
        updated_count = 0
        skipped_count = 0

        unmatched_game_count = 0
        unmatched_player_count = 0
        unmatched_team_count = 0

        # ====================================================
        # Process NBA Rows
        # ====================================================

        rows = dataframe.to_dict(
            orient="records"
        )

        for row in rows:

            # ------------------------------------------------
            # NBA IDs
            # ------------------------------------------------

            nba_game_id = (
                self._normalize_game_id(
                    row.get("GAME_ID")
                )
            )

            nba_player_id = (
                self._safe_int(
                    row.get("PLAYER_ID")
                )
            )

            nba_team_id = (
                self._safe_int(
                    row.get("TEAM_ID")
                )
            )

            # ------------------------------------------------
            # Find Game
            # ------------------------------------------------

            game = game_map.get(
                nba_game_id
            )

            if game is None:
                unmatched_game_count += 1
                skipped_count += 1
                continue

            # ------------------------------------------------
            # Find Player
            # ------------------------------------------------

            player = player_map.get(
                nba_player_id
            )

            if player is None:
                unmatched_player_count += 1
                skipped_count += 1
                continue

            # ------------------------------------------------
            # Find Team
            # ------------------------------------------------

            team = team_map.get(
                nba_team_id
            )

            if team is None:
                unmatched_team_count += 1
                skipped_count += 1
                continue

            # =================================================
            # Basic Box Score
            # =================================================

            minutes = self._safe_float(
                row.get("MIN")
            )

            points = self._safe_int(
                row.get("PTS")
            )

            rebounds = self._safe_int(
                row.get("REB")
            )

            assists = self._safe_int(
                row.get("AST")
            )

            steals = self._safe_int(
                row.get("STL")
            )

            blocks = self._safe_int(
                row.get("BLK")
            )

            turnovers = self._safe_int(
                row.get("TOV")
            )

            # =================================================
            # Shooting
            # =================================================

            field_goals_made = (
                self._safe_int(
                    row.get("FGM")
                )
            )

            field_goals_attempted = (
                self._safe_int(
                    row.get("FGA")
                )
            )

            three_points_made = (
                self._safe_int(
                    row.get("FG3M")
                )
            )

            three_points_attempted = (
                self._safe_int(
                    row.get("FG3A")
                )
            )

            free_throws_made = (
                self._safe_int(
                    row.get("FTM")
                )
            )

            free_throws_attempted = (
                self._safe_int(
                    row.get("FTA")
                )
            )

            # =================================================
            # Other
            # =================================================

            plus_minus = (
                self._safe_float(
                    row.get("PLUS_MINUS")
                )
            )

            # =================================================
            # True Shooting %
            # =================================================

            true_shooting_pct = (
                self._true_shooting_pct(
                    points=points,
                    fga=(
                        field_goals_attempted
                    ),
                    fta=(
                        free_throws_attempted
                    ),
                )
            )

            # =================================================
            # Existing?
            # =================================================

            key = (
                game.id,
                player.id,
            )

            existing_stat = (
                stat_map.get(key)
            )

            # =================================================
            # CREATE
            # =================================================

            if existing_stat is None:

                stat = PlayerGameStat(
                    # -----------------------------------------
                    # Foreign keys
                    # -----------------------------------------

                    game_id=game.id,
                    player_id=player.id,
                    team_id=team.id,

                    # -----------------------------------------
                    # Starter
                    # -----------------------------------------

                    # PlayerGameLogs 不提供
                    # starter 信息。
                    #
                    # 后续我们会使用 BoxScore
                    # 或 Lineup 数据再补。
                    starter=False,

                    # -----------------------------------------
                    # Basic
                    # -----------------------------------------

                    minutes=minutes,
                    points=points,
                    rebounds=rebounds,
                    assists=assists,
                    steals=steals,
                    blocks=blocks,
                    turnovers=turnovers,

                    # -----------------------------------------
                    # Shooting
                    # -----------------------------------------

                    field_goals_made=(
                        field_goals_made
                    ),

                    field_goals_attempted=(
                        field_goals_attempted
                    ),

                    three_points_made=(
                        three_points_made
                    ),

                    three_points_attempted=(
                        three_points_attempted
                    ),

                    free_throws_made=(
                        free_throws_made
                    ),

                    free_throws_attempted=(
                        free_throws_attempted
                    ),

                    # -----------------------------------------
                    # Advanced
                    # -----------------------------------------

                    # PlayerGameLogs 不提供，
                    # 后续 Advanced Stats 再补。
                    usage_pct=None,

                    true_shooting_pct=(
                        true_shooting_pct
                    ),

                    offensive_rating=None,
                    defensive_rating=None,
                    pace=None,

                    plus_minus=plus_minus,
                )

                session.add(stat)

                # 当前同步过程中也记录，
                # 防止出现重复 row。
                stat_map[key] = stat

                created_count += 1

            # =================================================
            # UPDATE
            # =================================================

            else:

                # ---------------------------------------------
                # Team
                # ---------------------------------------------

                existing_stat.team_id = (
                    team.id
                )

                # ---------------------------------------------
                # Basic
                # ---------------------------------------------

                existing_stat.minutes = (
                    minutes
                )

                existing_stat.points = (
                    points
                )

                existing_stat.rebounds = (
                    rebounds
                )

                existing_stat.assists = (
                    assists
                )

                existing_stat.steals = (
                    steals
                )

                existing_stat.blocks = (
                    blocks
                )

                existing_stat.turnovers = (
                    turnovers
                )

                # ---------------------------------------------
                # Shooting
                # ---------------------------------------------

                existing_stat.field_goals_made = (
                    field_goals_made
                )

                existing_stat.field_goals_attempted = (
                    field_goals_attempted
                )

                existing_stat.three_points_made = (
                    three_points_made
                )

                existing_stat.three_points_attempted = (
                    three_points_attempted
                )

                existing_stat.free_throws_made = (
                    free_throws_made
                )

                existing_stat.free_throws_attempted = (
                    free_throws_attempted
                )

                # ---------------------------------------------
                # Advanced
                # ---------------------------------------------

                existing_stat.true_shooting_pct = (
                    true_shooting_pct
                )

                existing_stat.plus_minus = (
                    plus_minus
                )

                updated_count += 1

        # ====================================================
        # Commit
        # ====================================================

        await session.commit()

        # ====================================================
        # Response
        # ====================================================

        return {
            "season": season,

            "season_type": (
                season_type
            ),

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

            "unmatched_games": (
                unmatched_game_count
            ),

            "unmatched_players": (
                unmatched_player_count
            ),

            "unmatched_teams": (
                unmatched_team_count
            ),
        }

    # ========================================================
    # Get Player Game Stats
    # ========================================================

    async def get_player_game_stats(
        self,
        session: AsyncSession,
        *,
        limit: int = 100,
    ) -> list[PlayerGameStat]:
        """
        获取数据库中的球员逐场比赛数据。
        """

        result = await session.execute(
            select(PlayerGameStat)
            .join(
                Game,
                PlayerGameStat.game_id
                == Game.id,
            )
            .order_by(
                Game.tipoff_time.desc()
            )
            .limit(limit)
        )

        return list(
            result.scalars().all()
        )


# ============================================================
# Singleton
# ============================================================

nba_stats_service = NBAStatsService()