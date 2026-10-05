from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.game import Game
from app.models.news import NewsEvent
from app.models.player import Player
from app.models.team import Team
from app.schemas.news import NewsEventCreate


class NBANewsService:
    """
    NBA News / Injury Event Service。

    主要负责：

    1. 根据 NBA ID 查询 Player / Team / Game
    2. 创建结构化 NewsEvent
    3. 查询球员相关新闻
    4. 查询球队相关新闻
    5. 查询最近新闻事件

    数据流：

    Web Search
        ↓
    News Extraction Agent
        ↓
    NBANewsService
        ↓
    news_events
        ↓
    Context Builder
        ↓
    Forecaster Agent
    """

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
        根据 NBA Player ID 查询球员。

        例如：
        Donovan Mitchell
        nba_player_id = 1628378
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
                f"Player with NBA ID "
                f"{nba_player_id} not found."
            )

        return player

    # ========================================================
    # Team
    # ========================================================

    async def get_team_by_nba_id(
        self,
        session: AsyncSession,
        *,
        nba_team_id: int,
    ) -> Team:
        """
        根据 NBA Team ID 查询球队。
        """

        result = await session.execute(
            select(Team).where(
                Team.nba_team_id
                == nba_team_id
            )
        )

        team = result.scalar_one_or_none()

        if team is None:
            raise ValueError(
                f"Team with NBA ID "
                f"{nba_team_id} not found."
            )

        return team

    # ========================================================
    # Game
    # ========================================================

    async def get_game_by_nba_id(
        self,
        session: AsyncSession,
        *,
        nba_game_id: str,
    ) -> Game:
        """
        根据 NBA Game ID 查询比赛。
        """

        result = await session.execute(
            select(Game).where(
                Game.nba_game_id
                == nba_game_id
            )
        )

        game = result.scalar_one_or_none()

        if game is None:
            raise ValueError(
                f"Game with NBA ID "
                f"{nba_game_id} not found."
            )

        return game

    # ========================================================
    # Internal Resolve Player
    # ========================================================

    async def _resolve_player(
        self,
        session: AsyncSession,
        *,
        nba_player_id: int | None,
    ) -> Player | None:
        """
        如果提供 nba_player_id，则尝试匹配 Player。

        没有提供则返回 None。
        """

        if nba_player_id is None:
            return None

        return await self.get_player_by_nba_id(
            session,
            nba_player_id=nba_player_id,
        )

    # ========================================================
    # Internal Resolve Team
    # ========================================================

    async def _resolve_team(
        self,
        session: AsyncSession,
        *,
        nba_team_id: int | None,
    ) -> Team | None:
        """
        如果提供 nba_team_id，则尝试匹配 Team。

        没有提供则返回 None。
        """

        if nba_team_id is None:
            return None

        return await self.get_team_by_nba_id(
            session,
            nba_team_id=nba_team_id,
        )

    # ========================================================
    # Internal Resolve Game
    # ========================================================

    async def _resolve_game(
        self,
        session: AsyncSession,
        *,
        nba_game_id: str | None,
    ) -> Game | None:
        """
        如果提供 nba_game_id，则尝试匹配 Game。

        没有提供则返回 None。
        """

        if nba_game_id is None:
            return None

        return await self.get_game_by_nba_id(
            session,
            nba_game_id=nba_game_id,
        )

    # ========================================================
    # Create News Event
    # ========================================================

    async def create_event(
        self,
        session: AsyncSession,
        *,
        payload: NewsEventCreate,
    ) -> NewsEvent:
        """
        创建结构化 NBA 新闻事件。

        支持：

        injury
        rest
        ruled_out
        questionable
        probable
        available
        starting_lineup
        minutes_restriction
        role_change
        trade
        suspension
        return_from_injury
        """

        # ====================================================
        # Resolve entities
        # ====================================================

        player = await self._resolve_player(
            session,
            nba_player_id=(
                payload.nba_player_id
            ),
        )

        team = await self._resolve_team(
            session,
            nba_team_id=(
                payload.nba_team_id
            ),
        )

        game = await self._resolve_game(
            session,
            nba_game_id=(
                payload.nba_game_id
            ),
        )

        # ====================================================
        # Automatically infer team from player
        # ====================================================

        if team is not None:
            team_id = team.id

        elif (
            player is not None
            and player.current_team_id
            is not None
        ):
            team_id = (
                player.current_team_id
            )

        else:
            team_id = None

        # ====================================================
        # Create NewsEvent
        # ====================================================

        event = NewsEvent(
            # ------------------------------------------------
            # Relations
            # ------------------------------------------------

            game_id=(
                game.id
                if game is not None
                else None
            ),

            player_id=(
                player.id
                if player is not None
                else None
            ),

            team_id=team_id,

            # ------------------------------------------------
            # Event
            # ------------------------------------------------

            event_type=(
                payload.event_type
            ),

            player_status=(
                payload.player_status
            ),

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            title=(
                payload.title
            ),

            body=(
                payload.body
            ),

            # ------------------------------------------------
            # Source
            # ------------------------------------------------

            source=(
                payload.source
            ),

            source_url=(
                payload.source_url
            ),

            # ------------------------------------------------
            # Time
            # ------------------------------------------------

            published_at=(
                payload.published_at
            ),

            ingested_at=datetime.now(
                timezone.utc
            ),
        )

        session.add(event)

        await session.commit()

        await session.refresh(
            event
        )

        return event

    # ========================================================
    # Get Event By ID
    # ========================================================

    async def get_event_by_id(
        self,
        session: AsyncSession,
        *,
        event_id: UUID,
    ) -> NewsEvent | None:
        """
        根据 NewsEvent UUID 查询事件。
        """

        result = await session.execute(
            select(NewsEvent).where(
                NewsEvent.id == event_id
            )
        )

        return (
            result.scalar_one_or_none()
        )

    # ========================================================
    # Get Player Events
    # ========================================================

    async def get_player_events(
        self,
        session: AsyncSession,
        *,
        player_id: UUID,
        limit: int = 20,
    ) -> list[NewsEvent]:
        """
        获取某个球员最近的新闻事件。
        """

        result = await session.execute(
            select(NewsEvent)
            .where(
                NewsEvent.player_id
                == player_id
            )
            .order_by(
                NewsEvent.published_at.desc()
            )
            .limit(limit)
        )

        return list(
            result.scalars().all()
        )

    # ========================================================
    # Get Player Events By NBA ID
    # ========================================================

    async def get_player_events_by_nba_id(
        self,
        session: AsyncSession,
        *,
        nba_player_id: int,
        limit: int = 20,
    ) -> list[NewsEvent]:
        """
        直接通过 NBA Player ID
        获取球员最近新闻。
        """

        player = (
            await self.get_player_by_nba_id(
                session,
                nba_player_id=nba_player_id,
            )
        )

        return await self.get_player_events(
            session,
            player_id=player.id,
            limit=limit,
        )

    # ========================================================
    # Get Team Events
    # ========================================================

    async def get_team_events(
        self,
        session: AsyncSession,
        *,
        team_id: UUID,
        limit: int = 50,
    ) -> list[NewsEvent]:
        """
        获取某球队最近新闻事件。

        例如：
        - teammate injury
        - starting lineup
        - rotation changes
        - trades
        """

        result = await session.execute(
            select(NewsEvent)
            .where(
                NewsEvent.team_id
                == team_id
            )
            .order_by(
                NewsEvent.published_at.desc()
            )
            .limit(limit)
        )

        return list(
            result.scalars().all()
        )

    # ========================================================
    # Get Game Events
    # ========================================================

    async def get_game_events(
        self,
        session: AsyncSession,
        *,
        game_id: UUID,
        limit: int = 50,
    ) -> list[NewsEvent]:
        """
        获取和某场比赛直接相关的新闻。
        """

        result = await session.execute(
            select(NewsEvent)
            .where(
                NewsEvent.game_id
                == game_id
            )
            .order_by(
                NewsEvent.published_at.desc()
            )
            .limit(limit)
        )

        return list(
            result.scalars().all()
        )

    # ========================================================
    # Recent Events
    # ========================================================

    async def get_recent_events(
        self,
        session: AsyncSession,
        *,
        limit: int = 100,
    ) -> list[NewsEvent]:
        """
        获取整个系统最近的新闻事件。
        """

        result = await session.execute(
            select(NewsEvent)
            .order_by(
                NewsEvent.published_at.desc()
            )
            .limit(limit)
        )

        return list(
            result.scalars().all()
        )


# ============================================================
# Singleton
# ============================================================

nba_news_service = NBANewsService()