import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nba_api.stats.endpoints import playerindex
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

from app.core.config import settings
from app.models.player import Player
from app.models.team import Team


class NBADataService:
    """
    NBA 数据服务。

    当前支持：
    - Teams
    - Players
    - Current roster / player-team mapping
    - Player position

    后续继续加入：
    - Games
    - Player Game Stats
    - Advanced Stats
    """

    # ========================================================
    # Teams
    # ========================================================

    async def sync_teams(
        self,
        session: AsyncSession,
    ) -> dict:
        """
        同步 NBA 30 支球队。
        """

        nba_team_list = nba_teams.get_teams()

        created_count = 0
        updated_count = 0

        for team_data in nba_team_list:
            nba_team_id = team_data["id"]

            result = await session.execute(
                select(Team).where(
                    Team.nba_team_id == nba_team_id
                )
            )

            existing_team = result.scalar_one_or_none()

            if existing_team is None:
                team = Team(
                    nba_team_id=nba_team_id,
                    abbreviation=team_data["abbreviation"],
                    city=team_data["city"],
                    name=team_data["nickname"],
                    full_name=team_data["full_name"],
                    conference=None,
                    division=None,
                    active=True,
                )

                session.add(team)
                created_count += 1

            else:
                existing_team.abbreviation = (
                    team_data["abbreviation"]
                )

                existing_team.city = (
                    team_data["city"]
                )

                existing_team.name = (
                    team_data["nickname"]
                )

                existing_team.full_name = (
                    team_data["full_name"]
                )

                existing_team.active = True

                updated_count += 1

        await session.commit()

        return {
            "total": len(nba_team_list),
            "created": created_count,
            "updated": updated_count,
        }

    # ========================================================
    # Players
    # ========================================================

    async def sync_players(
        self,
        session: AsyncSession,
    ) -> dict:
        """
        同步 NBA 当前 active players。
        """

        nba_player_list = nba_players.get_active_players()

        created_count = 0
        updated_count = 0

        for player_data in nba_player_list:
            nba_player_id = player_data["id"]

            result = await session.execute(
                select(Player).where(
                    Player.nba_player_id == nba_player_id
                )
            )

            existing_player = result.scalar_one_or_none()

            if existing_player is None:
                player = Player(
                    nba_player_id=nba_player_id,
                    full_name=player_data["full_name"],
                    first_name=player_data.get("first_name"),
                    last_name=player_data.get("last_name"),
                    position=None,
                    current_team_id=None,
                    active=player_data.get(
                        "is_active",
                        True,
                    ),
                )

                session.add(player)
                created_count += 1

            else:
                existing_player.full_name = (
                    player_data["full_name"]
                )

                existing_player.first_name = (
                    player_data.get("first_name")
                )

                existing_player.last_name = (
                    player_data.get("last_name")
                )

                existing_player.active = (
                    player_data.get(
                        "is_active",
                        True,
                    )
                )

                updated_count += 1

        await session.commit()

        return {
            "total": len(nba_player_list),
            "created": created_count,
            "updated": updated_count,
        }

    # ========================================================
    # Rosters / PlayerIndex
    # ========================================================

    async def sync_rosters(
        self,
        session: AsyncSession,
    ) -> dict:
        """
        使用 NBA PlayerIndex 同步：

        Player
            ↓
        Team
        Position

        PlayerIndex 提供：
        - PERSON_ID
        - TEAM_ID
        - POSITION
        - TEAM_ABBREVIATION
        - ROSTER_STATUS
        """

        # ----------------------------------------------------
        # nba_api 是同步 HTTP library。
        #
        # 使用 asyncio.to_thread，
        # 避免阻塞 FastAPI event loop。
        # ----------------------------------------------------

        def fetch_player_index():
            endpoint = playerindex.PlayerIndex(
                season=settings.nba_season,

                # 只请求 active players
                active_nullable="1",

                timeout=60,
            )

            return (
                endpoint
                .player_index
                .get_data_frame()
            )

        dataframe = await asyncio.to_thread(
            fetch_player_index
        )

        # ----------------------------------------------------
        # 一次把 Teams 读入内存
        # ----------------------------------------------------

        team_result = await session.execute(
            select(Team)
        )

        teams = team_result.scalars().all()

        team_map = {
            team.nba_team_id: team
            for team in teams
        }

        # ----------------------------------------------------
        # 一次把 Players 读入内存
        # ----------------------------------------------------

        player_result = await session.execute(
            select(Player)
        )

        players = player_result.scalars().all()

        player_map = {
            player.nba_player_id: player
            for player in players
        }

        updated_count = 0
        created_count = 0
        unmatched_team_count = 0

        # ----------------------------------------------------
        # 遍历 PlayerIndex
        # ----------------------------------------------------

        for row in dataframe.to_dict(
            orient="records"
        ):

            nba_player_id = int(
                row["PERSON_ID"]
            )

            nba_team_id_raw = row.get(
                "TEAM_ID"
            )

            # -----------------------------------------------
            # Player
            # -----------------------------------------------

            player = player_map.get(
                nba_player_id
            )

            # 如果 PlayerIndex 里出现 static players
            # 没有的球员，则直接创建，保证同步完整。
            if player is None:

                first_name = (
                    row.get("PLAYER_FIRST_NAME")
                    or ""
                )

                last_name = (
                    row.get("PLAYER_LAST_NAME")
                    or ""
                )

                full_name = (
                    f"{first_name} {last_name}"
                    .strip()
                )

                player = Player(
                    nba_player_id=nba_player_id,
                    full_name=full_name,
                    first_name=first_name or None,
                    last_name=last_name or None,
                    position=None,
                    current_team_id=None,
                    active=True,
                )

                session.add(player)

                player_map[nba_player_id] = player

                created_count += 1

            # -----------------------------------------------
            # Position
            # -----------------------------------------------

            position = row.get(
                "POSITION"
            )

            if position:
                player.position = str(
                    position
                )

            # -----------------------------------------------
            # Team
            # -----------------------------------------------

            try:
                nba_team_id = int(
                    nba_team_id_raw
                )
            except (
                TypeError,
                ValueError,
            ):
                nba_team_id = 0

            # TEAM_ID = 0 一般代表没有当前球队
            if nba_team_id == 0:
                player.current_team_id = None

            else:
                team = team_map.get(
                    nba_team_id
                )

                if team is not None:
                    player.current_team_id = (
                        team.id
                    )

                else:
                    player.current_team_id = None
                    unmatched_team_count += 1

            player.active = True

            updated_count += 1

        # ----------------------------------------------------
        # Commit
        # ----------------------------------------------------

        await session.commit()

        return {
            "season": settings.nba_season,
            "rows_received": len(dataframe),
            "created_players": created_count,
            "updated_players": updated_count,
            "unmatched_teams": unmatched_team_count,
        }

    # ========================================================
    # Get Teams
    # ========================================================

    async def get_teams(
        self,
        session: AsyncSession,
    ) -> list[Team]:
        """
        获取全部球队。
        """

        result = await session.execute(
            select(Team).order_by(
                Team.abbreviation
            )
        )

        return list(
            result.scalars().all()
        )

    # ========================================================
    # Get Players
    # ========================================================

    async def get_players(
        self,
        session: AsyncSession,
        *,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Player]:
        """
        查询球员。
        """

        query = select(Player)

        if active_only:
            query = query.where(
                Player.active.is_(True)
            )

        query = (
            query
            .order_by(Player.full_name)
            .limit(limit)
        )

        result = await session.execute(
            query
        )

        return list(
            result.scalars().all()
        )


nba_data_service = NBADataService()