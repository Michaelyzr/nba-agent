from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.player import Player
from app.news.context import load_evidence


class ContextService:
    """
    Forecaster Context Builder。

    将结构化 news_events 转成
    Forecaster Agent 可以直接使用的上下文。

    后续还会加入：
    - teammate injury
    - starting lineup
    - rest
    - back-to-back
    - trade
    - role change
    """

    async def build_player_context(
        self,
        session: AsyncSession,
        *,
        player: Player,
        lookback_hours: int | None = None,
    ) -> str:
        """
        构建近期新闻及未解除长期事件上下文。
        """

        events = await load_evidence(
            session, as_of=datetime.now(timezone.utc), player_ids=[player.id],
            team_ids=[player.current_team_id] if player.current_team_id else [],
            lookback_hours=lookback_hours,
        )

        if not events:
            return (
                "No relevant recent news or unresolved long-term events were found. "
                "This does not establish that any player is healthy or available."
            )

        context_lines = [
            "Long-term reports remain until an explicit follow-up resolves them. "
            "needs_review=True means the old report does not confirm current status."
        ]

        for item in events:
            event = item.event

            line = (
                f"[{event.event_type}] "
                f"{event.title}"
            )

            if event.player_status:
                line += (
                    f" | status="
                    f"{event.player_status}"
                )

            if event.body:
                line += (
                    f" | details="
                    f"{event.body}"
                )

            line += (
                f" | source="
                f"{event.source}"
            )

            line += (
                f" | published_at="
                f"{event.published_at.isoformat()}"
            )
            line += f" | persistent={item.persistent} | needs_review={item.needs_review}"

            context_lines.append(line)

        return "\n".join(
            context_lines
        )


context_service = ContextService()
