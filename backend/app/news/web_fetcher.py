import asyncio
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from app.core.config import settings


class WebNewsFetcher:
    """
    NBA Web News Fetcher。

    负责：
    1. 使用 OpenAI Responses API 联网搜索
    2. 查找最新 NBA 新闻 / injury / lineup 信息
    3. 返回原始文本
    4. 返回搜索来源 URL

    注意：
    这一层暂时不负责：
    - 解析 event_type
    - 匹配 player_id
    - 写 news_events

    这些由下一层 News Extraction Agent 完成。
    """

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if not settings.openai_api_key:
            raise ValueError("请先配置 OPENAI_API_KEY")
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )
        return self._client

    # ========================================================
    # Sources
    # ========================================================

    @staticmethod
    def _extract_sources(
        response: Any,
    ) -> list[dict]:
        """
        从 Responses API 返回内容中提取 URL citations。

        返回：

        [
            {
                "title": "...",
                "url": "..."
            }
        ]
        """

        sources: list[dict] = []
        seen_urls: set[str] = set()

        for output_item in getattr(
            response,
            "output",
            [],
        ):
            if getattr(
                output_item,
                "type",
                None,
            ) != "message":
                continue

            for content_item in getattr(
                output_item,
                "content",
                [],
            ):
                annotations = getattr(
                    content_item,
                    "annotations",
                    [],
                )

                for annotation in annotations:
                    url = getattr(
                        annotation,
                        "url",
                        None,
                    )

                    title = getattr(
                        annotation,
                        "title",
                        None,
                    )

                    if (
                        url
                        and url not in seen_urls
                    ):
                        seen_urls.add(url)

                        sources.append(
                            {
                                "title": (
                                    title
                                    or "Unknown source"
                                ),
                                "url": url,
                            }
                        )

        return sources

    # ========================================================
    # Fetch Player News
    # ========================================================

    async def fetch_player_news(
        self,
        *,
        player_name: str,
        team_name: str | None,
        lookback_hours: int = 72,
    ) -> dict:
        """
        自动搜索某 NBA 球员最近新闻。

        重点搜索：
        - injury
        - questionable
        - doubtful
        - probable
        - out
        - rest
        - minutes restriction
        - starting lineup
        - role change
        """

        now = datetime.now(
            timezone.utc
        )

        team_text = (
            team_name
            if team_name
            else "unknown team"
        )

        prompt = f"""
Search the live web for the most recent credible NBA news
about the following player.

PLAYER:
{player_name}

TEAM:
{team_text}

CURRENT UTC TIME:
{now.isoformat()}

SEARCH WINDOW:
Approximately the last {lookback_hours} hours.

Focus specifically on information that could affect an NBA
game forecast:

- injury status
- out
- doubtful
- questionable
- probable
- available
- rest
- minutes restriction
- starting lineup changes
- role changes
- rotation changes
- return from injury
- suspension
- trade or roster movement

Use recent and credible sources.

Prioritize:
1. NBA / official team information
2. ESPN
3. other credible sports reporting

Do not invent information.

If no relevant recent information is found, clearly say so.

For every relevant item include:
- player name
- event
- status if available
- details
- publication time if available
- source
"""

        def run_search():
            return self.client.responses.create(
                model=settings.openai_model,

                tools=[
                    {
                        "type": "web_search",

                        # 先使用较可信来源。
                        # 后面可以继续扩展。
                        "filters": {
                            "allowed_domains": [
                                "nba.com",
                                "espn.com",
                            ]
                        },
                    }
                ],

                # 这个任务必须真的搜索，
                # 不能让模型选择跳过 web search。
                tool_choice="required",

                input=prompt,
            )

        response = await asyncio.to_thread(
            run_search
        )

        sources = self._extract_sources(
            response
        )

        return {
            "player_name": player_name,
            "team_name": team_name,
            "lookback_hours": lookback_hours,

            "searched_at": (
                now.isoformat()
            ),

            "text": (
                response.output_text
            ),

            "sources": sources,

            "response_id": (
                response.id
            ),
        }


web_news_fetcher = WebNewsFetcher()
