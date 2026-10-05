from datetime import datetime

from pydantic import BaseModel, Field


class NewsEventCreate(BaseModel):
    """
    写入 news_events 的请求结构。

    目前先允许我们手动写入新闻进行测试。
    后面真实 News Provider 也会复用这个结构。
    """

    nba_player_id: int | None = None
    nba_team_id: int | None = None
    nba_game_id: str | None = None

    event_type: str = Field(
        min_length=1,
        max_length=50,
    )

    player_status: str | None = Field(
        default=None,
        max_length=50,
    )

    title: str = Field(
        min_length=1,
        max_length=500,
    )

    body: str | None = None

    source: str = Field(
        min_length=1,
        max_length=200,
    )

    source_url: str | None = None

    published_at: datetime


class NewsEventResponse(BaseModel):
    id: str

    event_type: str
    player_status: str | None

    title: str
    body: str | None

    source: str
    source_url: str | None

    published_at: datetime
    ingested_at: datetime