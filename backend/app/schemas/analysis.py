from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=12000)


class AnalysisRequest(BaseModel):
    team_id: UUID
    player_id: UUID | None = None
    season: str = Field(pattern=r'^20\d{2}-\d{2}$')
    season_type: Literal['Regular Season', 'Playoffs', 'Pre Season'] = 'Regular Season'
    mode: Literal['chat', 'team_report', 'player_report'] = 'chat'
    web_search: bool = False
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)

    @model_validator(mode='after')
    def validate_conversation(self):
        if self.messages[-1].role != 'user':
            raise ValueError('最后一条消息必须是用户问题')
        if sum(len(m.content) for m in self.messages) > 40000:
            raise ValueError('对话过长，请新建对话')
        if self.mode == 'team_report' and self.player_id:
            raise ValueError('生成球队报告时请选择整支球队')
        if self.mode == 'player_report' and not self.player_id:
            raise ValueError('生成球员报告前请选择球员')
        return self
