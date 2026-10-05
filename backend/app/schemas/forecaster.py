from pydantic import BaseModel, Field


class ForecasterRunRequest(BaseModel):
    """
    目前允许手动传额外 context。

    后续这里会自动接：
    - Injury news
    - Roster changes
    - Minutes restriction
    - Starting lineup
    - Back-to-back
    """

    context: str | None = None


class ForecasterDecision(BaseModel):
    """
    GPT Forecaster 必须输出的结构。
    """

    minutes_adjustment: float = Field(
        ge=-8.0,
        le=8.0,
    )

    points_adjustment: float = Field(
        ge=-15.0,
        le=15.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    key_factors: list[str]

    reasoning: list[str]