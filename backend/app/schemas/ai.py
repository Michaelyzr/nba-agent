from pydantic import BaseModel, Field


class AITestRequest(BaseModel):
    """
    测试 OpenAI API 的请求数据。
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class AITestResponse(BaseModel):
    response_id: str
    model: str
    text: str
    usage: TokenUsage