from fastapi import APIRouter, HTTPException, status

from app.schemas.ai import AITestRequest, AITestResponse
from app.services.openai_service import (
    OpenAIServiceError,
    openai_service,
)


router = APIRouter()


@router.post(
    "/test",
    response_model=AITestResponse,
)
async def test_openai(
    payload: AITestRequest,
) -> AITestResponse:
    """
    测试：

    FastAPI
        ↓
    OpenAI Service
        ↓
    OpenAI API

    这里只是测试 API。
    还不是 Forecaster Agent。
    """

    try:
        result = await openai_service.generate_text(
            instructions=(
                "You are an AI component inside an NBA analytics platform. "
                "Answer clearly and concisely. "
                "Do not invent real-time NBA information."
            ),
            user_input=payload.message,
            max_output_tokens=1500,
        )

        return AITestResponse(**result)

    except OpenAIServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": exc.code,
                "message": exc.message,
            },
        ) from exc
    