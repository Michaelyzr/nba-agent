from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    RateLimitError,
)

from app.core.config import settings


class OpenAIServiceError(Exception):
    """
    项目内部统一的 OpenAI API 异常。
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
    ) -> None:
        self.code = code
        self.message = message

        super().__init__(message)


class OpenAIService:
    """
    整个项目统一的 OpenAI API Service。

    后续：
    - Forecaster Agent
    - Market Grader / Trader Agent
    - Reviewer Agent

    全部通过这里调用 OpenAI。
    """

    def __init__(self) -> None:

        self._client = None

        self.default_model = settings.openai_model

    @property
    def client(self):
        if not settings.openai_api_key:
            raise OpenAIServiceError(code="OPENAI_NOT_CONFIGURED", message="Configure OPENAI_API_KEY in backend/.env to use AI features.")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )

        return self._client

    async def generate_text(
        self,
        *,
        user_input: str,
        instructions: str,
        model: str | None = None,
        max_output_tokens: int = 1500,
    ) -> dict[str, Any]:
        """
        通用文本生成接口。
        """

        selected_model = model or self.default_model

        try:
            response = await self.client.responses.create(
                model=selected_model,

                instructions=instructions,

                input=user_input,

                # GPT-5 是 reasoning model。
                # 使用 low，避免简单任务消耗大量 reasoning tokens。
                reasoning={
                    "effort": "low"
                },

                # reasoning token 和最终文本都算在这里。
                max_output_tokens=max_output_tokens,

                # 重要 Agent 状态以后存到自己的数据库。
                store=False,
            )

            usage = getattr(response, "usage", None)

            output_details = (
                getattr(usage, "output_tokens_details", None)
                if usage
                else None
            )

            reasoning_tokens = (
                getattr(output_details, "reasoning_tokens", None)
                if output_details
                else None
            )

            text = response.output_text or ""

            # 如果 API 调用成功，但没有最终文字，
            # 不应该静默返回一个空字符串。
            if not text.strip():

                incomplete_details = getattr(
                    response,
                    "incomplete_details",
                    None,
                )

                raise OpenAIServiceError(
                    code="OPENAI_EMPTY_OUTPUT",
                    message=(
                        "OpenAI returned no visible text. "
                        f"status={response.status}, "
                        f"incomplete_details={incomplete_details}, "
                        f"reasoning_tokens={reasoning_tokens}"
                    ),
                )

            return {
                "response_id": response.id,
                "model": selected_model,
                "text": text,
                "usage": {
                    "input_tokens": (
                        getattr(usage, "input_tokens", None)
                        if usage
                        else None
                    ),
                    "output_tokens": (
                        getattr(usage, "output_tokens", None)
                        if usage
                        else None
                    ),
                    "total_tokens": (
                        getattr(usage, "total_tokens", None)
                        if usage
                        else None
                    ),
                },
            }

        except OpenAIServiceError:
            raise

        except RateLimitError as exc:
            raise OpenAIServiceError(
                code="OPENAI_RATE_LIMIT",
                message="OpenAI API rate limit reached.",
            ) from exc

        except APIConnectionError as exc:
            raise OpenAIServiceError(
                code="OPENAI_CONNECTION_ERROR",
                message="Could not connect to OpenAI API.",
            ) from exc

        except APIStatusError as exc:
            raise OpenAIServiceError(
                code="OPENAI_API_ERROR",
                message=f"OpenAI API returned HTTP {exc.status_code}.",
            ) from exc

        except Exception as exc:
            raise OpenAIServiceError(
                code="OPENAI_UNKNOWN_ERROR",
                message=str(exc),
            ) from exc


openai_service = OpenAIService()
