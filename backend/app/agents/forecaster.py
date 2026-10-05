import asyncio
import json
import time
from datetime import datetime, timezone

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.agent_run import AgentRun
from app.models.forecast import Forecast
from app.models.player import Player
from app.prediction.service import prediction_service
from app.schemas.forecaster import ForecasterDecision
from app.services.context_service import context_service


class ForecasterAgent:
    """
    NBA Forecaster Agent。

    工作流程：

    Historical Stats
        ↓
    Baseline Prediction
        ↓
    Automatic News Context
        ↓
    Optional User Context
        ↓
    GPT Adjustment
        ↓
    Final Forecast
        ↓
    Forecast + AgentRun
    """

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if not settings.openai_api_key:
            raise ValueError("Configure OPENAI_API_KEY in backend/.env to use the player forecaster.")
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )
        return self._client

    # ========================================================
    # OpenAI
    # ========================================================

    async def _call_openai(
        self,
        *,
        player_name: str,
        baseline_minutes: float,
        baseline_points: float,
        minutes_std: float,
        points_std: float,
        context: str,
    ) -> tuple[ForecasterDecision, object]:
        """
        调用 OpenAI。

        LLM 不负责从零预测，
        只负责判断 baseline 是否需要调整。
        """

        system_prompt = """
You are an NBA forecasting adjustment agent.

You NEVER create forecasts from scratch.

A statistical forecasting system has already generated
baseline predictions using historical NBA game data.

Your job is to evaluate recent contextual information and
decide whether the baseline should be adjusted.

Context may include:

- injuries
- ruled-out players
- questionable status
- probable status
- minutes restrictions
- starting lineup changes
- teammate injuries
- role changes
- roster changes
- rest
- trades

RULES:

1. The statistical baseline is always the anchor.

2. Never invent injuries, news, roster changes,
   minutes restrictions, or player status.

3. Only use information provided in CONTEXT.

4. If the context does not provide credible evidence
   for an adjustment, keep adjustments near zero.

5. Be conservative.

6. Minutes adjustment must stay between:
   -8 and +8.

7. Points adjustment must stay between:
   -15 and +15.

8. Confidence represents confidence in the adjustment,
   NOT the probability that the player reaches the prediction.

9. A minutes restriction should usually have a direct
   impact on projected minutes and may also affect points.

10. Return JSON only.

Required JSON:

{
  "minutes_adjustment": 0.0,
  "points_adjustment": 0.0,
  "confidence": 0.5,
  "key_factors": [
    "factor"
  ],
  "reasoning": [
    "reason"
  ]
}
"""

        user_prompt = f"""
PLAYER:

{player_name}


STATISTICAL BASELINE:

Minutes:
{baseline_minutes:.2f}

Minutes standard deviation:
{minutes_std:.2f}

Points:
{baseline_points:.2f}

Points standard deviation:
{points_std:.2f}


RECENT CONTEXT:

{context}


TASK:

Determine whether the statistical baseline should be adjusted.

Return JSON only.
"""

        def run_request():
            return self.client.responses.create(
                model=settings.openai_model,
                instructions=system_prompt,
                input=user_prompt,
            )

        response = await asyncio.to_thread(
            run_request
        )

        raw_text = response.output_text

        # ====================================================
        # Clean JSON
        # ====================================================

        clean_text = raw_text.strip()

        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]

        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]

        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.strip()

        # ====================================================
        # JSON -> Pydantic
        # ====================================================

        parsed_json = json.loads(
            clean_text
        )

        decision = (
            ForecasterDecision.model_validate(
                parsed_json
            )
        )

        return decision, response

    # ========================================================
    # Baseline
    # ========================================================

    async def _get_baseline(
        self,
        session: AsyncSession,
        *,
        nba_player_id: int,
    ) -> dict:
        """
        获取指定球员下一场比赛的 baseline forecast。

        如果不存在 baseline，
        自动调用 PredictionService 创建。
        """

        # ----------------------------------------------------
        # Player
        # ----------------------------------------------------

        player = (
            await prediction_service
            .get_player_by_nba_id(
                session,
                nba_player_id=nba_player_id,
            )
        )

        # ----------------------------------------------------
        # Next Game
        # ----------------------------------------------------

        next_game = (
            await prediction_service
            .get_next_game(
                session,
                player=player,
            )
        )

        # ----------------------------------------------------
        # Minutes baseline
        # ----------------------------------------------------

        minutes_result = await session.execute(
            select(Forecast)
            .where(
                Forecast.game_id
                == next_game.id,

                Forecast.player_id
                == player.id,

                Forecast.forecast_type
                == "minutes",

                Forecast.model_name
                == "rolling_weighted_baseline",
            )
            .order_by(
                Forecast.created_at.desc()
            )
            .limit(1)
        )

        minutes_forecast = (
            minutes_result.scalar_one_or_none()
        )

        # ----------------------------------------------------
        # Points baseline
        # ----------------------------------------------------

        points_result = await session.execute(
            select(Forecast)
            .where(
                Forecast.game_id
                == next_game.id,

                Forecast.player_id
                == player.id,

                Forecast.forecast_type
                == "points",

                Forecast.model_name
                == "rolling_weighted_baseline",
            )
            .order_by(
                Forecast.created_at.desc()
            )
            .limit(1)
        )

        points_forecast = (
            points_result.scalar_one_or_none()
        )

        # ----------------------------------------------------
        # Baseline 不存在 -> 自动生成
        # ----------------------------------------------------

        if (
            minutes_forecast is None
            or points_forecast is None
        ):
            await (
                prediction_service
                .create_player_forecast(
                    session=session,
                    nba_player_id=nba_player_id,
                    history_games=10,
                )
            )

            return await self._get_baseline(
                session,
                nba_player_id=nba_player_id,
            )

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        minutes_distribution = (
            minutes_forecast.distribution
            or {}
        )

        points_distribution = (
            points_forecast.distribution
            or {}
        )

        return {
            "player": player,
            "game": next_game,

            "minutes_forecast": (
                minutes_forecast
            ),

            "points_forecast": (
                points_forecast
            ),

            "baseline_minutes": float(
                minutes_forecast.predicted_value
            ),

            "baseline_points": float(
                points_forecast.predicted_value
            ),

            "minutes_std": float(
                minutes_distribution.get(
                    "std",
                    0.0,
                )
            ),

            "points_std": float(
                points_distribution.get(
                    "std",
                    0.0,
                )
            ),
        }

    # ========================================================
    # Context
    # ========================================================

    async def _build_context(
        self,
        session: AsyncSession,
        *,
        player: Player,
        manual_context: str | None,
    ) -> dict:
        """
        自动构建 Agent Context。

        来源：

        1. news_events 自动 context
        2. 用户可选 manual context

        最终合并后交给 GPT。
        """

        # ====================================================
        # Automatic database context
        # ====================================================

        automatic_context = (
            await context_service
            .build_player_context(
                session,
                player=player,
            )
        )

        # ====================================================
        # Merge Context
        # ====================================================

        context_parts = [
            "AUTOMATIC DATABASE CONTEXT:",
            automatic_context,
        ]

        if (
            manual_context is not None
            and manual_context.strip()
        ):
            context_parts.extend(
                [
                    "",
                    "ADDITIONAL USER CONTEXT:",
                    manual_context.strip(),
                ]
            )

        combined_context = "\n".join(
            context_parts
        )

        return {
            "automatic": automatic_context,
            "manual": manual_context,
            "combined": combined_context,
        }

    # ========================================================
    # Run Agent
    # ========================================================

    async def run(
        self,
        session: AsyncSession,
        *,
        nba_player_id: int,
        context: str | None = None,
    ) -> dict:
        """
        执行完整 Forecaster Agent。

        Baseline
            ↓
        Automatic Context
            ↓
        GPT Adjustment
            ↓
        Guardrails
            ↓
        Final Forecast
            ↓
        Database
        """

        started_at = datetime.now(
            timezone.utc
        )

        timer_start = (
            time.perf_counter()
        )

        # ====================================================
        # Baseline
        # ====================================================

        baseline = await self._get_baseline(
            session,
            nba_player_id=nba_player_id,
        )

        player: Player = baseline["player"]
        game = baseline["game"]

        # ====================================================
        # Automatic Context
        # ====================================================

        context_data = await self._build_context(
            session,
            player=player,
            manual_context=context,
        )

        combined_context = (
            context_data["combined"]
        )

        # ====================================================
        # AgentRun
        # ====================================================

        agent_run = AgentRun(
            game_id=game.id,

            agent_name="forecaster",

            model_provider="openai",

            model_name=(
                settings.openai_model
            ),

            status="running",

            input_data={
                "nba_player_id": (
                    nba_player_id
                ),

                "player_name": (
                    player.full_name
                ),

                "baseline_minutes": (
                    baseline[
                        "baseline_minutes"
                    ]
                ),

                "baseline_points": (
                    baseline[
                        "baseline_points"
                    ]
                ),

                "minutes_std": (
                    baseline[
                        "minutes_std"
                    ]
                ),

                "points_std": (
                    baseline[
                        "points_std"
                    ]
                ),

                # 保存自动 context
                "automatic_context": (
                    context_data[
                        "automatic"
                    ]
                ),

                # 保存用户额外 context
                "manual_context": (
                    context_data[
                        "manual"
                    ]
                ),

                "combined_context": (
                    combined_context
                ),
            },

            output_data=None,

            error_message=None,

            latency_ms=None,

            started_at=started_at,

            completed_at=None,
        )

        session.add(agent_run)

        await session.flush()

        try:
            # ================================================
            # GPT Adjustment
            # ================================================

            decision, response = (
                await self._call_openai(
                    player_name=(
                        player.full_name
                    ),

                    baseline_minutes=(
                        baseline[
                            "baseline_minutes"
                        ]
                    ),

                    baseline_points=(
                        baseline[
                            "baseline_points"
                        ]
                    ),

                    minutes_std=(
                        baseline[
                            "minutes_std"
                        ]
                    ),

                    points_std=(
                        baseline[
                            "points_std"
                        ]
                    ),

                    context=(
                        combined_context
                    ),
                )
            )

            # ================================================
            # Hard Guardrails
            # ================================================

            minutes_adjustment = max(
                -8.0,
                min(
                    8.0,
                    decision.minutes_adjustment,
                ),
            )

            points_adjustment = max(
                -15.0,
                min(
                    15.0,
                    decision.points_adjustment,
                ),
            )

            baseline_minutes = (
                baseline[
                    "baseline_minutes"
                ]
            )

            baseline_points = (
                baseline[
                    "baseline_points"
                ]
            )

            # ================================================
            # Final Forecast
            # ================================================

            final_minutes = max(
                0.0,
                min(
                    48.0,
                    baseline_minutes
                    + minutes_adjustment,
                ),
            )

            final_points = max(
                0.0,
                baseline_points
                + points_adjustment,
            )

            # ================================================
            # Minutes Forecast
            # ================================================

            minutes_forecast = Forecast(
                game_id=game.id,

                player_id=player.id,

                forecast_type="minutes",

                predicted_value=(
                    final_minutes
                ),

                distribution={
                    "baseline": (
                        baseline_minutes
                    ),

                    "adjustment": (
                        minutes_adjustment
                    ),

                    "final": (
                        final_minutes
                    ),

                    "baseline_std": (
                        baseline[
                            "minutes_std"
                        ]
                    ),
                },

                confidence=(
                    decision.confidence
                ),

                model_name=(
                    "gpt_forecaster_agent"
                ),

                model_version="1.1",

                features={
                    "baseline_model": (
                        "rolling_weighted_baseline"
                    ),

                    "automatic_context": (
                        context_data[
                            "automatic"
                        ]
                    ),

                    "manual_context": (
                        context_data[
                            "manual"
                        ]
                    ),

                    "key_factors": (
                        decision.key_factors
                    ),
                },

                explanation=(
                    " | ".join(
                        decision.reasoning
                    )
                ),
            )

            # ================================================
            # Points Forecast
            # ================================================

            points_forecast = Forecast(
                game_id=game.id,

                player_id=player.id,

                forecast_type="points",

                predicted_value=(
                    final_points
                ),

                distribution={
                    "baseline": (
                        baseline_points
                    ),

                    "adjustment": (
                        points_adjustment
                    ),

                    "final": (
                        final_points
                    ),

                    "baseline_std": (
                        baseline[
                            "points_std"
                        ]
                    ),
                },

                confidence=(
                    decision.confidence
                ),

                model_name=(
                    "gpt_forecaster_agent"
                ),

                model_version="1.1",

                features={
                    "baseline_model": (
                        "rolling_weighted_baseline"
                    ),

                    "automatic_context": (
                        context_data[
                            "automatic"
                        ]
                    ),

                    "manual_context": (
                        context_data[
                            "manual"
                        ]
                    ),

                    "key_factors": (
                        decision.key_factors
                    ),
                },

                explanation=(
                    " | ".join(
                        decision.reasoning
                    )
                ),
            )

            session.add(
                minutes_forecast
            )

            session.add(
                points_forecast
            )

            await session.flush()

            # ================================================
            # Finish AgentRun
            # ================================================

            completed_at = datetime.now(
                timezone.utc
            )

            latency_ms = (
                (
                    time.perf_counter()
                    - timer_start
                )
                * 1000
            )

            usage = getattr(
                response,
                "usage",
                None,
            )

            output_data = {
                "response_id": (
                    response.id
                ),

                "baseline_minutes": (
                    baseline_minutes
                ),

                "minutes_adjustment": (
                    minutes_adjustment
                ),

                "final_minutes": (
                    final_minutes
                ),

                "baseline_points": (
                    baseline_points
                ),

                "points_adjustment": (
                    points_adjustment
                ),

                "final_points": (
                    final_points
                ),

                "confidence": (
                    decision.confidence
                ),

                "key_factors": (
                    decision.key_factors
                ),

                "reasoning": (
                    decision.reasoning
                ),

                "automatic_context": (
                    context_data[
                        "automatic"
                    ]
                ),
            }

            # ================================================
            # Token Usage
            # ================================================

            if usage is not None:
                output_data[
                    "usage"
                ] = {
                    "input_tokens": getattr(
                        usage,
                        "input_tokens",
                        None,
                    ),

                    "output_tokens": getattr(
                        usage,
                        "output_tokens",
                        None,
                    ),

                    "total_tokens": getattr(
                        usage,
                        "total_tokens",
                        None,
                    ),
                }

            # ================================================
            # AgentRun Complete
            # ================================================

            agent_run.status = (
                "completed"
            )

            agent_run.output_data = (
                output_data
            )

            agent_run.completed_at = (
                completed_at
            )

            agent_run.latency_ms = (
                latency_ms
            )

            await session.commit()

            # ================================================
            # API Response
            # ================================================

            return {
                "agent_run_id": str(
                    agent_run.id
                ),

                "player": {
                    "nba_player_id": (
                        player.nba_player_id
                    ),

                    "name": (
                        player.full_name
                    ),
                },

                "game": {
                    "nba_game_id": (
                        game.nba_game_id
                    ),

                    "tipoff_time": (
                        game.tipoff_time
                        .isoformat()
                    ),
                },

                # ============================================
                # Context
                # ============================================

                "context": {
                    "automatic": (
                        context_data[
                            "automatic"
                        ]
                    ),

                    "manual": (
                        context_data[
                            "manual"
                        ]
                    ),
                },

                # ============================================
                # Baseline
                # ============================================

                "baseline": {
                    "minutes": round(
                        baseline_minutes,
                        2,
                    ),

                    "points": round(
                        baseline_points,
                        2,
                    ),
                },

                # ============================================
                # Adjustment
                # ============================================

                "adjustment": {
                    "minutes": round(
                        minutes_adjustment,
                        2,
                    ),

                    "points": round(
                        points_adjustment,
                        2,
                    ),
                },

                # ============================================
                # Final
                # ============================================

                "final": {
                    "minutes": round(
                        final_minutes,
                        2,
                    ),

                    "points": round(
                        final_points,
                        2,
                    ),
                },

                "confidence": (
                    decision.confidence
                ),

                "key_factors": (
                    decision.key_factors
                ),

                "reasoning": (
                    decision.reasoning
                ),

                "model": (
                    settings.openai_model
                ),
            }

        except Exception as exc:

            # ================================================
            # Failure audit
            # ================================================

            agent_run.status = (
                "failed"
            )

            agent_run.error_message = (
                str(exc)
            )

            agent_run.completed_at = (
                datetime.now(
                    timezone.utc
                )
            )

            agent_run.latency_ms = (
                (
                    time.perf_counter()
                    - timer_start
                )
                * 1000
            )

            await session.commit()

            raise


forecaster_agent = ForecasterAgent()
