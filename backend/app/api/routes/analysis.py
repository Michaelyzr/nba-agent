import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.context import build_context, catalog
from app.analysis.intent import resolve_scope
from app.analysis.service import model_events
from app.core.config import settings
from app.db.session import get_db
from app.schemas.analysis import AnalysisRequest

router = APIRouter()


@router.get('/catalog')
async def analysis_catalog(db: AsyncSession = Depends(get_db)):
    return await catalog(db)


@router.post('/chat')
async def chat(payload: AnalysisRequest, request: Request, db: AsyncSession = Depends(get_db)):
    if not settings.openai_api_key:
        raise HTTPException(503, '请在 backend/.env 配置 OPENAI_API_KEY 后重启后端。')
    payload, routing = await resolve_scope(db, payload)
    context = await build_context(db, payload)
    context["meta"]["routing"] = routing
    # Release the read transaction before a potentially slow model response.
    await db.rollback()

    async def events():
        yield 'data: ' + json.dumps(dict(type='meta', **context['meta']), ensure_ascii=False) + '\n\n'
        generator = model_events(payload, context)
        try:
            async for event in generator:
                if await request.is_disconnected():
                    break
                yield 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
        finally:
            await generator.aclose()

    return StreamingResponse(events(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
