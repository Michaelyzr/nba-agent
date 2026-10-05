"""Idempotent collection and cancellable polling; errors do not erase state."""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
from sqlalchemy import select

from app.core.config import settings
from app.models.game import Game
from app.models.player import Player
from app.models.team import Team
from app.news.archive import archive, utcnow
from app.news.extraction import EntityMapper, extract_document
from app.news.sources import (ESPN_INJURIES, ESPN_RSS, download, latest_official,
                              parse_espn_injuries, parse_pdf, parse_report, parse_rss, report_time)

logger = logging.getLogger(__name__)


async def mapper_for(db):
    teams = (await db.scalars(select(Team))).all()
    players = (await db.scalars(select(Player))).all()
    games = (await db.scalars(select(Game).where(Game.tipoff_time >= utcnow()-timedelta(days=2)))).all()
    return EntityMapper(teams, players, games)


def record_events(store, document_id, events, mapper, provider, *, automatic):
    count = 0
    for event in events:
        data = mapper({**event, "provider": provider})
        # News stays pending even when its entities and quote match.
        approved = automatic and not data["match_error"]
        _, created = store.add_event(document_id, data, "approved" if approved else "pending")
        count += created
    return count


async def import_report(db, url, store=None, mapper=None):
    store = store or archive
    when = report_time(url)
    if when > utcnow():
        raise ValueError("报告发布时间在未来")
    mapper = mapper or await mapper_for(db)
    content = await download(url)
    text = await asyncio.to_thread(parse_pdf, content)
    events = parse_report(text, mapper.teams)
    if not events:
        raise ValueError("PDF 未解析出完整的日期、对阵、球队、球员与状态；未写入任何伤病断言")
    doc_id, created = await asyncio.to_thread(store.document, url=url, provider="nba_official",
                                             published_at=when.isoformat(), title="NBA 官方伤病报告", body=text)
    # Keep the original PDF for audit, named by content hash, all inside project.
    path = store.path.parent / "reports" / f"{doc_id}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        await asyncio.to_thread(path.write_bytes, content)
    rows = [{**e, "published_at": when.isoformat()} for e in events]
    count = await asyncio.to_thread(record_events, store, doc_id, rows, mapper, "nba_official", automatic=True)
    return {"documents_added": int(created), "events_added": count, "parsed": len(events), "url": url}


async def extract_pending(db, store=None, mapper=None):
    store = store or archive
    if not settings.openai_api_key:
        raise ValueError("请配置 OPENAI_API_KEY 后运行新闻提取")
    mapper = mapper or await mapper_for(db)
    docs = await asyncio.to_thread(store.unextracted, settings.news_extract_limit)
    results = {"processed": 0, "events_added": 0, "errors": []}
    for document in docs:
        try:
            # Each call has a hard wall-clock limit, including retry time.
            rows = await asyncio.wait_for(extract_document(document), timeout=80)
            results["events_added"] += await asyncio.to_thread(
                record_events, store, document["id"], rows, mapper, "espn_rss", automatic=False)
            await asyncio.to_thread(store.mark_extracted, document["id"])
            results["processed"] += 1
        except Exception as exc:
            # Do not log vendor errors containing request content or secrets.
            results["errors"].append({"document_id": document["id"], "error": type(exc).__name__,
                                       "detail": str(exc) if isinstance(exc, ValueError) else "模型请求未完成，可重试"})
    return results


async def extract_task(db, store=None):
    store = store or archive; owner = str(uuid4())
    if not await asyncio.to_thread(store.lease, owner):
        raise ValueError("采集或提取任务正在运行，请稍后重试")
    try:
        return await extract_pending(db, store)
    finally:
        await asyncio.to_thread(store.release, owner)


async def collect(db, *, store=None, season=None, extract=False):
    store = store or archive
    owner = str(uuid4())
    if not await asyncio.to_thread(store.lease, owner):
        raise ValueError("采集任务正在运行，请稍后刷新")
    now = utcnow()
    result = {"started_at": now.isoformat(), "sources": {}, "events_added": 0, "documents_added": 0}
    try:
        mapper = await mapper_for(db)
        await asyncio.to_thread(store.rematch, mapper)
        outcomes = await asyncio.gather(download(ESPN_INJURIES), download(ESPN_RSS),
                                       latest_official(season or settings.nba_season, now), return_exceptions=True)
        for provider, outcome in zip(("espn_injuries", "espn_rss", "nba_official"), outcomes):
            try:
                if isinstance(outcome, Exception):
                    raise outcome
                if provider == "nba_official":
                    imported = await import_report(db, outcome, store, mapper)
                    result["events_added"] += imported["events_added"]
                    result["documents_added"] += imported["documents_added"]
                    result["sources"][provider] = {"status": "ok", **imported}
                elif provider == "espn_injuries":
                    payload = json.loads(outcome)
                    rows = parse_espn_injuries(payload)
                    identity, created = await asyncio.to_thread(store.document,
                        url=ESPN_INJURIES, provider=provider, published_at=now.isoformat(),
                        title="ESPN NBA 伤病列表", body=outcome.decode())
                    count = await asyncio.to_thread(record_events, store, identity, rows, mapper, provider, automatic=True)
                    result["documents_added"] += int(created); result["events_added"] += count
                    result["sources"][provider] = {"status": "ok", "parsed": len(rows), "events_added": count}
                else:
                    rows = [d for d in parse_rss(outcome) if 0 <=
                            (now-datetime.fromisoformat(d["published_at"])).total_seconds()
                            <= settings.news_recent_hours*3600]
                    for doc in rows:
                        _, created = await asyncio.to_thread(store.document, **doc)
                        result["documents_added"] += int(created)
                    result["sources"][provider] = {"status": "ok", "parsed": len(rows), "scope": "RSS 原文摘要"}
            except Exception as exc:
                status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                result["sources"][provider] = {"status": "error", "error": type(exc).__name__,
                    "http_status": status_code, "detail": str(exc) if isinstance(exc, ValueError) else "来源请求失败，不代表没有新闻或伤病"}
        if extract:
            result["extraction"] = await extract_pending(db, store, mapper)
        from app.intelligence.injury import capture_upcoming
        result["snapshots"] = await capture_upcoming(db, store)
        return result
    except Exception as exc:
        result["error"] = type(exc).__name__
        raise
    finally:
        result["finished_at"] = utcnow().isoformat()
        await asyncio.to_thread(store.put, "last_collection", result)
        await asyncio.to_thread(store.release, owner)


async def poll_news():
    from app.db.session import AsyncSessionLocal
    # Let API startup complete first. Polling never writes the PostgreSQL DB.
    await asyncio.sleep(5)
    while True:
        try:
            preferences = await asyncio.to_thread(archive.preferences)
            last = await asyncio.to_thread(archive.get, "last_collection", {})
            due = not last or (utcnow()-datetime.fromisoformat(last["finished_at"])).total_seconds() >= settings.news_poll_seconds
            if preferences["auto_collect"] and due:
                async with AsyncSessionLocal() as db:
                    await collect(db, extract=preferences["auto_extract"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("News collector unavailable: %s", type(exc).__name__)
        await asyncio.sleep(min(60, settings.news_poll_seconds))
