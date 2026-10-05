"""Time-aware news evidence, including unresolved long-term events.

State is reconstructed from append-only NewsEvent rows; no schema migration.
An old report without a follow-up is evidence requiring review, not proof that
the player is still out. Explicit updates must refer to the same entity/scope.
"""
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, or_, select

from app.core.config import settings
from app.models.news import NewsEvent


PERSISTENT_TYPES = {
    "injury": "injury", "long_term_injury": "injury",
    "season_ending_injury": "injury", "ruled_out": "injury",
    "suspension": "suspension", "trade": "trade",
    "role_change": "role", "minutes_restriction": "minutes",
    "coach_change": "coach",
}
RESOLUTION_TYPES = {
    "return_from_injury": "injury", "injury_resolved": "injury",
    "available": "injury", "return_from_suspension": "suspension",
    "suspension_ended": "suspension", "minutes_restriction_lifted": "minutes",
    "role_change_ended": "role", "coach_change_ended": "coach",
    "trade_cancelled": "trade",
}
STATE_TYPES = tuple(PERSISTENT_TYPES) + tuple(RESOLUTION_TYPES)


@dataclass(frozen=True)
class Evidence:
    event: NewsEvent
    persistent: bool
    needs_review: bool


def select_evidence(events, *, as_of, team_ids=(), player_ids=(), game_id=None,
                    lookback_hours=None, review_after_hours=None):
    """Resolve state before limiting recent news; never consume future reports."""
    lookback_hours = settings.news_recent_hours if lookback_hours is None else lookback_hours
    review_after_hours = (settings.news_review_after_hours if review_after_hours is None
                          else review_after_hours)
    teams, players = set(team_ids), set(player_ids)
    since = as_of - timedelta(hours=lookback_hours)
    review_since = as_of - timedelta(hours=review_after_hours)
    visible = sorted((e for e in events if e.published_at <= as_of and e.ingested_at <= as_of
                      and (e.game_id is None or e.game_id == game_id)),
                     key=lambda e: (e.published_at, e.ingested_at, str(e.id)))

    def relevant(e):
        return ((game_id is not None and e.game_id == game_id)
                or (e.team_id is not None and e.team_id in teams)
                or (e.player_id is not None and e.player_id in players))

    latest, ordinary = {}, []
    for e in visible:
        kind = e.event_type.lower().strip()
        family = PERSISTENT_TYPES.get(kind) or RESOLUTION_TYPES.get(kind)
        if family:
            # A player returning does not clear another player's injury. A
            # one-game availability report does not resolve a global report.
            # Reports about unnamed players cannot safely supersede one another.
            subject = ("player", e.player_id) if e.player_id else (
                ("team", e.team_id) if family == "coach" and e.team_id else ("event", e.id))
            key = (subject, family, e.game_id)
            resolved = (kind in RESOLUTION_TYPES
                        or (e.player_status or "").lower().strip() in {"resolved", "ended"}
                        or (family == "injury" and (e.player_status or "").lower().strip() == "available"))
            latest[key] = None if resolved else e
            if resolved and relevant(e) and e.published_at >= since:
                ordinary.append(e)
        elif relevant(e) and e.published_at >= since:
            ordinary.append(e)

    persistent = [Evidence(e, True, e.published_at < review_since
                           or (e.player_id is None and e.event_type.lower().strip() != "coach_change"))
                  for e in latest.values() if e is not None and relevant(e)]
    # The 30-item cap only applies to ordinary recent news, never long-term state.
    recent = [Evidence(e, False, False) for e in ordinary[-30:]]
    return sorted(persistent + recent, key=lambda item: (
        not item.persistent, -item.event.published_at.timestamp(), str(item.event.id)))


async def load_evidence(db, *, as_of, team_ids=(), player_ids=(), game_id=None,
                        lookback_hours=None):
    """Fetch related reports and follow-ups, including player updates after a trade."""
    conditions = []
    if team_ids:
        conditions.append(NewsEvent.team_id.in_(team_ids))
    if player_ids:
        conditions.append(NewsEvent.player_id.in_(player_ids))
    if game_id is not None:
        conditions.append(NewsEvent.game_id == game_id)
    if not conditions:
        return []
    visible = (NewsEvent.published_at <= as_of, NewsEvent.ingested_at <= as_of)
    recent_hours = settings.news_recent_hours if lookback_hours is None else lookback_hours
    news_scope = or_(NewsEvent.published_at >= as_of - timedelta(hours=recent_hours),
                     func.lower(func.trim(NewsEvent.event_type)).in_(STATE_TYPES))
    events = (await db.scalars(select(NewsEvent).where(
        *visible, news_scope, or_(*conditions),
    ))).all()
    # A return or subsequent trade can be recorded without the original team ID.
    linked_players = {e.player_id for e in events if e.player_id is not None}
    if linked_players:
        updates = (await db.scalars(select(NewsEvent).where(
            *visible, NewsEvent.player_id.in_(linked_players),
            func.lower(func.trim(NewsEvent.event_type)).in_(STATE_TYPES),
        ))).all()
        events = list({e.id: e for e in [*events, *updates]}.values())
    return select_evidence(events, as_of=as_of, team_ids=team_ids,
                           player_ids=player_ids, game_id=game_id,
                           lookback_hours=lookback_hours)


def evidence_data(item):
    e = item.event
    return {
        "id": str(e.id), "title": e.title, "event_type": e.event_type,
        "player_id": str(e.player_id) if e.player_id else None,
        "team_id": str(e.team_id) if e.team_id else None,
        "game_id": str(e.game_id) if e.game_id else None,
        "player_status": e.player_status, "source": e.source,
        "source_url": e.source_url, "published_at": e.published_at.isoformat(),
        "persistent": item.persistent, "needs_review": item.needs_review,
        "context_status": ("长期事件 · 待复核" if item.needs_review else
                           "长期事件 · 未记录解除" if item.persistent else "近期新闻"),
    }


async def archived_evidence(*, as_of, team_ids, game_id, player_ids=(), store=None):
    """Apply the same long-term retention to local collected facts and recoveries."""
    import asyncio
    from datetime import datetime
    from types import SimpleNamespace
    from uuid import UUID
    from app.news.archive import archive
    store = store or archive
    rows = await asyncio.to_thread(store.events)
    events, pending = [], []
    sources = {"nba_official": "NBA 官方报告", "espn_injuries": "ESPN 伤病列表", "espn_rss": "ESPN RSS 新闻摘要"}
    for row in rows:
        d = row["data"]
        published = datetime.fromisoformat(d.get("published_at", row["published_at"]))
        observed = datetime.fromisoformat(row["observed_at"])
        if published > as_of or observed > as_of or row["review"] == "rejected":
            continue
        if ((d.get("team_id") not in {str(t) for t in team_ids} and d.get("player_id") not in {str(p) for p in player_ids})
                or d.get("game_id") not in {None, str(game_id)}):
            continue
        if d.get("event_type") == "report_coverage":
            continue
        event = SimpleNamespace(id=row["id"], player_id=UUID(d["player_id"]) if d.get("player_id") else None,
            team_id=UUID(d["team_id"]) if d.get("team_id") else None, game_id=UUID(d["game_id"]) if d.get("game_id") else None,
            event_type=d["event_type"], player_status=d.get("player_status"),
            title=f"{d.get('player_name') or d.get('team_name') or '未匹配实体'} · {d.get('reason') or row['title']}",
            source=sources.get(row["provider"], row["provider"]), source_url=row["url"],
            published_at=published, ingested_at=observed)
        if row["review"] == "approved" and row["reviewed_at"] and datetime.fromisoformat(row["reviewed_at"]) <= as_of:
            # Review is also an information availability constraint.
            event.ingested_at = max(observed, datetime.fromisoformat(row["reviewed_at"]))
            events.append(event)
        elif published >= as_of-timedelta(hours=settings.news_recent_hours):
            pending.append({**evidence_data(Evidence(event, False, True)), "context_status": "待审核 · 未用于预测特征"})
    selected = select_evidence(events, as_of=as_of, team_ids=team_ids, game_id=game_id, player_ids=player_ids)
    return [evidence_data(e) for e in selected] + pending
