"""Bounded local facts. No network fetches, model-generated metrics or DB writes."""
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import func, or_, select

from app.core.config import settings
from app.models.game import Game
from app.models.player import Player
from app.models.stats import PlayerGameStat
from app.models.team import Team
from app.news.context import archived_evidence, evidence_data, load_evidence


FIELDS = ('minutes', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'plus_minus', 'true_shooting_pct', 'usage_pct',
          'offensive_rating', 'defensive_rating', 'pace')


def team_data(team):
    return dict(id=str(team.id), name=team.full_name, abbreviation=team.abbreviation,
                conference=team.conference, division=team.division, updated_at=team.updated_at.isoformat())


def player_data(player):
    return dict(id=str(player.id), name=player.full_name, position=player.position,
                team_id=str(player.current_team_id) if player.current_team_id else None,
                updated_at=player.updated_at.isoformat())


async def catalog(db):
    teams = (await db.scalars(select(Team).where(Team.active.is_(True)).order_by(Team.full_name))).all()
    players = (await db.scalars(select(Player).where(Player.active.is_(True)).order_by(Player.full_name))).all()
    return dict(teams=[team_data(t) for t in teams], players=[player_data(p) for p in players],
                model=settings.openai_model, configured=bool(settings.openai_api_key))


async def build_context(db, payload, *, now=None):
    now = now or datetime.now(timezone.utc)
    team = await db.get(Team, payload.team_id)
    if not team:
        raise HTTPException(404, '球队不存在，请先同步球队目录')
    player = await db.get(Player, payload.player_id) if payload.player_id else None
    if payload.player_id and (not player or player.current_team_id != team.id):
        raise HTTPException(422, '球员不属于所选当前阵容，请重新选择球员或同步阵容')
    roster = (await db.scalars(select(Player).where(Player.current_team_id == team.id,
        Player.active.is_(True)).order_by(Player.full_name).limit(60))).all()
    if player and player.id not in {p.id for p in roster}:
        roster.append(player)
    scope = dict(season=payload.season, season_type=payload.season_type)
    eligible = (Game.season == payload.season, Game.season_type == payload.season_type,
        Game.status == 'final', Game.tipoff_time <= now - timedelta(hours=6),
        Game.home_score >= 0, Game.away_score >= 0, Game.home_score != Game.away_score,
        Game.updated_at <= now)
    games = (await db.scalars(select(Game).where(*eligible,
        or_(Game.home_team_id == team.id, Game.away_team_id == team.id))
        .order_by(Game.tipoff_time.desc()).limit(200))).all()
    games_data = []
    for g in games:
        home = g.home_team_id == team.id
        scored, allowed = (g.home_score, g.away_score) if home else (g.away_score, g.home_score)
        games_data.append(dict(nba_game_id=g.nba_game_id, tipoff_time=g.tipoff_time.isoformat(),
            home=home, opponent_id=str(g.away_team_id if home else g.home_team_id),
            points=scored, opponent_points=allowed, win=scored > allowed))
    opponents = (await db.scalars(select(Team).where(Team.id.in_(
        {g.away_team_id if g.home_team_id == team.id else g.home_team_id for g in games})))).all() if games else []
    names = {str(t.id): t.full_name for t in opponents}
    for g in games_data:
        g['opponent'] = names.get(g['opponent_id'], '本地球队目录缺失')
    ids = [player.id] if player else [p.id for p in roster]
    # Group by historical team, so traded-player statistics are never silently
    # attributed to the player's current team. Null fields retain sample counts.
    columns = [PlayerGameStat.player_id, PlayerGameStat.team_id, func.count(),
        func.min(Game.tipoff_time), func.max(Game.tipoff_time)]
    for field in FIELDS:
        columns.extend([func.avg(getattr(PlayerGameStat, field)), func.count(getattr(PlayerGameStat, field))])
    rows = (await db.execute(select(*columns).join(Game, Game.id == PlayerGameStat.game_id).where(
        *eligible, PlayerGameStat.player_id.in_(ids), PlayerGameStat.minutes > 0,
        PlayerGameStat.updated_at <= now).group_by(PlayerGameStat.player_id, PlayerGameStat.team_id))).all() if ids else []
    all_teams = (await db.scalars(select(Team))).all()
    all_names = {t.id: t.full_name for t in all_teams}
    roster_names = {p.id: p.full_name for p in roster}
    stats = []
    for row in rows:
        values = {field: dict(mean=round(row[5+i*2], 2) if row[5+i*2] is not None else None,
                              samples=row[6+i*2]) for i, field in enumerate(FIELDS)}
        stats.append(dict(player_id=str(row[0]), player=roster_names.get(row[0]),
            historical_team=all_names.get(row[1]), historical_team_id=str(row[1]), played_games=row[2],
            first_game=row[3].isoformat(), last_game=row[4].isoformat(), averages=values))
    sql_events = await load_evidence(db, as_of=now, team_ids=[team.id], player_ids=ids)
    news = [evidence_data(e) for e in sql_events]
    news += await archived_evidence(as_of=now, team_ids=[team.id], player_ids=ids,
        game_id=None, store=getattr(db, 'news_archive', None))
    if player:
        news = [e for e in news if e['player_id'] in {None, str(player.id)}]
    news.sort(key=lambda e: e['published_at'], reverse=True)
    news = news[:40]
    wins = sum(g['win'] for g in games_data)
    available_scopes = []
    if player:
        coverage = (await db.execute(select(Game.season, Game.season_type, func.count(PlayerGameStat.id))
            .join(PlayerGameStat, PlayerGameStat.game_id == Game.id).where(
                PlayerGameStat.player_id == player.id, PlayerGameStat.minutes > 0,
                Game.status == 'final', Game.tipoff_time <= now-timedelta(hours=6),
                Game.home_score >= 0, Game.away_score >= 0, Game.home_score != Game.away_score,
                Game.season_type.in_(['Regular Season', 'Playoffs', 'Pre Season']),
                Game.updated_at <= now, PlayerGameStat.updated_at <= now)
            .group_by(Game.season, Game.season_type).order_by(Game.season.desc()).limit(12))).all()
        available_scopes = [dict(season=row[0], season_type=row[1], played_games=row[2]) for row in coverage]
    facts = dict(as_of=now.isoformat(), scope=scope, team=team_data(team),
        selected_player=player_data(player) if player else None, available_player_scopes=available_scopes,
        current_roster=[player_data(p) for p in roster],
        team_record=dict(games=len(games), wins=wins, losses=len(games)-wins,
            points_per_game=round(sum(g['points'] for g in games_data)/len(games),2) if games else None,
            opponent_points_per_game=round(sum(g['opponent_points'] for g in games_data)/len(games),2) if games else None),
        recent_games=games_data[:10], player_statistics=stats, news=news)
    sources = [dict(id='D1', title='NBA 球队目录与当前阵容', kind='local',
                    updated_at=max([team.updated_at, *[p.updated_at for p in roster]]).isoformat()),
        dict(id='D2', title=f'{payload.season} · {payload.season_type} 比赛与逐场统计', kind='local',
             updated_at=max((g.updated_at for g in games), default=None).isoformat() if games else None)]
    for i, event in enumerate(news, 3):
        event['citation'] = f'D{i}'
        sources.append(dict(id=f'D{i}', title=event['title'], kind='news', url=event['source_url'],
                            published_at=event['published_at'], status=event['context_status']))
    limitations = [('本次启用网页搜索补充证据；网页结果不自动写入本地统计或伤病状态，也不自动同步 NBA 数据。'
        if payload.web_search else '读取已同步的本地数据；本次对话不联网搜索，也不自动同步 NBA 数据。'),
        '当前阵容仅表示最后同步结果，不代表历史赛季阵容；updated_at 是本地记录更新时间。',
        '比赛记录最多 200 场，近况展示最近 10 场；统计按赛季、比赛类型与历史所属球队分组。',
        '仅使用已结束且开赛至少 6 小时的比赛；6 小时是赛果可用时间的保守估计。',
        '统计为已出场比赛的逐场均值（包括效率/命中率字段均值，不是累计加权值）；空值不作零值，字段样本数不同。',
        '新闻为最多 40 条相关近期/长期状态证据；旧报告、待审核事件不确认当前伤病，未出现不代表健康。',
        '模型评价和建议为定性分析，不更改 Elo 胜率，不提供未经验证的概率修正。']
    if not games: limitations.append('所选赛季/比赛类型没有已同步的可用赛果；本地数据不支持该范围战绩或走势。')
    if not stats: limitations.append('缺少所选范围的本地球员逐场统计；若用网页补充，需注明来源、赛季及比赛类型，不能当成本地样本。')
    facts['limitations'] = limitations
    return dict(facts=facts, meta=dict(as_of=now.isoformat(), model=settings.openai_model,
        team=team_data(team), player=player_data(player) if player else None, scope=scope,
        roster_count=len(roster), game_count=len(games), stats_count=len(stats),
        web_search=dict(enabled=payload.web_search, status='pending' if payload.web_search else 'disabled'),
        sources=sources, limitations=limitations))
