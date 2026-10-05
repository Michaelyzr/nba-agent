from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.intelligence.engine import GameRecord, forecast, walk_forward
from app.intelligence.market import fetch_book
from app.intelligence.risk import evaluate
from app.models.game import Game
from app.models.intelligence import GamePrediction, GameSignal, PaperPosition
from app.models.market import MarketSnapshot
from app.news.context import archived_evidence, evidence_data, load_evidence
from app.models.team import Team


def parameters():
    return {"k": settings.elo_k, "home_advantage": settings.elo_home_advantage,
            "retention": settings.elo_season_retention}


def record(game):
    return GameRecord(str(game.id), str(game.home_team_id), str(game.away_team_id),
                      game.tipoff_time, game.season, game.home_score, game.away_score,
                      game.status, game.season_type)


def prediction_data(prediction):
    return {"id": str(prediction.id), "as_of": prediction.as_of.isoformat(), **prediction.output}


def snapshot_data(snapshot):
    return {"id": str(snapshot.id), "provider": snapshot.provider, "selection": snapshot.selection,
            "contract_id": snapshot.contract_id, "market_probability": snapshot.market_probability,
            "bid_price": snapshot.bid_price, "ask_price": snapshot.ask_price,
            "liquidity": snapshot.liquidity, "observed_at": snapshot.observed_at.isoformat()}


async def require_game(db, game_id):
    game = await db.get(Game, game_id)
    if game is None:
        raise ValueError("比赛不存在，请先同步 NBA 数据")
    return game


async def latest_snapshot(db, game_id, selection):
    return await db.scalar(select(MarketSnapshot).where(
        MarketSnapshot.game_id == game_id, MarketSnapshot.market_type == "game_winner",
        MarketSnapshot.selection == selection,
    ).order_by(MarketSnapshot.observed_at.desc(), MarketSnapshot.created_at.desc()).limit(1))


def check_signal(game, prediction, snapshot, selection, now):
    features = prediction.features
    p = prediction.home_probability if selection == "home" else 1 - prediction.home_probability
    decision = evaluate(probability=p, snapshot=snapshot,
                        history_count=min(features["home_history_games"], features["away_history_games"]),
                        game=game, now=now, settings=settings)
    if not 0 <= (now - prediction.as_of).total_seconds() <= settings.market_max_age_seconds:
        decision["status"] = "blocked"
        decision["reasons"].append("预测过期，请重新运行 Agent")
    return decision


async def run_game(db, game_id):
    now = datetime.now(timezone.utc)
    game = await require_game(db, game_id)
    if game.status != "scheduled" or game.tipoff_time <= now:
        raise ValueError("仅支持尚未开赛的 scheduled 比赛；历史比赛请使用回测")
    if game.season_type not in {"Regular Season", "Playoffs"}:
        raise ValueError("首版仅支持常规赛和季后赛，季前赛不能混入模型")
    history = (await db.scalars(select(Game).where(
        Game.status == "final", Game.tipoff_time <= now - timedelta(hours=6),
    ).order_by(Game.tipoff_time, Game.nba_game_id))).all()
    output = forecast([record(g) for g in history], record(game), now, **parameters())
    output["estimated_time_games"] = sum(bool(g.tipoff_time_estimated) for g in history)
    if output["estimated_time_games"]:
        output["limitations"].append(f"{output['estimated_time_games']} 场历史比赛使用日期估算的开赛时间。")
    from app.intelligence.injury import enrich
    local_archive = getattr(db, "news_archive", None)
    injury = await enrich(db, game, output, now, store=local_archive)
    events = await load_evidence(db, as_of=now, game_id=game.id,
                                 team_ids=[game.home_team_id, game.away_team_id])
    collected = await archived_evidence(as_of=now, game_id=game.id,
                                        team_ids=[game.home_team_id, game.away_team_id],
                                        player_ids=[UUID(p["player_id"]) for p in injury["players"]], store=local_archive)
    output["events"] = [evidence_data(item) for item in events] + collected
    output["news_policy"] = {"recent_hours": settings.news_recent_hours,
                             "review_after_hours": settings.news_review_after_hours,
                             "persistent_events": sum(item.persistent for item in events) + sum(e["persistent"] for e in collected)}
    output["pipeline"] = [
        {"step": "Data", "status": "complete", "detail": f"{len(history)} 场已结束比赛"},
        {"step": "News / Injury", "status": "features_ready" if injury["players"] else "context_only",
         "detail": f"{len(output['events'])} 条证据；{len(injury['players'])} 位球员状态；{injury['model_reason']}"},
        {"step": "Feature", "status": "complete", "detail": "Elo 实力 + 主场；跨赛季回归均值"},
        {"step": "Prediction", "status": "complete", "detail": f"{output['model_version']} · 未校准"},
    ]
    prediction = GamePrediction(game_id=game.id, as_of=now,
                                home_probability=output["home_probability"], model_version=output["model_version"],
                                features=output["features"], output=output)
    db.add(prediction)
    await db.flush()
    signals = []
    for selection in ("home", "away"):
        snapshot = await latest_snapshot(db, game.id, selection)
        decision = check_signal(game, prediction, snapshot, selection, now)
        signal = GameSignal(prediction_id=prediction.id, snapshot_id=snapshot.id if snapshot else None,
                            selection=selection, status=decision["status"], decision=decision)
        db.add(signal)
        await db.flush()
        signals.append({"id": str(signal.id), "selection": selection, **decision})
    prediction.output = {**output, "pipeline": output["pipeline"] + [
        {"step": "Market", "status": "complete", "detail": "读取已保存的两个胜者方向报价"},
        {"step": "Risk / Audit", "status": "complete", "detail": "保存信号或拒绝原因；不执行真实交易"},
    ]}
    await db.commit()
    return {"prediction": prediction_data(prediction), "signals": signals}


async def list_games(db, season=None, limit=100):
    now = datetime.now(timezone.utc)
    query = select(Game).where(Game.tipoff_time >= now, Game.status == "scheduled")
    if season:
        query = query.where(Game.season == season)
    games = (await db.scalars(query.order_by(Game.tipoff_time).limit(limit))).all()
    ids = [g.id for g in games]
    teams = {t.id: t for t in (await db.scalars(select(Team))).all()}
    predictions = (await db.scalars(select(GamePrediction).where(GamePrediction.game_id.in_(ids))
                                  .order_by(GamePrediction.as_of.desc()))).all() if ids else []
    markets = (await db.scalars(select(MarketSnapshot).where(MarketSnapshot.game_id.in_(ids),
                               MarketSnapshot.market_type == "game_winner")
                               .order_by(MarketSnapshot.observed_at.desc(), MarketSnapshot.created_at.desc()))).all() if ids else []
    latest, quotes = {}, {}
    for p in predictions:
        latest.setdefault(p.game_id, p)
    for m in markets:
        quotes.setdefault((m.game_id, m.selection), m)
    rows = []
    for g in games:
        p = latest.get(g.id)
        def team_data(team_id):
            t = teams.get(team_id)
            return {"id": str(team_id), "name": t.full_name if t else "Unknown",
                    "abbreviation": t.abbreviation if t else "?"}
        rows.append({"id": str(g.id), "nba_game_id": g.nba_game_id,
                     "tipoff_time": g.tipoff_time.isoformat(), "season": g.season,
                     "season_type": g.season_type, "status": g.status,
                     "home": team_data(g.home_team_id), "away": team_data(g.away_team_id),
                     "prediction": prediction_data(p) if p else None,
                     "markets": {s: snapshot_data(quotes[(g.id, s)]) for s in ("home", "away") if (g.id, s) in quotes},
                     "decisions": {s: check_signal(g, p, quotes.get((g.id, s)), s, now) for s in ("home", "away")} if p else {}})
    return rows


async def save_market(db, game_id, request):
    game = await require_game(db, game_id)
    now = datetime.now(timezone.utc)
    if game.status != "scheduled" or game.tipoff_time <= now:
        raise ValueError("只能为未开赛比赛保存报价")
    if request.provider == "polymarket":
        data = await fetch_book(request.token_id)
    else:
        data = {"bid_price": request.bid_price, "ask_price": request.ask_price,
                "market_probability": (request.bid_price + request.ask_price) / 2,
                "liquidity": request.liquidity, "observed_at": request.observed_at or now,
                "raw_data": {"mapping_confirmed": False}}
    if data["observed_at"] > now + timedelta(seconds=5):
        raise ValueError("报价时间不能在未来")
    snapshot = MarketSnapshot(game_id=game_id, provider=request.provider, market_type="game_winner",
                              contract_id=request.token_id or "manual", selection=request.selection, **data)
    if request.provider == "polymarket":
        snapshot.raw_data = {**snapshot.raw_data, "mapping_confirmed": True,
                             "mapping_method": "user_confirmed_game_winner_outcome"}
    db.add(snapshot)
    await db.commit()
    return snapshot_data(snapshot)


async def backtest(db, season=None):
    now = datetime.now(timezone.utc)
    # Prior seasons are warm-up data; scoring can be restricted to one season.
    games = (await db.scalars(select(Game).where(Game.status == "final",
                   Game.tipoff_time <= now - timedelta(hours=6)).order_by(Game.tipoff_time))).all()
    records = [record(g) for g in games]
    if season:
        records = [g for g in records if g.season <= season]
    result = walk_forward(records, min_history=settings.min_team_history, **parameters())
    if season:
        # Keep historical warm-up, recompute metrics only on requested-season rows.
        target_ids = {g.id for g in records if g.season == season}
        from app.intelligence.metrics import summarize
        scored = [r for r in result["rows"] if r["game_id"] in target_ids]
        result = summarize(scored, sum(g.eligible_result and g.season == season for g in records))
    result.update({"model_version": "elo-v1", "calibrated": False, "season": season,
                   "estimated_time_games": sum(bool(g.tipoff_time_estimated) for g in games),
                   "method": "chronological walk-forward; results available at tipoff + 6h",
                   "parameters": parameters(), "min_team_history": settings.min_team_history,
                   "limitations": ["只评估胜率，不计算交易收益。", "尚未拟合参数、训练校准器或引入伤病特征。"]})
    return result


def position_data(p):
    return {"id": str(p.id), "game_id": str(p.game_id), "selection": p.selection,
            "stake": p.stake, "entry_price": p.entry_price, "shares": p.shares,
            "status": p.status, "pnl": p.pnl, "created_at": p.created_at.isoformat()}


async def open_paper(db, request):
    # Serialize paper exposure checks across processes using a transaction lock.
    await db.execute(text("SELECT pg_advisory_xact_lock(73102401)"))
    signal = await db.get(GameSignal, request.signal_id)
    if signal is None:
        raise ValueError("模拟信号不存在")
    prediction = await db.get(GamePrediction, signal.prediction_id)
    game = await require_game(db, prediction.game_id)
    snapshot = await latest_snapshot(db, game.id, signal.selection)
    if snapshot is None or snapshot.id != signal.snapshot_id:
        raise ValueError("报价已变化，请重新运行 Agent")
    latest_prediction = await db.scalar(select(GamePrediction).where(GamePrediction.game_id == game.id)
                                       .order_by(GamePrediction.as_of.desc()).limit(1))
    if latest_prediction.id != prediction.id:
        raise ValueError("预测已更新，请使用最新信号")
    decision = check_signal(game, prediction, snapshot, signal.selection, datetime.now(timezone.utc))
    if decision["status"] != "paper_signal":
        raise ValueError("；".join(decision["reasons"]))
    if request.stake > settings.paper_max_stake:
        raise ValueError(f"单笔模拟金额上限为 {settings.paper_max_stake}")
    if request.stake > (snapshot.liquidity or 0):
        raise ValueError("模拟金额超过最优卖价档位深度")
    positions = (await db.scalars(select(PaperPosition))).all()
    if any(p.game_id == game.id and p.selection == signal.selection for p in positions):
        raise ValueError("该比赛方向已有模拟记录")
    if sum(p.stake for p in positions if p.status == "open") + request.stake > settings.paper_max_exposure:
        raise ValueError("超过总模拟敞口上限")
    p = PaperPosition(game_id=game.id, signal_id=signal.id, selection=signal.selection,
                      stake=request.stake, entry_price=decision["effective_price"],
                      shares=request.stake / decision["effective_price"], status="open")
    db.add(p)
    await db.commit()
    return position_data(p)


async def settle_paper(db):
    await db.execute(text("SELECT pg_advisory_xact_lock(73102401)"))
    rows = (await db.execute(select(PaperPosition, Game).join(Game, Game.id == PaperPosition.game_id)
                            .where(PaperPosition.status == "open", Game.status == "final"))).all()
    count = 0
    for position, game in rows:
        if not record(game).eligible_result:
            continue
        home_win = game.home_score > game.away_score
        won = home_win if position.selection == "home" else not home_win
        position.pnl = (position.shares if won else 0) - position.stake
        position.status = "settled"
        position.settled_at = datetime.now(timezone.utc)
        count += 1
    await db.commit()
    return {"settled": count, "method": "NBA final score research settlement; not exchange resolution"}
