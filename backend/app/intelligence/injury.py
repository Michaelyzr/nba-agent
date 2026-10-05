"""Evidence -> availability exposure -> validated statistical correction.

Minutes and scoring exposure are descriptive, not a causal point spread. No LLM
or hand-written probability coefficient is used. Only prospectively archived
pre-game features can train the optional residual logistic model.
"""
import asyncio
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select

from app.core.config import settings
from app.models.game import Game
from app.models.player import Player
from app.models.stats import PlayerGameStat
from app.news.archive import archive, utcnow

FEATURES = ["out_minutes_diff", "uncertain_minutes_diff", "limited_minutes_diff", "out_points_diff"]
FEATURE_VERSION = "injury-exposure-v1"


def active_events(rows, game, as_of, player_ids=()):
    """Pick a game-specific report or global update per player/state family."""
    visible = []
    for row in rows:
        e = row["data"]
        if row["review"] != "approved" or not e.get("player_id") or e.get("match_error"):
            continue
        times = [e.get("published_at", row["published_at"]), row["observed_at"], row["reviewed_at"]]
        if any(not t or datetime.fromisoformat(t) > as_of for t in times):
            continue
        if e.get("game_id") not in {None, str(game.id)}:
            continue
        # Never apply an unmatched one-game report as a global injury.
        if e.get("game_date") and not e.get("game_id"):
            continue
        if (e.get("team_id") not in {str(game.home_team_id), str(game.away_team_id)}
                and e.get("player_id") not in player_ids):
            continue
        kind = e["event_type"]
        family = ("minutes" if kind in {"minutes_restriction", "minutes_restriction_lifted"}
                  else "suspension" if kind in {"suspension", "suspension_ended"}
                  else "injury" if kind in {"injury", "available"}
                  else None)
        if family is None:
            continue
        visible.append({**row, "family": family})
    grouped = defaultdict(list)
    for row in visible:
        grouped[(row["data"]["player_id"], row["family"])].append(row)
    result = []
    for reports in grouped.values():
        # Newer facts win. On the same timestamp a matched official game report
        # outranks a global secondary list. A stale official row cannot suppress
        # a later explicit recovery update.
        reports.sort(key=lambda r: (r["data"].get("published_at", r["published_at"]),
                                   r["provider"] == "nba_official", bool(r["data"].get("game_id")),
                                   r["observed_at"], r["id"]))
        latest = reports[-1]
        if latest["data"]["event_type"] in {"minutes_restriction_lifted", "suspension_ended"}:
            continue
        when = datetime.fromisoformat(latest["data"].get("published_at", latest["published_at"]))
        max_hours = settings.injury_status_max_hours if latest["family"] == "injury" else settings.news_review_after_hours
        stale = (as_of-when).total_seconds() > max_hours*3600
        # A later authoritative report resolves earlier secondary uncertainty.
        conflict = latest["provider"] != "nba_official" and any(
            abs((when-datetime.fromisoformat(r["data"].get("published_at", r["published_at"]))).total_seconds()) < 12*3600
            and r["provider"] != latest["provider"]
            and r["data"].get("player_status") != latest["data"].get("player_status")
            for r in reports[:-1])
        result.append({**latest, "stale": stale, "conflict": conflict})
    return result


async def exposures(db, game, *, as_of=None, store=None):
    store = store or archive; as_of = as_of or utcnow()
    players = (await db.scalars(select(Player).where(Player.current_team_id.in_(
        [game.home_team_id, game.away_team_id])))).all()
    by_id = {str(p.id): p for p in players}
    games = (await db.scalars(select(Game).where(Game.status == "final",
        Game.season_type.in_(["Regular Season", "Playoffs"]),
        Game.tipoff_time <= as_of-timedelta(hours=6),
        Game.tipoff_time >= as_of-timedelta(days=180)))).all()
    past_ids = [g.id for g in games]
    stats = (await db.scalars(select(PlayerGameStat).where(PlayerGameStat.game_id.in_(past_ids),
        PlayerGameStat.player_id.in_([p.id for p in players]), PlayerGameStat.created_at <= as_of,
        PlayerGameStat.updated_at <= as_of))).all() if past_ids and players else []
    dates = {g.id: g.tipoff_time for g in games}
    history = defaultdict(list)
    for s in stats:
        player = by_id.get(str(s.player_id))
        if (player and s.team_id == player.current_team_id and s.minutes is not None
                and math.isfinite(s.minutes) and 0 < s.minutes <= 65):
            history[str(s.player_id)].append(s)
    summaries = {}
    for identity, rows in history.items():
        rows.sort(key=lambda s: dates[s.game_id])
        rows = rows[-10:]
        # Played-game average: recent absences must not silently redefine a
        # player's usual minutes as zero. It still is not replacement value.
        scoring = [r.points for r in rows if r.points is not None and math.isfinite(r.points) and r.points >= 0]
        summaries[identity] = {"games": len(rows), "minutes": sum(r.minutes for r in rows)/len(rows),
                              "points": sum(scoring)/len(scoring) if len(scoring) >= 5 else None}
    archived = await asyncio.to_thread(store.events)
    events = active_events(archived, game, as_of, set(by_id))
    combined = defaultdict(dict)
    for e in events:
        combined[e["data"]["player_id"]][e["family"]] = e
    totals = {side: {"out_minutes": 0., "uncertain_minutes": 0., "limited_minutes": 0.,
                     "out_points": 0., "missing_stats": 0, "review_required": 0} for side in ("home", "away")}
    details = []
    for identity, families in combined.items():
        player = by_id.get(identity)
        if not player:
            continue
        side = "home" if player.current_team_id == game.home_team_id else "away"
        for report in families.values():
            if report["data"].get("team_id") != str(player.current_team_id):
                report["stale"] = True
        injury, restriction = families.get("injury"), families.get("minutes")
        if injury and not injury["data"].get("game_id") and game.tipoff_time > as_of+timedelta(hours=48):
            injury["stale"] = True
        if injury and injury["provider"] == "nba_official":
            # Removal from a newer complete report is a follow-up requiring
            # review, not proof the old Out status still applies or of recovery.
            later_reports = [e for e in archived if e["provider"] == "nba_official" and e["review"] == "approved"
                and e["data"].get("event_type") == "report_coverage"
                and e["data"].get("team_id") == injury["data"].get("team_id")
                and e["data"].get("game_id") == str(game.id)
                and e["data"]["published_at"] > injury["data"]["published_at"]
                and e["reviewed_at"] and datetime.fromisoformat(e["reviewed_at"]) <= as_of
                and datetime.fromisoformat(e["observed_at"]) <= as_of]
            if later_reports:
                injury["stale"] = True
        status = injury["data"].get("player_status", "unknown") if injury else "unknown"
        if families.get("suspension"):
            status = "out"
        review = any(r["stale"] or r["conflict"] for r in families.values())
        if review:
            status = "unknown"
        baseline = summaries.get(identity)
        enough = baseline is not None and baseline["games"] >= 5
        cap = restriction["data"].get("minutes_limit") if restriction else None
        if restriction and restriction["data"]["event_type"] == "minutes_restriction_lifted":
            cap = None
        missing_limit = bool(restriction and restriction["data"]["event_type"] == "minutes_restriction" and cap is None)
        if missing_limit:
            review = True
        low, high = None, None
        if enough:
            minutes = min(48., baseline["minutes"])
            ceiling = min(minutes, cap) if cap is not None and not review else minutes
            if status == "out":
                low = high = 0.; totals[side]["out_minutes"] += minutes
                if baseline["points"] is not None:
                    totals[side]["out_points"] += baseline["points"]
                else:
                    totals[side]["missing_stats"] += 1
            elif status == "available":
                low = high = ceiling
                totals[side]["limited_minutes"] += minutes-ceiling
            else:
                low, high = 0., ceiling
                totals[side]["uncertain_minutes"] += minutes
        else:
            totals[side]["missing_stats"] += 1
        if review:
            totals[side]["review_required"] += 1
        details.append({"player_id": identity, "player_name": player.full_name, "side": side,
                        "status": status, "needs_review": review, "stats_games": baseline["games"] if baseline else 0,
                        "baseline_minutes": round(baseline["minutes"], 2) if baseline else None,
                        "baseline_points": round(baseline["points"], 2) if baseline and baseline["points"] is not None else None,
                        "minutes_range": [round(low, 2), round(high, 2)] if low is not None else None,
                        "minutes_limit": cap, "missing_minutes_limit": missing_limit,
                        "evidence_ids": [r["id"] for r in families.values()]})
    features = {name: (totals["home"][name.removesuffix("_diff")]-totals["away"][name.removesuffix("_diff")])/
                (120 if name == "out_points_diff" else 240) for name in FEATURES}
    # Absence of a report is not a known healthy lineup. Require a recent official
    # report for both teams before recording an eligible supervised sample.
    reports_by_team = {}
    for e in archived:
        d = e["data"]
        if (e["provider"] == "nba_official" and e["review"] == "approved"
                and d.get("event_type") == "report_coverage" and d.get("game_id") == str(game.id)
                and e["reviewed_at"] and datetime.fromisoformat(e["reviewed_at"]) <= as_of
                and datetime.fromisoformat(e["observed_at"]) <= as_of
                and 0 <= (as_of-datetime.fromisoformat(d["published_at"])).total_seconds() <= settings.injury_status_max_hours*3600):
            previous = reports_by_team.get(d["team_id"])
            if not previous or (d["published_at"], e["observed_at"]) > (previous["published_at"], previous["observed_at"]):
                reports_by_team[d["team_id"]] = {**d, "observed_at": e["observed_at"]}
    covered = {identity for identity, d in reports_by_team.items() if d.get("submitted", True)}
    # One unresolved or unmapped row in this game's report blocks completeness.
    unmatched = any(e["provider"] == "nba_official" and e["review"] == "pending"
                    and e["data"].get("event_type") != "role_change"
                    and e["data"].get("game_id") == str(game.id) for e in archived)
    coverage = {str(game.home_team_id), str(game.away_team_id)} <= covered
    eligible = (coverage and not unmatched and all(not t["missing_stats"] and not t["review_required"] for t in totals.values()))
    return {"version": FEATURE_VERSION, "as_of": as_of.isoformat(), "players": details,
            "teams": totals, "features": features, "official_both_teams": coverage,
            "training_eligible": eligible, "model_applied": False,
            "limitations": ["分钟区间是出场/缺阵情景范围，不是统计置信区间；没有给 questionable 指定固定出场概率。",
                            "近 10 场已出场数据估计通常分钟，至少 5 场；缺阵得分是进攻产出暴露，不是球队净损失。",
                            "未估计替补补偿、球员因果价值或完整阵容协同；旧报告、冲突和未注明的分钟限制须复核。",
                            "缺少双方新鲜官方报告时不生成可训练样本，也不启用伤病胜率修正。"]}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def losses(p, y):
    p = np.clip(p, 1e-8, 1-1e-8)
    return {"brier": float(np.mean((p-y)**2)),
            "log_loss": float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))}


def fit_model(rows, minimum=200):
    """Fixed L2 residual logistic fit with an untouched chronological holdout.

    Offset = logit(Elo); coefficients must explain information beyond Elo, not
    count full player value again. No tuning on the holdout, no random splitting.
    """
    rows = sorted(rows, key=lambda r: (r["tipoff_time"], r["game_id"]))
    if len(rows) < minimum:
        return {"status": "insufficient_data", "sample_size": len(rows), "minimum": minimum,
                "detail": "需要真实赛前采集并具有完整伤病特征的已结束比赛，不能用当前伤病倒填历史"}
    split = int(len(rows)*.8)
    # Purge overlapping result availability at the validation boundary.
    boundary = datetime.fromisoformat(rows[split]["tipoff_time"])
    train = [r for r in rows[:split] if datetime.fromisoformat(r["tipoff_time"])+timedelta(hours=6) < boundary]
    validation = rows[split:]
    if len(train) < minimum*.6 or min(sum(r["home_win"] for r in train), sum(1-r["home_win"] for r in train)) < 20:
        return {"status": "insufficient_variation", "sample_size": len(rows), "detail": "训练集赛果或时间分布不足"}
    def arrays(group):
        x = np.array([[r["features"][key] for key in FEATURES] for r in group], dtype=float)
        y = np.array([r["home_win"] for r in group], dtype=float)
        p = np.clip(np.array([r["base_probability"] for r in group]), 1e-6, 1-1e-6)
        return x, y, np.log(p/(1-p)), p
    x, y, offset, _ = arrays(train)
    if not np.isfinite(x).all() or not np.any(np.std(x, axis=0) > .001):
        return {"status": "insufficient_variation", "sample_size": len(rows), "detail": "伤病特征缺少变化"}
    scale = np.maximum(np.std(x, axis=0), .02); z = x/scale
    coef = np.zeros(len(FEATURES)); ridge = .05
    for _ in range(2000):
        p = sigmoid(offset+z@coef)
        gradient = z.T@(p-y)/len(y)+ridge*coef
        hessian = (z.T*(p*(1-p)))@z/len(y)+ridge*np.eye(len(FEATURES))
        step = np.linalg.solve(hessian, gradient)
        coef -= step
        if np.linalg.norm(step) < 1e-8:
            break
    vx, vy, voffset, vp = arrays(validation)
    adjusted = sigmoid(voffset+(vx/scale)@coef)
    base_metrics, new_metrics = losses(vp, vy), losses(adjusted, vy)
    approved = new_metrics["brier"] < base_metrics["brier"] and new_metrics["log_loss"] < base_metrics["log_loss"]
    return {"status": "validated" if approved else "rejected", "version": FEATURE_VERSION,
            "sample_size": len(rows), "train_games": len(train), "validation_games": len(validation),
            "coefficients": (coef/scale).tolist(), "feature_names": FEATURES,
            "feature_bounds": {key: [float(np.min(x[:, i])), float(np.max(x[:, i]))] for i, key in enumerate(FEATURES)},
            "training_end": train[-1]["tipoff_time"], "validation_end": validation[-1]["tipoff_time"],
            "baseline_metrics": base_metrics, "model_metrics": new_metrics,
            "method": "80/20 chronological holdout; 6h purge; Elo logit offset; fixed L2=0.05",
            "calibrated": False, "detail": "验证优于 Elo 基线后启用，仍未做独立概率校准" if approved else "验证未同时改善 Brier 和 Log Loss，继续使用 Elo"}


async def train(db, store=None):
    store = store or archive
    snapshots = await asyncio.to_thread(store.snapshots)
    games = (await db.scalars(select(Game).where(Game.status == "final",
        Game.tipoff_time <= utcnow()-timedelta(hours=6)))).all()
    by_id = {str(g.id): g for g in games}
    latest = {}
    for snapshot in snapshots:
        g = by_id.get(snapshot["game_id"]); data = snapshot["data"]
        if (g and g.season_type in {"Regular Season", "Playoffs"} and g.home_score is not None
                and g.away_score is not None and g.home_score != g.away_score
                and datetime.fromisoformat(snapshot["as_of"]) < g.tipoff_time
                and data.get("version") == FEATURE_VERSION and data.get("training_eligible")):
            latest[str(g.id)] = {"game_id": str(g.id), "tipoff_time": g.tipoff_time.isoformat(),
                                "home_win": int(g.home_score > g.away_score), **data}
    result = await asyncio.to_thread(fit_model, list(latest.values()), settings.injury_training_min_games)
    result["trained_at"] = utcnow().isoformat()
    await asyncio.to_thread(store.put, "last_training", result)
    if result["status"] == "validated":
        await asyncio.to_thread(store.put, "injury_model", result)
    return result


def apply_model(base, exposure, model, as_of):
    if not model or model.get("status") != "validated" or not exposure["training_eligible"]:
        return base, "尚无通过验证的模型或本场官方伤病/统计数据不完整"
    if datetime.fromisoformat(model["trained_at"]) > as_of or model.get("version") != FEATURE_VERSION:
        return base, "模型时间或特征版本不匹配"
    for key in FEATURES:
        low, high = model["feature_bounds"][key]
        if not low-.01 <= exposure["features"][key] <= high+.01:
            return base, "伤病特征超出训练覆盖范围，回退 Elo"
    offset = math.log(base/(1-base))
    adjustment = sum(exposure["features"][key]*coef for key, coef in zip(FEATURES, model["coefficients"]))
    value = float(sigmoid(offset+adjustment))
    return value, "通过时间留出验证的伤病残差模型；未校准"


async def enrich(db, game, output, now, store=None):
    store = store or archive
    exposure = await exposures(db, game, as_of=now, store=store)
    if min(output["features"]["home_history_games"], output["features"]["away_history_games"]) < settings.min_team_history:
        exposure["training_eligible"] = False
    model = await asyncio.to_thread(store.get, "injury_model")
    base = output["home_probability"]
    probability, reason = apply_model(base, exposure, model, now)
    exposure.update({"base_home_probability": base, "adjusted_home_probability": probability,
                     "probability_change": probability-base, "model_applied": probability != base,
                     "model_reason": reason})
    output["injury_impact"] = exposure
    output["features"]["injury"] = exposure["features"]
    output["home_probability"] = probability; output["away_probability"] = 1-probability
    output["limitations"][-1] = "近期胜率与休息天数仅展示；伤病修正仅在模型验证通过且本场数据完整时启用。"
    if probability != base:
        output["model_version"] = "elo+injury-residual-v1"
        output["factors"].append({"name": "已验证伤病残差模型", "value": probability-base})
    await asyncio.to_thread(store.snapshot, str(game.id), now.isoformat(),
                            {**exposure, "base_probability": base})
    return exposure


async def capture_upcoming(db, store=None):
    """Automatic local-only pre-game snapshots; no PostgreSQL prediction writes."""
    from app.intelligence.engine import forecast
    from app.intelligence.service import record, parameters
    store = store or archive; now = utcnow()
    upcoming = (await db.scalars(select(Game).where(Game.status == "scheduled",
        Game.season_type.in_(["Regular Season", "Playoffs"]), Game.tipoff_time > now,
        Game.tipoff_time <= now+timedelta(hours=48)).order_by(Game.tipoff_time).limit(30))).all()
    if not upcoming:
        return {"captured": 0, "eligible": 0}
    games = (await db.scalars(select(Game).where(Game.status == "final",
        Game.tipoff_time <= now-timedelta(hours=6)))).all()
    history = [record(g) for g in games]
    count = 0
    for game in upcoming:
        output = forecast(history, record(game), now, **parameters())
        exposure = await enrich(db, game, output, now, store)
        count += exposure["training_eligible"]
    return {"captured": len(upcoming), "eligible": count}
