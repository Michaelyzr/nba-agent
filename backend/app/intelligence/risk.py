"""Explainable checks for paper signals; price includes a configured cost buffer."""
import math


def evaluate(*, probability, snapshot, history_count, game, now, settings):
    reasons = []
    effective_price = None
    edge = None
    if game.status != "scheduled" or game.tipoff_time <= now:
        reasons.append("比赛已开赛或当前状态不允许赛前信号")
    if game.season_type not in {"Regular Season", "Playoffs"}:
        reasons.append("首版不支持季前赛或未知比赛类型")
    if history_count < settings.min_team_history:
        reasons.append(f"历史样本不足：双方至少需要 {settings.min_team_history} 场")
    if snapshot is None:
        reasons.append("尚无市场报价")
    else:
        values = [v for v in (snapshot.bid_price, snapshot.ask_price, snapshot.liquidity) if v is not None]
        if any(not math.isfinite(v) for v in values):
            return {"status": "blocked", "reasons": ["报价含非有限数值"],
                    "model_probability": probability, "effective_price": None,
                    "edge": None, "cost_buffer": settings.paper_cost_buffer}
        age = (now - snapshot.observed_at).total_seconds()
        if age < 0 or age > settings.market_max_age_seconds:
            reasons.append("报价过期或时间戳异常")
        if snapshot.provider != "polymarket":
            reasons.append("手动报价仅供比较，不触发模拟交易信号")
        if snapshot.ask_price is None or snapshot.bid_price is None:
            reasons.append("缺少可执行买卖报价")
        else:
            spread = snapshot.ask_price - snapshot.bid_price
            if not (0 < snapshot.bid_price <= snapshot.ask_price < 1):
                reasons.append("盘口价格异常")
            elif spread > settings.market_max_spread:
                reasons.append("买卖价差过大")
            effective_price = snapshot.ask_price + settings.paper_cost_buffer
            edge = probability - effective_price
            if edge < settings.signal_min_edge or effective_price >= 1:
                reasons.append("扣除成本缓冲后的概率差不足")
        if (snapshot.liquidity or 0) < settings.market_min_liquidity:
            reasons.append("卖盘报价深度不足")
    return {
        "status": "blocked" if reasons else "paper_signal",
        "reasons": reasons or ["通过模拟信号检查；模型尚未校准"],
        "model_probability": probability, "effective_price": effective_price,
        "edge": edge, "cost_buffer": settings.paper_cost_buffer,
    }
