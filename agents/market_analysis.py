"""Read-only binary moneyline execution comparisons; never submits orders."""
import argparse
import json
import math
from pathlib import Path

from data_sources.inplay_types import utc


def _finite(value, name, lo=0, hi=None):
    value = float(value)
    if not math.isfinite(value) or value < lo or (hi is not None and value > hi):
        raise ValueError(f"invalid {name}")
    return value


def quote_problems(quote, snapshot, side, max_age=15):
    if not isinstance(quote, dict):
        return ["行情不是有效对象"]
    problems = []
    try:
        at = utc(snapshot["as_of"])
        for field in ("observed_at", "updated_at"):
            age = (at-utc(quote[field])).total_seconds()
            if not math.isfinite(age) or age < 0 or age > max_age:
                problems.append("行情晚于模型时刻或超过新鲜度限制")
        if str(quote["game_id"]) != str(snapshot["game_id"]):
            problems.append("比赛 ID 不匹配")
        if quote.get("kind") != "moneyline" or not quote.get("includes_overtime") or not quote.get("rules_verified"):
            problems.append("非含加时全场胜负或结算规则未核对")
        if not quote.get("active"):
            problems.append("市场暂停／关闭")
        if not quote.get("fee_verified"):
            problems.append("费用参数未经核验")
        if not quote.get("condition_id") or not quote.get("token_id") or quote.get("side") != side:
            problems.append("outcome token 映射不匹配")
        bid, ask = quote.get("bid"), quote.get("ask")
        if bid is not None and ask is not None and bid >= ask:
            problems.append("订单簿交叉／锁定，等待刷新")
    except (KeyError, TypeError, ValueError):
        problems.append("行情身份或源时间缺失／无效")
    return list(dict.fromkeys(problems))


def walk_asks(quote, budget, extra_cost=0, max_shares=None):
    """Cash budget includes price-dependent fees, depth slippage and extra per-share reserve."""
    budget = _finite(budget, "budget")
    extra_cost = _finite(extra_cost, "extra_cost", hi=.5)
    rate = _finite(quote["fee_rate"], "fee_rate", hi=1)
    exponent = _finite(quote.get("fee_exponent", 1), "fee_exponent", lo=.01, hi=5)
    cap = float("inf") if max_shares is None else _finite(max_shares, "max_shares")
    levels = []
    for row in quote.get("asks", []):
        p, size = _finite(row["price"], "price", lo=.000001, hi=.999999), _finite(row["size"], "size")
        if size:
            levels.append((p, size))
    levels.sort()
    left, shares, notional, fees, fills = budget, 0., 0., 0., []
    for p, available in levels:
        fee = rate * (p*(1-p))**exponent
        unit_cost = p + fee + extra_cost
        quantity = min(available, left/unit_cost, max(0, cap-shares))
        if quantity <= 1e-8:
            break
        shares += quantity; notional += quantity*p; fees += quantity*fee
        left -= quantity*unit_cost
        fills.append({"price": p, "shares": quantity, "fee": quantity*fee})
    cost = budget-left
    return {"shares": shares, "cost": cost, "unused_budget": max(0, left), "notional": notional,
            "fees": fees, "extra_cost": shares*extra_cost,
            "effective_price": cost/shares if shares else None, "vwap": notional/shares if shares else None,
            "depth_limited": left > 1e-6, "fills": fills}


def compare_routes(snapshot, quotes, budget=100, uncertainty_pp=5, extra_cost=.002, min_edge_pp=1):
    if not isinstance(quotes, dict):
        raise ValueError("quotes must be an object")
    budget = _finite(budget, "budget", lo=.01)
    buffer = _finite(uncertainty_pp, "uncertainty_pp", hi=100)/100
    extra_cost = _finite(extra_cost, "extra_cost", hi=.5)
    min_edge = _finite(min_edge_pp, "min_edge_pp", hi=100)/100
    rows = []
    home_quote, away_quote = quotes.get("home"), quotes.get("away")
    mismatched_pair = isinstance(home_quote, dict) and isinstance(away_quote, dict) and (home_quote.get("condition_id") != away_quote.get("condition_id")
                                                    or home_quote.get("token_id") == away_quote.get("token_id"))
    for side in ("home", "away"):
        quote = quotes.get(side)
        p = snapshot.get("p_home")
        if p is not None:
            p = p if side == "home" else 1-p
        problems = ["没有匹配订单簿"] if quote is None else quote_problems(quote, snapshot, side)
        if mismatched_pair:
            problems.append("双方不属于同一二元市场")
        if p is None or snapshot.get("quote_state") not in {"live", "pregame"}:
            problems.append("非可比较的预测状态")
        fill = None
        if not problems:
            try:
                fill = walk_asks(quote, budget, extra_cost)
                if not fill["shares"]:
                    problems.append("卖盘无可用深度")
                elif fill["cost"] < _finite(quote.get("minimum_notional", 0), "minimum_notional") or fill["shares"] < _finite(quote.get("minimum_shares", 0), "minimum_shares"):
                    problems.append("深度不足以满足最小下单量")
            except (TypeError, ValueError, KeyError):
                problems.append("费用或订单簿数据无效")
        row = {"side": side, "p_model": p, "problems": problems, "eligible": False,
               "ev": None, "robust_ev": None, "roi": None, "edge_pp": None, "fill": fill}
        if not problems:
            cost, n = fill["cost"], fill["shares"]
            conservative_p = max(0, p-buffer)
            row.update(ev=n*p-cost, robust_ev=n*conservative_p-cost, roi=(n*p-cost)/cost,
                       edge_pp=(p-fill["effective_price"])*100, effective_odds=1/fill["effective_price"],
                       win_profit=n-cost, loss=-cost, probability_buffer_pp=buffer*100)
            effect = snapshot.get("report", {}).get("news_effect")
            prior_news_p = (effect["p_home_before"] if side == "home" else 1-effect["p_home_before"]) if effect else p
            row.update(before_news_ev=n*prior_news_p-cost, new_news_ev_change=n*(p-prior_news_p))
            row["eligible"] = row["robust_ev"] > 0 and conservative_p-fill["effective_price"] >= min_edge
        rows.append(row)
    candidates = sorted([r for r in rows if r["eligible"]], key=lambda r: r["robust_ev"], reverse=True)
    candidate = candidates[0]["side"] if candidates else "wait"
    reasons = []
    if any(q.get("synthetic") for q in quotes.values() if isinstance(q, dict)) or snapshot.get("report", {}).get("synthetic"):
        reasons.append("合成数据仅用于方案演示")
    if not snapshot.get("model_trained") or snapshot.get("calibration_status") != "calibrated_heldout":
        reasons.append("模型未完成时间外概率校准与评估")
    if (snapshot.get("heldout_validation") or {}).get("development_only"):
        reasons.append("训练结果仍处于研究阶段")
    if snapshot.get("quote_quality") == "provisional" or snapshot.get("freshness") == "source_timestamp_unverified":
        reasons.append("预测或比分源时间未经充分核验")
    if snapshot.get("errors") or snapshot.get("news_health") == "degraded":
        reasons.append("新闻／比分覆盖降级")
    return {"budget": budget, "uncertainty_pp": uncertainty_pp, "extra_cost": extra_cost,
            "routes": rows, "candidate": candidate, "decision": "wait" if reasons else candidate,
            "decision_reasons": reasons, "baseline": {"side": "wait", "ev": 0, "loss": 0},
            "meaning": "按当前信息比较持有至结算的预期值；不使用未来时点选最佳买入时机，不估计限价单成交概率。",
            "uncertainty_note": "概率缓冲是用户选择的压力参数，不是统计置信区间。"}


def synthetic_quotes(snapshot, game, previous=None):
    """Explicitly fictitious book: smooth no-news quote with past observed news only."""
    p = snapshot.get("p_home_without_news", snapshot.get("p_home"))
    if p is None or snapshot.get("quote_state") == "final":
        return {}
    lag = (previous or {}).get("news_effect_pp", 0) / 100
    market_p = min(.98, max(.02, p + .45*lag))
    out = {}
    for side, team, price in (("home", game.home_team, market_p), ("away", game.away_team, 1-market_p)):
        ask = min(.995, price+.012)
        out[side] = {"venue": "polymarket_format_synthetic", "synthetic": True, "game_id": str(game.game_id),
                     "kind": "moneyline", "includes_overtime": True, "rules_verified": True, "side": side,
                     "team": team, "token_id": "synthetic-"+side, "condition_id": "synthetic-condition",
                     "bid": max(.005, price-.012), "ask": ask, "mid": price,
                     "asks": [{"price": ask, "size": 80}, {"price": min(.999, ask+.018), "size": 400}],
                     "bids": [{"price": max(.005, price-.012), "size": 80}, {"price": max(.001, price-.030), "size": 400}],
                     "active": True, "fee_rate": .05, "fee_exponent": 1, "fee_verified": True,
                     "minimum_notional": 5, "observed_at": snapshot["as_of"], "updated_at": snapshot["as_of"],
                     "note": "合成订单簿；Sports rate=.05 是演示参数，真实市场逐个读取费率。"}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=100)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    if "snapshot" in snapshot:
        snapshot = snapshot["snapshot"]
    market = json.loads(args.market.read_text())
    result = compare_routes(snapshot, market.get("quotes", market), args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(result["candidate"], result["decision"], result["decision_reasons"])


if __name__ == "__main__":
    main()
