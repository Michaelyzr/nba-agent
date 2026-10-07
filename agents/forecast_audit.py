"""Explain raw forecast inputs and reproducible parameter sensitivity."""
import math
from statistics import NormalDist

from forecast.inplay import InPlayWinModel, remaining_seconds
from data_sources.inplay_types import utc


def visible_original_data(payload, snapshot, overview=False):
    original = dict(payload["original_data"])
    at = utc(snapshot["as_of"])
    live = snapshot.get("pipeline") == "inplay"
    for field in ("input_scores", "input_events", "input_pregame_news"):
        allowed = live if field != "input_pregame_news" else not live
        original[field] = [r for r in original.get(field, []) if allowed and
                           (overview or (utc(r["observed_at"]) <= at and ("published_at" not in r or utc(r["published_at"]) <= at)))]
    if not overview:
        original["historical_games"] = [r for r in original["historical_games"] if utc(r["final_at"]) <= at]
        original["player_games"] = [r for r in original["player_games"] if utc(r["historical_final_at"]) <= at]
        if not live:
            original["team_records"] = []  # Opening-time summaries are not pregame as-of summaries.
    original["view_mode"] = "full_sample" if overview else "as_of_observation"
    return original

MODEL_OPTIONS = [
    {"model": "扩散原型（当前）", "status": "已实现，未校准", "benefit": "可解释比分、剩余时间与新闻分差", "data": "历史基准、实时比分与球员状态", "limitation": "固定 σ=14；统一每分钟 0.12 分；原型未使用球权特征"},
    {"model": "Logistic + 时间外 sigmoid 校准", "status": "训练入口可用，样例无训练权重", "benefit": "学习比分、时间、基准、新闻与球权权重，单独校准概率", "data": "按比赛分组的历史赛中快照、赛果、事件首次观察时间", "limitation": "需在后续完整比赛测试，不能把同场快照随机拆分"},
    {"model": "单调梯度提升 + sigmoid 校准", "status": "训练入口可用，样例无训练权重", "benefit": "学习非线性临场交互；比分、先验与新闻方向保持单调", "data": "更多真实赛中快照；后续扩展节奏、犯规、球权与阵容", "limitation": "无真实留出评估时不能声称比原型更准确"},
    {"model": "阵容／回合模型 + 正则化球员影响", "status": "后续研究", "benefit": "按替补与剩余轮换估算净影响，区分明星和替补", "data": "逐回合、换人阵容、球员与替补净影响、预计剩余轮换", "limitation": "场均得分不能直接当作球员缺阵造成的净分差"},
]
SOURCES = [
    {"label": "Polymarket 订单簿", "url": "https://docs.polymarket.com/api-reference/market-data/get-order-book"},
    {"label": "Polymarket 费用与市场参数", "url": "https://docs.polymarket.com/market-data/market-details"},
    {"label": "Polymarket 费用公式", "url": "https://docs.polymarket.com/trading/fees"},
    {"label": "Polymarket 结算规则", "url": "https://docs.polymarket.com/concepts/resolution"},
    {"label": "概率校准", "url": "https://scikit-learn.org/stable/modules/calibration.html"},
    {"label": "单调梯度提升", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"},
]


def parameter_scenarios(snapshot):
    if snapshot.get("quote_state") != "live" or not snapshot.get("prior"):
        return []
    sigma = snapshot.get("model_parameters", {}).get("sigma", 14)
    news = snapshot["news_margin"]
    cases = [("扩散参考（基准参数）", sigma, 1), ("新闻影响减半", sigma, .5),
             ("较低比分波动 σ=10", 10, 1), ("较高比分波动 σ=18", 18, 1)]
    result = []
    for label, uncertainty, weight in cases:
        p = InPlayWinModel(sigma=uncertainty).predict(snapshot["score"], snapshot["prior"]["p_home"], news * weight)
        result.append({"label": label, "sigma": uncertainty, "news_weight": weight,
                       "p_home": p, "home_odds": 1 / p, "away_odds": 1 / (1-p)})
    return result


def audit_snapshot(snapshot, previous=None):
    if snapshot.get("quote_state") != "live" or snapshot.get("p_home") is None:
        return {"state": "不适用：无赛中预测", "scenarios": [], "decomposition": []}
    score, prior = snapshot["score"], snapshot["prior"]["p_home"]
    fraction = max(remaining_seconds(score) / 2880, 1 / 2880)
    sigma = snapshot.get("model_parameters", {}).get("sigma", 14)
    before = snapshot["p_home_before_new_events"]
    reference, after = snapshot["p_home_without_news"], snapshot["p_home"]
    old = (previous or {}).get("p_home")
    parts = [] if old is None else [
        {"component": "比分、时钟与既有因素剩余影响", "delta_pp": (before-old)*100},
        {"component": "本轮新增事件（同比分／时钟）", "delta_pp": (after-before)*100}]
    return {"state": "扩散原型，可复算" if not snapshot["model_trained"] else "训练模型，扩散仅作敏感性参考",
            "prior_p_home": prior, "prior_source": snapshot["prior"]["model"], "prior_as_of": snapshot["prior"]["as_of"],
            "score_margin": score["home_score"]-score["away_score"], "remaining_seconds": remaining_seconds(score),
            "fraction_remaining": fraction, "sigma": sigma,
            "prior_probit": NormalDist().inv_cdf(prior), "prior_remaining_margin": NormalDist().inv_cdf(prior)*sigma*fraction,
            "remaining_sigma": sigma*math.sqrt(fraction), "news_margin": snapshot["news_margin"],
            "no_news_p": reference, "old_factors_p": before, "after_news_p": after,
            "previous_p": old, "decomposition": parts, "scenarios": parameter_scenarios(snapshot),
            "sensitivity_note": "参数情景范围，不是置信区间；未在真实数据上证明准确率。",
            "formula": "p = Φ((比分差 + Φ⁻¹(开赛基准) × σ × 剩余比例 + 新闻净分差) / (σ × √剩余比例))；odds = 1/p" if not snapshot["model_trained"] else f"当前预测使用 {snapshot['model']} 的训练特征与权重；下列 σ 与 Φ 情景仅是扩散参考，不是训练模型的计算公式。",
            "issues": ["剩余分钟从历史常规分钟减去已上场分钟估算，缺少实际轮换与替补预测。",
                       "统一 0.12 分/分钟及存疑状态损失份额是人工假设，需要球员净影响与真实事件校准。",
                       "扩散原型临近终场不建模球权、暂停、犯规战术；0.001–0.999 截断不是准确率证明。",
                       "新闻必须按首次观察时刻进入模型；同一球员事件去重，避免重复叠加。",
                       "胜率变化和 odds 变化为非线性倒数关系；高胜率会对应较低公平赔率。"]}
