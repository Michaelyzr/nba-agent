"""Streamlit forecast audit and read-only Polymarket route comparison."""
import json

from agents.market_analysis import compare_routes
from agents.forecast_audit import visible_original_data


def render_analysis(payload, snapshot, key, overview=False):
    import pandas as pd
    import streamlit as st

    st.subheader("计算拆解 · 原始数据与事件影响")
    audit = snapshot.get("forecast_audit") or {}
    if audit.get("scenarios"):
        st.code(audit["formula"], language=None)
        st.caption(payload["original_data"]["prior_equation"] + " 主场修正 " + str(payload["original_data"]["home_edge"]))
        st.dataframe(pd.DataFrame(payload["original_data"]["team_records"]), hide_index=True)
        raw = {"开赛历史基准": audit["prior_p_home"], "历史基准来源": audit["prior_source"],
               "历史数据截止": payload["original_data"]["history_cutoff"], "当前比分差": audit["score_margin"],
               "剩余比赛秒数": audit["remaining_seconds"], "σ": audit["sigma"],
               "先验剩余分差": audit["prior_remaining_margin"], "新闻净分差": audit["news_margin"],
               "剩余标准差": audit["remaining_sigma"], "无新闻胜率": audit["no_news_p"],
               "旧因素胜率": audit["old_factors_p"], "新因素胜率": audit["after_news_p"]}
        st.dataframe(pd.DataFrame({"计算项": raw.keys(), "当前值": [str(v) for v in raw.values()]}), hide_index=True)
        if audit["decomposition"]:
            st.dataframe(pd.DataFrame(audit["decomposition"]), hide_index=True)
        if snapshot["player_effects"]:
            st.dataframe(pd.DataFrame(snapshot["player_effects"]), hide_index=True)
        st.dataframe(pd.DataFrame(audit["scenarios"]), hide_index=True)
        st.caption(audit["sensitivity_note"])
        with st.expander("当前算法可改进的假设"):
            for issue in audit["issues"]:
                st.write(issue)
    else:
        st.info("当前时刻无赛中预测；赛前与终场单独查看。")

    st.subheader("Polymarket · 模型与可成交成本对照")
    columns = st.columns(3)
    budget = columns[0].number_input("比较预算", min_value=1., value=100., key=key+"_budget")
    buffer = columns[1].number_input("概率缓冲 (pp)", min_value=0., max_value=100., value=5., key=key+"_buffer")
    extra = columns[2].number_input("每份额外成本", min_value=0., max_value=.5, value=.002, step=.001, format="%.3f", key=key+"_extra")
    upload = st.file_uploader("导入只读 Polymarket 订单簿 JSON", type=["json"], key=key+"_market_import")
    quotes = snapshot.get("market_quotes", {})
    if upload:
        try:
            loaded = json.loads(upload.getvalue())
            quotes = loaded.get("quotes", loaded)
            if not isinstance(quotes, dict) or any(q is not None and not isinstance(q, dict) for side, q in quotes.items() if side in ("home", "away")):
                raise ValueError("quotes must be an object")
            st.caption("已导入真实抓取数据；按当前快照的比赛 ID、时刻和结算规则检查。")
            if loaded.get("rules"):
                st.write(loaded["rules"])
        except (ValueError, AttributeError):
            st.error("JSON 必须包含双方 quotes；导入失败，恢复样例行情。")
            quotes = snapshot.get("market_quotes", {})
    else:
        st.caption("合成 Polymarket 格式订单簿，非真实报价。Sports 费率 0.05 为演示假设，真实费率逐市场读取。")
    result = compare_routes(snapshot, quotes, budget, buffer, extra)
    prices, routes = [], []
    for row in result["routes"]:
        side = row["side"]
        q = quotes.get(side) or {}
        fill = row["fill"] or {}
        prices.append({"outcome": payload[side+"_team"], "token_id": q.get("token_id"),
                       "model_probability": row["p_model"], "ask": q.get("ask"), "bid": q.get("bid"),
                       "effective_price": fill.get("effective_price"), "effective_odds": row.get("effective_odds"),
                       "observed_at": q.get("observed_at"), "source_updated_at": q.get("updated_at")})
        routes.append({"路线": payload[side+"_team"]+" 胜出份额", "预算使用": fill.get("cost"), "份额": fill.get("shares"),
                       "费用": fill.get("fees"), "深度 VWAP": fill.get("vwap"), "EV": row["ev"], "ROI": row["roi"],
                       "缓冲后 EV": row["robust_ev"], "事件前 EV": row.get("before_news_ev"), "事件 EV 变化": row.get("new_news_ev_change"),
                       "盈利情景": row.get("win_profit"), "亏损情景": row.get("loss"),
                       "阻断／深度": "; ".join(row["problems"]) or ("深度不足，预算未全用完" if fill.get("depth_limited") else "当前深度可覆盖")})
    routes.append({"路线": "等待", "预算使用": 0, "EV": 0, "亏损情景": 0})
    st.dataframe(pd.DataFrame(prices), hide_index=True)
    st.dataframe(pd.DataFrame(routes), hide_index=True)
    name = lambda side: payload[side+"_team"] if side in ("home", "away") else "等待"
    st.write(f"**条件下最优模拟候选：{name(result['candidate'])}；当前可执行判断：{name(result['decision'])}**")
    st.caption("；".join(result["decision_reasons"]) or "只读分析，不发送订单。")
    st.caption(result["meaning"]+" "+result["uncertainty_note"]+" 仅考虑压力后净优势至少 1 pp，按压力 EV 排序。")
    st.code("EV = 份额 × 概率 − 总成本\n总成本 = 卖盘逐价成本 + 逐价费用 + 额外成本\n可成交 odds = 1 / 每份总成本", language=None)
    with st.expander("如何对应实时市场与结算"):
        st.write("先匹配球队、开赛时间和全场 moneyline 的 outcome token，再核对含加时、取消与推迟的结算条款。买入读取 ask 与深度；卖出读取 bid。中间价和最新成交价不代表可成交价格。")
        st.code("python -m data_sources.polymarket_books --event-slug <slug> --game-id <id> --source frozen --output runs/market-book.json")
        st.write("原始条款确认后可加 --rules-verified。15 秒以上行情或晚于当前模型的行情被阻断。历史样例无法套用今天的实时报价。")
    st.subheader("模型改进与验证")
    st.dataframe(pd.DataFrame(payload["model_options"]), hide_index=True)
    if snapshot.get("heldout_validation"):
        st.json(snapshot["heldout_validation"])
    else:
        st.caption("样例没有真实训练／校准结果，不展示虚构的准确率、Brier、log loss 或收益。")
    with st.expander("查看原始历史数据、快照、订单簿"):
        st.json({"snapshot": snapshot, "original_data": visible_original_data(payload, snapshot, overview), "market_quotes": quotes})
    st.markdown(" · ".join(f"[{r['label']}]({r['url']})" for r in payload["reference_sources"]))
