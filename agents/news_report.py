"""Deterministic, cited news-loop briefs. Presentation only; no forecasting/I/O sources."""
import json
from pathlib import Path

from data_sources.inplay_types import event_id
from agents.game_timeline import news_effect

STATUS_LABELS = {
    "out": "确认缺阵", "available": "确认可以出场", "active": "可以出场",
    "questionable": "出场存疑", "doubtful": "大概率缺阵", "probable": "大概率出场",
    "minutes_limit": "上场时间限制", "left_injured": "因伤离场，回归未定",
    "injury_out": "本场无法回归", "questionable_return": "回归存疑",
    "doubtful_return": "大概率无法回归", "returned": "已回归比赛",
    "ejected": "被驱逐", "fouled_out": "六犯离场", "rescinded": "离场判罚被撤销",
    "review": "待核验，不自动调整预测",
}


def make_report(snapshot, game, players, previous=None, new_rows=()):
    """All displayed numbers come from the snapshot; never invent causal effects."""
    pipeline = snapshot.get("pipeline", "pregame")
    names = dict(zip(players.player_id.astype(int), players.player_name))
    current, old = snapshot.get("p_home"), (previous or {}).get("p_home")
    reference = snapshot.get("p_home_without_news")
    if reference is None and pipeline == "pregame":
        reference = snapshot["baseline"]["p_home"]
    incremental = snapshot.get("new_event_effect_pp")
    if pipeline == "pregame" and current is not None:
        incremental = snapshot.get("delta_home", 0) * 100
    ids = snapshot.get("new_event_ids", snapshot.get("new_news_ids", []))
    selected = {r.get("event_id", r.get("news_id")) for r in snapshot.get("factors", [])}
    evidence = []
    seen = set()
    for row in new_rows:
        rid = row.get("event_id", row.get("news_id"))
        if rid not in ids or rid in seen:
            continue
        seen.add(rid)
        evidence.append({"id": rid, "player_id": row["player_id"],
                         "player": names.get(int(row["player_id"]), str(row["player_id"])),
                         "status": row["status"], "status_label": STATUS_LABELS.get(row["status"], row["status"]),
                         "text": row.get("text", ""), "source": row["source"],
                         "source_label": row.get("source_label", row["source"]),
                         "url": row.get("url", ""), "synthetic": bool(row.get("synthetic", False)),
                         "published_at": str(row["published_at"]), "observed_at": str(row["observed_at"]),
                         "selected": rid in selected})
    factors = [{"player": names.get(int(r["player_id"]), str(r["player_id"])),
                "status": STATUS_LABELS.get(r["status"], r["status"]), "source": r.get("source_label", r["source"])}
               for r in snapshot.get("factors", [])]
    state = snapshot.get("quote_state", "pregame")
    synthetic = bool(snapshot.get("synthetic")) or any(r["synthetic"] for r in evidence)
    synthetic |= (snapshot.get("score") or {}).get("source") == "synthetic_inplay_demo"
    synthetic |= any(r.get("state") == "synthetic" for r in snapshot.get("source_coverage", []))
    headline = f"{game.away_team} @ {game.home_team} · {'赛前' if pipeline == 'pregame' else '赛中'}更新"
    if current is None:
        summary = f"当前状态为 {state}，缺少可用于更新的比赛数据，暂不输出新的胜率和赔率。"
    elif state == "final":
        summary = f"比赛已结束，{game.home_team if current == 1 else game.away_team} 获胜；循环停止，终场结果不是新的预测。"
    else:
        summary = f"当前 {game.home_team} 胜率 {current:.2%}，公平十进制赔率 {snapshot['home_decimal_odds']:.3f}。"
        if old is not None:
            summary += f" 相比上次轮询变化 {(current - old) * 100:+.2f} 个百分点。"
        if reference is not None:
            summary += f" 同一比分/时钟下的无新闻参考为 {reference:.2%}。" if pipeline == "inplay" else f" 历史基准为 {reference:.2%}。"
        if incremental is not None:
            summary += f" 本轮新增消息的模型影响为 {incremental:+.2f} 个百分点。"
    notes = ["赔率为模型公平赔率，不含庄家利润；消息权重和赛中原型需要历史校准。"]
    if synthetic:
        notes.insert(0, "合成新闻、比分与发布情景，仅用于功能演示，不是实际报道或历史实况。")
    if snapshot.get("conflicts"):
        notes.append("存在冲突证据，采用当前体系的来源优先级，保留其他报道供核验。")
    if snapshot.get("errors"):
        notes.append("数据源覆盖不完整；查看错误与新鲜度字段后再解释结果。")
    if snapshot.get("freshness") == "source_timestamp_unverified":
        notes.append("源更新时间未经验证，HTTP 获取时间不能证明比分内容刚刚更新。")
    report = {"report_id": event_id(pipeline, game.game_id, snapshot["as_of"], ids),
              "pipeline": pipeline, "as_of": snapshot["as_of"], "headline": headline,
              "summary": summary, "synthetic": synthetic, "quote_state": state,
              "p_home": current, "home_decimal_odds": snapshot.get("home_decimal_odds"),
              "away_decimal_odds": snapshot.get("away_decimal_odds"),
              "previous_p_home": old, "reference_p_home": reference,
              "total_delta_pp": (current - old) * 100 if current is not None and old is not None else None,
              "active_news_effect_pp": (current - reference) * 100 if current is not None and reference is not None and state != "final" else None,
              "new_event_effect_pp": incremental, "news_effect": news_effect(snapshot), "new_evidence": evidence,
              "factors": factors, "conflict_count": len(snapshot.get("conflicts", [])),
              "notes": notes,
              "trace": [{"stage": "retrieve", "detail": f"本轮接收 {len(ids)} 条新的球员消息"},
                        {"stage": "extract", "detail": f"保留 {len(factors)} 个球员状态，{len(snapshot.get('conflicts', []))} 个冲突"},
                        {"stage": "forecast", "detail": f"{snapshot['model']} · {state}"},
                        {"stage": "report", "detail": "使用模型输出与原始证据生成简报"},
                        {"stage": "record", "detail": "写入本体系的快照、简报和恢复状态"}]}
    report["markdown"] = render_markdown(report)
    return report


def render_markdown(report):
    lines = [f"# {report['headline']}", "", f"观察时间：{report['as_of']}", "", report["summary"], "", "## 当前因素", ""]
    lines.extend(f"- {r['player']}：{r['status']}；来源 {r['source']}。" for r in report["factors"])
    if not report["factors"]:
        lines.append("暂无可应用的球员变化因素。")
    effect = report.get("news_effect")
    if effect and report["new_evidence"]:
        lines += ["", "## 新消息影响（同一比分与时钟）", "",
                  f"主队胜率：{effect['p_home_before']:.2%} → {effect['p_home_after']:.2%}（{effect['home_delta_pp']:+.2f} pp）。",
                  f"主队 odds：{effect['home_odds_before']:.3f} → {effect['home_odds_after']:.3f}（{effect['home_odds_delta']:+.3f}）。",
                  f"客队 odds：{effect['away_odds_before']:.3f} → {effect['away_odds_after']:.3f}（{effect['away_odds_delta']:+.3f}）。"]
    lines += ["", "## 本轮证据", ""]
    for i, row in enumerate(report["new_evidence"], 1):
        source = ("合成情景 · " if row["synthetic"] else "") + row["source_label"]
        if not row["synthetic"] and row["url"].startswith("https://"):
            source = f"[{source}]({row['url']})"
        lines.append(f"{i}. {row['player']}：{row['text']} — {source}。发布 {row['published_at']}，首次观察 {row['observed_at']}。")
    if not report["new_evidence"]:
        lines.append("本轮没有新的球员消息；重复报道不会再次叠加影响。")
    lines += ["", "## 说明", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def persist_report(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reports.jsonl").open("a") as file:
        file.write(json.dumps(report, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    temporary = output / "latest_report.md.tmp"
    temporary.write_text(report["markdown"])
    temporary.replace(output / "latest_report.md")
