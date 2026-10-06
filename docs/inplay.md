# 独立的赛中新闻、胜率和赔率系统

`agents/inplay.py` 提供独立循环：**实际开赛 → 拉取比分/时钟 → 接收赛中消息 →
更新球员状态 → 重算胜率和公平赔率 → 记录 → 继续，直到终场**。
它不导入 `agents/pregame.py`，不读赛前新闻表、缓存或运行状态，不修改原有模型权重。
两套系统只共用只读球队账号目录、历史统计和历史基准模型；可分别启动、停止和恢复。
赛中数据写到 `runs/inplay/<name>/`，赛前数据仍在原目录。

## 运行

先激活项目虚拟环境，在项目根目录执行。无需联网的合成场景演示：

```bash
source .venv/bin/activate
python -m agents.inplay --mode replay --source sample --game-id 401811041 --name inplay-demo --initial-p-home 0.60
streamlit run inplay_app.py
```

演示依次包含伤退、确认无法回归、回归、对手被驱逐和终场；比分及消息均为合成数据，
历史表仅提供球员和轮换背景。`--initial-p-home` 可直接指定初始主队胜率；不指定时，
加载已有 M4，从开赛时可见的历史数据计算独立基准。如果本机还没有 M4 权重，
可先运行 `python -m forecast.win --source sample`。

真正的赛中运行：

```bash
python -m agents.inplay --mode live --source frozen --game-id <当前比赛的ESPN事件ID> --poll-seconds 5 --name live-game
```

`data/frozen/` 必须已有当前比赛的赛程、历史 `player_games.parquet` 和本地
`players.parquet`，且 ID 一致。赛程更新仍由原数据管道负责；历史 sample 不能当作今天的赛程。
可提前启动，开赛时间之前不发送检索请求；之后也要等数据源确认比赛处于 live 状态，才开启新闻检索。
NBA 官方 ID 与 ESPN 事件 ID 不同，自动匹配失败时可额外指定 `--nba-game-id <10位NBA比赛ID>`。
已开赛超过八小时的历史场次会被 live 模式拒绝。

独立页面每五秒刷新已有日志，不会启动检索或修改状态。终端循环需要持续运行；
Ctrl+C 会关闭流连接，保留恢复状态。相同 `--name` 恢复同一场比赛；更换参数或模型应使用新名字。

## 检索来源与时效

| 来源 | 用途与方式 |
| --- | --- |
| NBA liveData box score / play-by-play | 当前比分、节次、时钟、已打分钟、六犯离场、明确驱逐/伤退事件；默认五秒轮询 |
| ESPN summary + scoreboard | NBA 接口失败时提供备用比分/事件，明确标记 fallback |
| Shams、两队官方/PR 账号、两队跟队记者 | 使用 X filtered stream 持续接收帖子，新消息唤醒循环 |
| ESPN / CBS NBA RSS | 每六十秒补充权威媒体消息 |

仅检索本场球队的账号和相关球员。账号目录为 `data_sources/news_sources.json`，
具体名单见 [新闻来源目录](news_sources.md)。NBA liveData 在本次网络检查中返回 403，
ESPN 返回 200；因此备用路径与来源异常提示都保留，不把 fallback 写成 NBA 成功。

X 需要在 `.env` 配置 **`INPLAY_X_BEARER_TOKEN`**，并拥有 filtered stream 和 recent-search
访问权限。不会自动复用赛前的 `X_BEARER_TOKEN`。建议使用独立 X App/project，
避免两套系统竞争额度或连接。未配置时记录 `unconfigured`，其他来源继续运行。
客户端只添加本场带 `nba-agent-inplay:` 标签的缺失规则，不删除已有规则；规则会在 X 中持久保留。
X 的连接数及规则上限取决于当前权限，多场并发需要按权限配置共用流分发或独立连接。

[X 官方文档](https://docs.x.com/x-api/posts/filtered-stream/introduction) 将 filtered stream
描述为近实时传输；它仍有平台延迟，记者也需要时间发布消息。五秒是比分轮询的目标周期，
耗时请求会延长周期；流消息触发后也需获取可用的当前比分再计算。
无法保证伤退发生的瞬间就被发现，也无法以检索频率保证胜率准确。

断线后按退避重连，recent-search 以三十秒索引缓冲、两分钟重叠窗口补回，保存未完成分页。
429 会暂停重试；重连缺口未补齐、队列溢出、来源失败会标记覆盖不完整。
图文中的图片保留原链接及附件供核验，尚未实现 OCR。

## 事件与冲突

| 状态 | 当前计算使用的剩余上场时间损失比例 |
| --- | --- |
| 伤退且确认无法回归、被驱逐、六犯离场 | 1.00 |
| doubtful to return | 0.75 |
| questionable to return | 0.50 |
| 因伤离场/进入更衣室，尚无回归结论 | 0.35 |
| 明确已回归、驱逐或第六犯被撤销 | 0.00 |

解析要求明确赛中措辞与可匹配球员；否定、假设、未来会回归、历史伤情和多人歧义仅记录待审，
普通换人不视为伤退。当前名单按独一的全名映射到本地球员 ID，避免混淆 NBA 和 ESPN ID，
也允许当前 box score 修正旧历史名单。目录中没有的球员需要先更新球员表。

冲突顺序是 **NBA 官方比赛数据 → ESPN 等已登记权威媒体 → 球队官方/PR → Shams → 跟队记者**。
每个来源保留最新证据，同层优先最新发布时间；低优先级矛盾消息保留并显示，不直接覆盖权威结论。
只有离场事实、尚无结论时，后续明确的回归预期可补充它。
驱逐/六犯不能被同来源的普通“回归”报道自动撤销，需要明确撤销；权威来源可纠正低层级误报。
确认回归后的新伤退建立新事件阶段，不被旧回归证据压住。重复消息不反复叠加损失。

## 模型与新鲜度

初始胜率固定在开赛时的历史视图，不读取赛前循环的因素。
独立 `forecast/inplay.py` 根据比分差、剩余时间、基准胜率和剩余球员影响更新结果，支持加时。
只调整剩余预计上场分钟，而不是再次扣除已经打完的整场贡献；当前统一替补净影响假设
为每损失一分钟 0.12 分，上表比例也都是**情景假设，需要赛中历史数据校准**。
默认扩散模型明确标为 `inplay-diffusion-prototype`。

公平十进制赔率为 `1 / P(win)`，不含庄家利润，不是抓取的市场报价。
终场胜方概率为 1、败方为 0；败方赔率用 null 表示，循环停止。
暂停、无可验证比分、比分时间倒退或默认超过三十秒的新鲜度限制时，不输出新的赔率。
不会在比分接口失败后仅靠缓存比分重新给伤退定价。
请求观察时间与源更新时间分开记录；没有可验证源时间戳时标记
`source_timestamp_unverified` 和 provisional，不把新的 HTTP 请求当成内容刚刚更新。

可用独立历史赛中快照训练 LogisticRegression，不覆盖原 M4：

```bash
python -m forecast.inplay --training-data data/inplay_training.parquet --split 2026-02-01 --output models/inplay/win.pkl
python -m agents.inplay --mode live --source frozen --game-id <当前事件ID> --model-file models/inplay/win.pkl --name trained-live
```

训练表需要 `game_id, tip_time, final_at, as_of, home_win` 与四个特征
`scaled_margin, scaled_prior, scaled_news, scaled_possession`。
运行日志中的 `training_features` 可提供特征，但赛后必须单独补充真实终场标签；
正式训练需要有代表性的真实历史比赛快照，不能用合成演示评估准确率。
特征只能含该观察时刻已知信息，快照必须在开赛后、终场前；训练仅使用切分日前已完成的比赛，
按比赛时间隔离测试集，训练时每场总权重相等。输出保留测试 Brier score、log loss 和比赛数。
“已拟合”本身不代表已校准；仍需检查留出数据、伤退因素和概率可靠性。
只加载可信本地生成的模型文件。

## 回放与记录

真实回放可提供独立 parquet 文件：

```bash
python -m agents.inplay --mode replay --source frozen --game-id <id> --scores <scores.parquet> --events <events.parquet> --name historical-live
```

比分列：`game_id, phase, home_score, away_score, period, clock_seconds, observed_at, source`，
可选 `updated_at, player_minutes, player_teams, roster, possession_sign`。
其中 player_minutes/player_teams 使用 `{本地球员ID字符串: 数值}`；phase 是 scheduled/live/final/suspended。
事件列：`event_id, game_id, player_id, status, source, published_at, observed_at, text, url`。
时间均带时区。回放同时检查发布与首次观察时间，决策时刻之后的消息不能提前使用；
它不搜索今天的新闻来解释历史比赛。

| 文件 | 内容 |
| --- | --- |
| `inplay_state.json` | 独立恢复状态、固定基准、消息候选与数据源游标；不含凭证 |
| `inplay_snapshots.jsonl` | 比分、胜率、赔率、球员影响、冲突、延迟、来源覆盖与模型验证信息 |
| `inplay_events.jsonl` | 去重的球员事件和待审解析 |
| `inplay_evidence.jsonl` | 原始媒体/帖子与官方事件证据 |

赛前原入口 `agents.pregame`、原页面 `app.py` 及其状态格式保持不变。
赛中独立验证：`pytest -q tests/test_inplay.py`；完整回归：`pytest -q`。
