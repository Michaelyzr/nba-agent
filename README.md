# NBA analytics agent (DASC7606C prototype)

One LangGraph agent that answers a question about an NBA player or team with a cited note.
Every number cites saved games, every quote cites a saved paragraph, and a signing figure is
labeled as a model prediction with its held-out error. See `../DASC7606C_Group_Proposal_v0.md`.

> **The data shipped with this prototype is synthetic.** `make_sample_data.py` builds a fake
> 2025-26 season (real team names, made-up players, stats, contracts, and news) so the whole
> loop runs offline. Run `download_season.py` for the real season before any real result is reported.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
# If PyPI is slow from HK, use: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -r requirements.txt
python make_sample_data.py      # synthetic season, labels, 30 test questions
python train_signing.py         # signing network (seconds)
python train_ranker.py          # optional: fine-tune the ranker cross-encoder (downloads the base model)
python train_support.py         # optional: fine-tune the support cross-encoder
cp .env.example .env            # optional: add GEMINI_API_KEY; without it the chat steps use offline rules
```

## Run

```bash
python graph.py "Did <player>'s true shooting change after he was traded this season?"
python graph.py --plant playoffs "<same question>"    # inject a fault to watch checks.py reject and the loop retry
streamlit run app.py                                   # demo page
python evaluate.py                                     # 30 questions, Recall@5, F1, MAE tables
pytest -q                                              # planted failures
python export_csv.py                                   # every parquet file as CSV in data/exports/
```

Player names come from the sample data; `data/eval/questions.json` lists questions that work.

## How the pieces map to the proposal

| File | Role |
| --- | --- |
| `graph.py` | The one agent. LangGraph state, nodes, arrows, retry limit (3), pause for follow-up |
| `steps.py` | Each node: clarify, wait, plan, rank, write code, run code, rank paragraphs, build features, predict, check, write note, support check, unresolved |
| `checks.py` | Hard rules: playoff mix, wrong team, rate basis, game count, stat mismatch, citation, quote, article number, signing label/error/cap claim |
| `models.py` | Ranker, support check, signing network. Uses the fine-tuned cross-encoders once trained; until then BM25 and a number-match rule stand in, and the trace says so |
| `llm.py` | Gemini client. With no key the four chat steps fall back to rules and templates |
| `skills/*.md` | One per question type; the plan step opens exactly one |
| `download_season.py` | Real one-time pull from `nba_api` (V3 advanced box scores, transactions file). Resumable |
| `make_sample_data.py` | Synthetic stand-in with the same schema |
| `train_*.py`, `evaluate.py` | The three models and their baselines |

Fault names for `--plant` (applied on the first try only): `playoffs`, `wrong_team`, `no_count`,
`rate_mix`, `invented_quote`, `wrong_paragraph`, `no_error`, `note_number`.

## Known limits of the prototype

- Generated code is run with `exec` on your machine. Fine for a course demo; not safe for a public service.
- `download_season.py` has not been run end to end against stats.nba.com; smoke-test it with `--limit 5`.
- Contracts and news paragraphs are not in `nba_api`. For the real build, put hand-made
  `data/info/contracts.csv` (PLAYER_ID, SEASON, ANNUAL_PAY, YEARS, PRIOR_PAY, SPLIT) and
  `data/news/paragraphs.csv` (PARA_ID, DATE, PLAYER_ID, KIND, SOURCE, URL, TEXT) in place.
- The rank and support labels are generated from templates. The Models subgroup replaces them with hand labels.
