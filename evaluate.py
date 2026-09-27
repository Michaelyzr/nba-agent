"""Run the held-out evaluation and print the tables the report copies.

1. The 30 questions in data/eval/questions.json through the full agent.
2. Ranker Recall@5 against BM25 on the test split of rank_labels.csv.
3. Support-check F1 against the number-match rule and a majority-class baseline.
4. Signing-network MAE against linear regression and the median (from train_signing.py).
"""
import json
import math

import pandas as pd
from sklearn.metrics import f1_score

import models
from graph import start
from tables import EVAL, MODELS

VALUE_KEYS = {"ts_pct": ("before_ts", "after_ts"), "net_rating": ("before_net", "after_net")}


def score_question(q):
    e = q["expect"]
    out = start(q["question"], plant=e.get("plant"))
    fired = {t["detail"].split(":")[0] for t in out["trace"] if t["step"] in ("checks.py", "support_check")}
    trace_text = " ".join(t["detail"] for t in out["trace"])
    if q["type"] == "followup":
        ok = out["paused"] and e["followup_field"] in out.get("missing", [])
        return ok, f"asked for {out.get('missing')}"
    if out["paused"]:
        return False, f"unexpected follow-up: {out.get('followup')}"
    result = out.get("result", {})
    if q["type"] == "planted":
        ok = e["expect_check"] in trace_text and out.get("status") == "answered"
        return ok, f"{e['expect_check']} {'fired, then retry passed' if ok else 'not handled'}"
    if out.get("status") != "answered":
        return False, "unresolved"
    if q["type"] in ("trade_split", "team_window"):
        b, a = result["windows"]
        key = next(iter(b["values"]))
        kb, ka = VALUE_KEYS[key]
        v = e["values"]
        ok = (b["games"] == v["before_games"] and a["games"] == v["after_games"]
              and math.isclose(b["values"][key], v[kb], abs_tol=1e-3) and math.isclose(a["values"][key], v[ka], abs_tol=1e-3))
        return ok, f"{key} {b['values'][key]:.3f} -> {a['values'][key]:.3f}"
    if q["type"] == "rate":
        w, v = result["windows"][0], e["values"]
        ok = (w["games"] == v["games"] and math.isclose(w["values"]["pts_per_game"], v["pts_per_game"], abs_tol=1e-3)
              and math.isclose(w["values"]["pts_per_100"], v["pts_per_100"], abs_tol=1e-3))
        return ok, f"{w['values']['pts_per_game']:.1f} per game, {w['values']['pts_per_100']:.1f} per 100"
    if q["type"] == "context":
        kept = {p["para_id"] for p in result.get("paragraphs", [])}
        ok = bool(set(e["relevant_para_ids"]) & kept)
        return ok, f"kept {sorted(kept)}"
    if q["type"] == "signing":
        err = abs(result["prediction"] - e["actual_annual_pay"])
        return True, f"predicted ${result['prediction'] / 1e6:.2f}M, actual ${e['actual_annual_pay'] / 1e6:.2f}M, error ${err / 1e6:.2f}M"
    return False, "unknown type"


def recall_at_5():
    labels = pd.read_csv(EVAL / "rank_labels.csv")
    test = labels[labels.split == "test"]
    trained = models._cross_encoder("ranker")
    rows = {"BM25": [], "cross-encoder ranker": []}
    for query, group in test.groupby("query"):
        rel = set(group[group.label == 1].doc_id)
        if not rel:
            continue
        texts, ids = group.text.tolist(), group.doc_id.tolist()
        denom = min(len(rel), 5)
        bm = models.bm25_scores(query, texts)
        top = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -bm[i])[:5]]
        rows["BM25"].append(len(rel & set(top)) / denom)
        if trained is not None:
            sc = trained.predict([(query, t) for t in texts])
            top = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -sc[i])[:5]]
            rows["cross-encoder ranker"].append(len(rel & set(top)) / denom)
    return {k: (sum(v) / len(v) if v else None) for k, v in rows.items()}, test["query"].nunique()


def support_f1():
    labels = pd.read_csv(EVAL / "support_labels.csv")
    test = labels[labels.split == "test"]
    y = test.label.tolist()
    rule = [int(set(models._numbers(s)) <= set(models._numbers(e))) for s, e in zip(test.sentence, test.evidence)]
    majority = int(labels[labels.split == "train"].label.mean() >= 0.5)
    out = {"number-match rule": f1_score(y, rule), "majority class": f1_score(y, [majority] * len(y), zero_division=0)}
    trained = models._cross_encoder("support")
    if trained is not None:
        pred = [int(s >= 0.5) for s in trained.predict(list(zip(test.sentence, test.evidence)))]
        out["cross-encoder support"] = f1_score(y, pred)
    return out, len(test)


def main():
    questions = json.loads((EVAL / "questions.json").read_text())
    print(f"== Agent on {len(questions)} held-out questions ==")
    results = []
    for q in questions:
        ok, detail = score_question(q)
        results.append({"id": q["id"], "type": q["type"], "pass": ok, "detail": detail})
        print(f"{q['id']} {q['type']:<12} {'PASS' if ok else 'FAIL'}  {detail}")
    by_type = pd.DataFrame(results).groupby("type")["pass"].agg(passed="sum", total="count")
    print(by_type.to_string())

    r5, nq = recall_at_5()
    print(f"\n== Ranker Recall@5 on {nq} held-out queries ==")
    for k, v in r5.items():
        print(f"{k:<22} {'not trained' if v is None else f'{v:.3f}'}")

    f1, n = support_f1()
    print(f"\n== Support check F1 on {n} held-out pairs ==")
    for k, v in f1.items():
        print(f"{k:<22} {v:.3f}")

    path = MODELS / "signing" / "metrics.json"
    if path.exists():
        m = json.loads(path.read_text())
        print(f"\n== Signing MAE on {m['n_heldout']} held-out contracts ==")
        for k, label in (("mlp_mae", "network"), ("linreg_mae", "linear regression"), ("median_mae", "median pay")):
            print(f"{label:<22} ${m[k] / 1e6:.2f}M")

    (EVAL / "results.json").write_text(json.dumps({"questions": results, "recall_at_5": r5, "support_f1": f1}, indent=2))


if __name__ == "__main__":
    main()
