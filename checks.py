"""Rules the chat model cannot skip. Plain Python, no model calls.

check_result() judges the table/prediction/paragraph result before a note exists.
check_note() judges the written note against that result.
Each returns a list of failures: [{"check": name, "detail": text}]. Empty means pass.
"""
import math
import re

import pandas as pd

from tables import load

REL_TOL = 1e-4
CAP_WORDS = re.compile(r"cap room|cap space|under the cap|salary cap|recommend", re.I)
TAG = re.compile(r"\[(W|P|M):([^\]]+)\]")
QUOTE = re.compile(r"[\"“]([^\"”]+)[\"”]")
DATES = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}-\d{2}\b|\b\d{1,2} (January|February|March|April|May|June|July|August|"
    r"September|October|November|December) \d{4}\b|\b(19|20)\d{2}\b|per 100\b"
)
NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?M?")


def fail(check, detail):
    return {"check": check, "detail": detail}


def close(a, b):
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return False
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-6)


# ---------- ground truth recomputed from the saved season ----------

def true_shooting(rows):
    denom = 2 * (rows.FGA.sum() + 0.44 * rows.FTA.sum())
    return float(rows.PTS.sum() / denom) if denom else float("nan")


def recompute(metric_key, subject, game_ids):
    if subject["kind"] == "team":
        rows = load("team_regular")
        rows = rows[(rows.TEAM_ID == subject["id"]) & rows.GAME_ID.isin(game_ids)]
        if metric_key == "net_rating":
            return float((rows.PTS.sum() - rows.OPP_PTS.sum()) / rows.POSS.sum() * 100)
    rows = load("player_regular")
    rows = rows[(rows.PLAYER_ID == subject["id"]) & rows.GAME_ID.isin(game_ids)]
    if metric_key == "ts_pct":
        return true_shooting(rows)
    if metric_key == "pts_per_game":
        return float(rows.PTS.sum() / len(rows)) if len(rows) else float("nan")
    if metric_key == "pts_per_100":
        adv = load("advanced_regular")
        poss = adv[(adv.PLAYER_ID == subject["id"]) & adv.GAME_ID.isin(game_ids)].POSS.sum()
        return float(rows.PTS.sum() / poss * 100) if poss else float("nan")
    raise KeyError(metric_key)


def signing_features(player_id):
    reg = load("player_regular")
    rows = reg[reg.PLAYER_ID == player_id]
    if rows.empty:
        return None
    adv = load("advanced_regular")
    players, contracts = load("players"), load("contracts")
    prior = contracts.loc[contracts.PLAYER_ID == player_id, "PRIOR_PAY"]
    prior = float(prior.iloc[0]) if len(prior) and pd.notna(prior.iloc[0]) else None
    return {
        "GP": int(rows.GAME_ID.nunique()),
        "MPG": float(rows.MIN.mean()),
        "PPG": float(rows.PTS.mean()),
        "TS": true_shooting(rows),
        "USG": float(adv[adv.PLAYER_ID == player_id].USG_PCT.mean()),
        "AGE": int(players.loc[players.PLAYER_ID == player_id, "AGE"].iloc[0]),
        "PRIOR_PAY": prior or 0.0,
        "HAS_PRIOR": int(prior is not None),
    }


# ---------- result checks ----------

def check_stat(form, result):
    out = []
    subject = result.get("subject", {})
    playoff_ids = set(load("player_playoffs").GAME_ID) | set(load("team_playoffs").GAME_ID)
    regular_ids = set(load("team_regular").GAME_ID)
    units = result.get("units", {})

    if "comparison" in result:
        c = result["comparison"]
        if units.get(c.get("a")) != units.get(c.get("b")):
            out.append(fail("rate_basis", f"{c.get('a')} and {c.get('b')} have different units and were compared"))

    for w in result.get("windows", []):
        ids = list(w.get("game_ids", []))
        label = w.get("label", "?")
        mixed = [g for g in ids if g in playoff_ids or g not in regular_ids]
        if mixed:
            out.append(fail("playoff_mix", f"window '{label}' has {len(mixed)} non-regular-season games, e.g. {mixed[0]}"))
        if not w.get("games") or w["games"] != len(ids):
            out.append(fail("no_game_count", f"window '{label}' game count is missing or does not match its game list"))
        for key in w.get("values", {}):
            if key not in units:
                out.append(fail("rate_basis", f"value '{key}' in window '{label}' has no stated unit"))

        if subject.get("kind") == "player":
            reg = load("player_regular")
            rows = reg[(reg.PLAYER_ID == subject["id"]) & reg.GAME_ID.isin(ids)]
            if "team_id" in w and (rows.TEAM_ID != w["team_id"]).any():
                bad = rows[rows.TEAM_ID != w["team_id"]].GAME_ID.iloc[0]
                out.append(fail("wrong_team", f"window '{label}' charges game {bad} to a team he was not on"))

        if not mixed:
            for key, value in w.get("values", {}).items():
                truth = recompute(key, subject, ids)
                if not close(value, truth):
                    out.append(fail("stat_mismatch", f"{key} in '{label}' is {value}, saved box score gives {truth:.4f}"))

    if result.get("metric") == "trade_split" and subject.get("kind") == "player":
        trades = load("trades")
        t = trades[trades.PLAYER_ID == subject["id"]]
        if t.empty:
            out.append(fail("wrong_team", "no saved trade for this player"))
        else:
            t = t.iloc[0]
            wins = {w.get("label"): w for w in result.get("windows", [])}
            if wins.get("before", {}).get("team_id") != t.FROM_TEAM_ID or wins.get("after", {}).get("team_id") != t.TO_TEAM_ID:
                out.append(fail("wrong_team", "before/after windows are not the from/to teams in trades.parquet"))
    return out


def check_context(form, result):
    out = []
    saved = load("paragraphs").set_index("PARA_ID")
    subject = result.get("subject", {})
    window = result.get("window") or {}
    start = pd.Timestamp(window["start"]) if window.get("start") else None
    end = pd.Timestamp(window["end"]) if window.get("end") else None
    for p in result.get("paragraphs", []):
        if p["para_id"] not in saved.index:
            out.append(fail("quote_not_saved", f"paragraph {p['para_id']} is not in paragraphs.parquet"))
            continue
        row = saved.loc[p["para_id"]]
        if row.PLAYER_ID != subject.get("id"):
            out.append(fail("wrong_paragraph", f"paragraph {p['para_id']} is about another player"))
        if (start is not None and row.DATE < start) or (end is not None and row.DATE > end):
            out.append(fail("wrong_paragraph", f"paragraph {p['para_id']} dated {row.DATE.date()} is outside the window"))
    kept = {p["para_id"] for p in result.get("paragraphs", [])}
    for q in result.get("quotes", []):
        if q["para_id"] not in kept or q["para_id"] not in saved.index or q["text"] not in saved.loc[q["para_id"], "TEXT"]:
            out.append(fail("quote_not_saved", f"quote '{q['text'][:40]}...' is not word for word in a kept paragraph"))
    facts = result.get("box_facts", {})
    if "games_in_window" in facts:
        reg = load("player_regular")
        rows = reg[reg.PLAYER_ID == subject.get("id")]
        if start is not None:
            rows = rows[rows.GAME_DATE >= start]
        if end is not None:
            rows = rows[rows.GAME_DATE <= end]
        if facts["games_in_window"] != rows.GAME_ID.nunique():
            out.append(fail("stat_mismatch", "games_in_window does not match the saved box score"))
    return out


def check_signing(form, result):
    out = []
    feats = result.get("features")
    truth = signing_features(result.get("subject", {}).get("id"))
    if not feats or truth is None or not feats.get("GP"):
        return [fail("missing_stats", "no regular-season stats for this player; prediction rejected")]
    for k in ("GP", "MPG", "PPG", "TS", "USG", "AGE"):
        if feats.get(k) is None or not close(feats[k], truth[k]):
            out.append(fail("stat_mismatch", f"feature {k}={feats.get(k)} does not match saved stats ({truth[k]})"))
    if result.get("label") != "prediction":
        out.append(fail("unlabeled_prediction", "signing figure is not labeled as a prediction"))
    if result.get("heldout_mae") is None:
        out.append(fail("no_heldout_error", "signing figure has no held-out error attached"))
    if result.get("prediction") is None or result["prediction"] <= 0:
        out.append(fail("missing_stats", "no positive prediction was produced"))
    return out


def check_result(form, result):
    if not isinstance(result, dict) or "error" in result:
        return [fail("code_error", str(result.get("error") if isinstance(result, dict) else result))]
    if result.get("metric") != form.get("metric"):
        return [fail("wrong_metric", f"result is {result.get('metric')}, form asked {form.get('metric')}")]
    if form["metric"] == "context":
        return check_context(form, result)
    if form["metric"] == "signing":
        return check_signing(form, result)
    return check_stat(form, result)


# ---------- note checks ----------

def allowed_numbers(result):
    vals = []
    for w in result.get("windows", []):
        vals.append(w.get("games"))
        for key, v in w.get("values", {}).items():
            vals.append(v)
            if key == "ts_pct":
                vals.append(v * 100)
    if result.get("metric") == "signing":
        for v in (result.get("prediction"), result.get("heldout_mae")):
            if v is not None:
                vals += [v, v / 1e6, v / 1e3]
        for k, v in (result.get("features") or {}).items():
            vals += [v, v / 1e6] if k == "PRIOR_PAY" else [v]
            if k in ("TS", "USG"):
                vals.append(v * 100)
    for v in (result.get("box_facts") or {}).values():
        if isinstance(v, (int, float)):
            vals.append(v)
    return [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(float(v))]


def number_matches(token, allowed):
    raw = token.replace("$", "").replace(",", "").replace("%", "").rstrip("M")
    try:
        x = float(raw)
    except ValueError:
        return True
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(x - a) <= tol for a in allowed)


def strip_for_numbers(note):
    text = TAG.sub(" ", note)
    text = QUOTE.sub(" ", text)
    text = re.sub(r"\bp\d{4}\b|\br\d{6}\b|\b00[24]25\d{5}\b", " ", text)
    for name in load("teams").TEAM_NAME:
        text = text.replace(name, " ")
    text = text.replace("76ers", " ")
    return DATES.sub(" ", text)


def check_note(form, result, note):
    out = []
    metric = form.get("metric")
    body = note.split("\nSources")[0]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    labels = {w.get("label") for w in result.get("windows", [])}
    para_ids = {p["para_id"] for p in result.get("paragraphs", [])}

    for s in sentences:
        if not NUMBER.search(strip_for_numbers(s)):
            continue
        tags = TAG.findall(s)
        if not tags:
            out.append(fail("no_game_list", f"sentence has a number but no citation: '{s.strip()[:70]}'"))
            continue
        for kind, ref in tags:
            if kind == "W" and ref not in labels:
                out.append(fail("no_game_list", f"citation [W:{ref}] does not match a window"))
            if kind == "P" and ref not in para_ids:
                out.append(fail("quote_not_saved", f"citation [P:{ref}] is not a kept paragraph"))

    allowed = allowed_numbers(result)
    saved = load("paragraphs").set_index("PARA_ID")
    cited_text = " ".join(saved.loc[p, "TEXT"] for p in para_ids if p in saved.index)
    for token in NUMBER.findall(strip_for_numbers(body)):
        if not number_matches(token, allowed):
            if metric == "context" and token.strip("$%M") in cited_text:
                out.append(fail("article_number", f"number {token} was taken from an article, not the box score"))
            else:
                out.append(fail("stat_mismatch", f"number {token} in the note is not in the checked result"))

    if metric == "context":
        for q in QUOTE.findall(body):
            if q not in cited_text:
                out.append(fail("quote_not_saved", f"quoted text '{q[:40]}' is not in a kept paragraph"))
    if metric == "signing":
        if "predict" not in body.lower():
            out.append(fail("unlabeled_prediction", "note does not call the figure a prediction"))
        mae = result.get("heldout_mae")
        if mae is not None and not any(number_matches(t, [mae, mae / 1e6, mae / 1e3]) for t in NUMBER.findall(body)):
            out.append(fail("no_heldout_error", "note does not state the held-out error"))
        if CAP_WORDS.search(note):
            out.append(fail("cap_claim", "note mentions cap room or a recommendation"))
    return out
