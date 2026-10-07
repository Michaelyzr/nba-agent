"""H: reviewer rules written in a small, safe language, compiled to the notebook's when/do format.

A rule is one line:

    when side_price <= 0.30 and news_age_minutes >= 30 do skip
    when gap <= 0.07 do min_edge 0.07
    when market_kind == game and market_move <= -0.02 do stake_scale 0.5

Grammar (hand-written tokenizer and recursive-descent parser; nothing is ever passed to eval):

    rule       := "when" condition ("and" condition)* "do" action
    condition  := RANGE_FIELD ("<=" | ">=") NUMBER  |  EQUAL_FIELD "==" WORD
    action     := "skip" | "min_edge" NUMBER | "stake_scale" NUMBER

Only conjunctions; at most MAX_CONDITIONS conditions; every constant is bounded per
field (BOUNDS). A parsed rule compiles to {"when": {field_min|field_max: value, ...},
"do": {"action": ..., "params": {...}}}, which notebook.validate checks again and
MarketAgent.analyse already applies. The compiled rule still has to pass the gate.

DSLReviewAgent swaps the template reviewer for a proposer that writes up to K rules
a day in this language: Gemini when a key is set (responses cached under
runs/llm_cache/), otherwise a deterministic stub that searches the same language.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import MarketAgent
from agents.notebook import LIMITS, matches, validate

MAX_TEXT = 240
MAX_CONDITIONS = 3
MAX_PROPOSALS = 3                # proposals per review day (K)
# Situation fields the trader sees (MarketAgent.analyse) and the range each constant must lie in.
BOUNDS = {
    "side_price": (0.01, 0.99),        # price paid for the side bought
    "gap": (0.0, 0.50),                # edge after fees
    "market_move": (-0.50, 0.50),      # price move toward the trade since the 24 h anchor
    "model_shift": (0.0, 0.50),        # size of the model's news adjustment
    "hours_to_tip": (0.0, 12.0),
    "news_age_minutes": (0.0, 10000.0),
}
EQUAL_FIELDS = {"market_kind": re.compile(r"game|spread|total|player"),
                "team": re.compile(r"[A-Z]{2,4}"), "opponent": re.compile(r"[A-Z]{2,4}")}
ACTION_BOUNDS = {"min_edge": ("edge", 0.041, 0.20), "stake_scale": ("scale", 0.10, 0.99)}
ACTION_ALIASES = {"skip": "skip_market", "skip_market": "skip_market", "min_edge": "min_edge",
                  "stake_scale": "stake_scale"}
TOKEN = re.compile(r"\s*(?:(?P<op><=|>=|==)|(?P<num>-?\d+(?:\.\d+)?)|(?P<word>[A-Za-z_][A-Za-z_]*))")
CACHE = Path(__file__).resolve().parent.parent / "runs" / "llm_cache"


class DSLError(ValueError):
    """A proposal the parser or validator rejects; str(err) is the reason."""


def tokenize(text: str) -> list:
    if not isinstance(text, str):
        raise DSLError("rule must be text")
    if len(text) > MAX_TEXT:
        raise DSLError(f"rule longer than {MAX_TEXT} characters")
    text = text.strip().rstrip(".;").strip()
    tokens, pos = [], 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise DSLError(f"unexpected character {text[pos:pos + 1]!r} at {pos}")
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return tokens


def parse(text: str) -> dict:
    """DSL text -> {"conditions": [(field, op, value)], "action": name, "param": value or None}."""
    toks = tokenize(text)
    i = 0

    def take(kind=None, value=None):
        nonlocal i
        if i >= len(toks):
            raise DSLError(f"rule ends early, expected {value or kind}")
        k, v = toks[i]
        if (kind and k != kind) or (value and v.lower() != value.lower()):
            raise DSLError(f"expected {value or kind}, found {v!r}")
        i += 1
        return v

    take("word", "when")
    conditions = []
    while True:
        field = take("word").lower()
        op = take("op")
        if field in BOUNDS:
            if op == "==":
                raise DSLError(f"{field} needs <= or >=")
            value = float(take("num"))
            lo, hi = BOUNDS[field]
            if not lo <= value <= hi:
                raise DSLError(f"{field} constant {value} outside [{lo}, {hi}]")
        elif field in EQUAL_FIELDS:
            if op != "==":
                raise DSLError(f"{field} only allows ==")
            value = take("word")
            if not EQUAL_FIELDS[field].fullmatch(value):
                raise DSLError(f"{field} value {value!r} not allowed")
        else:
            raise DSLError(f"unknown field {field!r}")
        conditions.append((field, op, value))
        if len(conditions) > MAX_CONDITIONS:
            raise DSLError(f"more than {MAX_CONDITIONS} conditions")
        if i < len(toks) and toks[i][0] == "word" and toks[i][1].lower() == "and":
            i += 1
            continue
        break
    take("word", "do")
    word = take("word")
    action = ACTION_ALIASES.get(word.lower())
    if action is None:
        raise DSLError(f"unknown action {word!r}")
    param = None
    if action in ACTION_BOUNDS:
        _, lo, hi = ACTION_BOUNDS[action]
        param = float(take("num"))
        if not lo <= param <= hi:
            raise DSLError(f"{action} constant {param} outside [{lo}, {hi}]")
    if i != len(toks):
        raise DSLError(f"unexpected text after the action: {toks[i][1]!r}")
    return {"conditions": conditions, "action": action, "param": param}


def compile_rule(parsed: dict) -> dict:
    """Parsed rule -> notebook when/do. Rejects repeated bounds and empty ranges."""
    when = {}
    for field, op, value in parsed["conditions"]:
        key = field if op == "==" else f"{field}_{'max' if op == '<=' else 'min'}"
        if key in when:
            raise DSLError(f"{key} given twice")
        when[key] = value if op == "==" else round(float(value), 4)
    for f in BOUNDS:
        if f"{f}_min" in when and f"{f}_max" in when and when[f"{f}_min"] > when[f"{f}_max"]:
            raise DSLError(f"{f} range is empty")
    params = {}
    if parsed["action"] in ACTION_BOUNDS:
        params = {ACTION_BOUNDS[parsed["action"]][0]: parsed["param"]}
    rule = {"when": {"market_kind": "game", **when}, "do": {"action": parsed["action"], "params": params}}
    problems = validate(rule)
    if problems:
        raise DSLError("; ".join(problems))
    return rule


def compile_text(text: str) -> dict:
    return compile_rule(parse(text))


def check(text: str) -> tuple:
    """(compiled rule, None) or (None, reason)."""
    try:
        return compile_text(text), None
    except DSLError as err:
        return None, str(err)


def to_text(rule: dict) -> str:
    """Notebook when/do -> DSL text (inverse of compile_text for rules the DSL can express)."""
    conds = []
    for key, v in rule["when"].items():
        if key in EQUAL_FIELDS:
            conds.append(f"{key} == {v}")
        else:
            field, end = key.rsplit("_", 1)
            conds.append(f"{field} {'<=' if end == 'max' else '>='} {v:g}")
    do = rule["do"]
    act = "skip" if do["action"] == "skip_market" else f"{do['action']} {next(iter(do['params'].values())):g}"
    return "when " + " and ".join(conds) + " do " + act


def predicate(text: str):
    """A callable situation -> bool for the rule's condition (same semantics as notebook.matches)."""
    rule = compile_text(text)
    return lambda situation: matches(rule, situation)


# ---------------- proposers ----------------

def cached_ask(prompt: str, model: str, ask=None, cache: Path = CACHE) -> dict:
    """LLM JSON answer cached by sha256(model, prompt); temperature 0. A hit never calls the model."""
    key = hashlib.sha256(json.dumps({"model": model, "prompt": prompt, "temperature": 0}).encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    if ask is None:
        import llm
        ask = llm.ask
    response = ask(prompt, want_json=True)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "response": response}, default=str))
    return response


PROMPT = """You review losing trades of a sports-market agent and propose rules that would have avoided them.
Write at most {k} rules, one per string, in this language and nothing else:

  when <condition> [and <condition>]* do <action>
  condition: <field> <= <number> | <field> >= <number> | market_kind == game
  fields: side_price (price paid, 0.01-0.99), gap (edge after fees, 0-0.5), market_move (price move toward the
          trade since a day earlier, -0.5-0.5), model_shift (size of the news adjustment, 0-0.5),
          hours_to_tip (0-12), news_age_minutes (0-10000)
  action: skip | min_edge <0.041-0.20> | stake_scale <0.10-0.99>

At most {c} conditions per rule. Rules are back-tested on earlier days before use, so prefer simple rules
that cover several trades. Return JSON {{"rules": ["when ... do ...", ...], "rationale": "one sentence"}}.
Trades settled before {now} (clv < 0 means the price moved against the trade):
{trades}"""


def llm_proposer(k=MAX_PROPOSALS, ask=None, model=None):
    def propose(trades: pd.DataFrame, now) -> list:
        import llm
        cols = ["side_price", "gap", "market_move", "model_shift", "hours_to_tip", "news_age_minutes", "clv",
                "contracts"]
        rows = trades[[c for c in cols if c in trades]].round(3).to_json(orient="records")
        prompt = PROMPT.format(k=k, c=MAX_CONDITIONS, now=pd.Timestamp(now).isoformat(), trades=rows)
        raw = cached_ask(prompt, model or llm.MODEL, ask)
        rules = raw.get("rules", []) if isinstance(raw, dict) else []
        return [str(r) for r in rules][:k] if isinstance(rules, list) else []
    return propose


GRID = {"side_price": [0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75],
        "gap": [0.06, 0.08, 0.1, 0.12, 0.15],
        "market_move": [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
        "model_shift": [0.005, 0.01, 0.02, 0.03, 0.05],
        "hours_to_tip": [0.75, 1.5, 3.0],
        "news_age_minutes": [15, 30, 60, 120]}


def stub_proposer(k=MAX_PROPOSALS, min_hits=2 * LIMITS["min_cases"], max_share=0.5):
    """Deterministic stand-in for the LLM: greedy search over one- and two-condition skip rules.

    Scores every grid threshold (and every pair of thresholds on different fields) by the
    CLV dollars of the recent trades it covers, keeps rules covering >= min_hits trades (and
    at most max_share of them, so "skip everything" is not a lesson) with negative mean CLV,
    and returns a ranked list, worst first; the agent gates the first k it has not already
    tried. Same language, inputs and gate as the LLM; it searches instead of reading.
    """
    def propose(trades: pd.DataFrame, now) -> list:
        if trades.empty:
            return []
        dollars = (trades.clv * trades.contracts).to_numpy()
        singles = []
        for f, values in GRID.items():
            if f not in trades:
                continue
            x = trades[f].to_numpy(float)
            for v in values:
                for op, mask in (("<=", x <= v), (">=", x >= v)):
                    singles.append((f"{f} {op} {v:g}", f, mask))
        found = []
        for i, (t1, f1, m1) in enumerate(singles):
            found.append(([t1], m1))
            for t2, f2, m2 in singles[i + 1:]:
                if f2 != f1:
                    found.append(([t1, t2], m1 & m2))
        scored = []
        for conds, mask in found:
            n = int(mask.sum())
            if n < min_hits or n > max_share * len(trades) or trades.clv.to_numpy()[mask].mean() >= 0:
                continue
            # Penalise the second condition a little so a single condition wins ties.
            scored.append((float(dollars[mask].sum()) + 0.25 * (len(conds) - 1), -n, conds))
        scored.sort(key=lambda s: (s[0], s[1], s[2]))
        out = []
        for _, _, conds in scored:
            text = "when " + " and ".join(conds) + " do skip"
            if text not in out:
                out.append(text)
            if len(out) == 10 * k:
                break
        return out
    return propose


# ---------------- the agent ----------------

class DSLReviewAgent(MarketAgent):
    """MarketAgent whose reviewer writes DSL rules instead of picking from REVIEW_TEMPLATES.

    Plugs in through MarketAgent._review_offline (review() still runs validate, the
    notebook's seen() check and the unchanged gate). Each review day the proposer writes up
    to K rules from the recent settled trades; each one that parses is sent through the
    review -> gate path in turn. Every proposal and its fate is kept in self.proposals.
    """

    def __init__(self, *args, proposer=None, k=MAX_PROPOSALS, **kwargs):
        # Prefer the de-overlapped total-CLV-dollars gate when the caller does not pick one.
        kwargs.setdefault("gate", "split")
        super().__init__(*args, **kwargs)
        self.k = k
        if proposer is None:
            import llm
            proposer = llm_proposer(k) if llm.available() else stub_proposer(k)
            self.proposer_name = f"gemini/{llm.MODEL}" if llm.available() else "stub-search"
        else:
            self.proposer_name = getattr(proposer, "__name__", "custom")
        self.proposer = proposer
        self.proposals = []
        self._queue = []
        self._current = self._last = None

    def with_situations(self, fills: pd.DataFrame) -> pd.DataFrame:
        """Attach decision-time situation fields so the proposer can read them."""
        if fills is None or fills.empty:
            return fills
        sit = self.situation_frame(fills)
        keep = [c for c in ["market_kind", *BOUNDS] if c in sit]
        out = fills.copy()
        for c in keep:
            out[c] = sit[c]
        return out

    def on_day_end(self, day, fills, rp):
        if not self.learn:
            return
        all_fills = pd.DataFrame(rp.fills)
        self._queue = []
        if len(all_fills):
            games = rp.t["games"]
            now = games.loc[games.date == day, "final_at"].max()
            # Same selection window the parent reviewer uses (split: last SELECT_DAYS; legacy: 21 days).
            selected = self.with_situations(self.selection(rp, all_fills, day))
            for text in self.proposer(selected, now):
                if len(self._queue) == self.k:
                    break
                rule, why = check(text)
                if rule is not None and self.notebook.seen(rule, now):
                    continue            # tried recently; the template reviewer skips these too
                entry = {"day": day, "text": text, "valid": rule is not None, "reason": why,
                         "proposer": self.proposer_name, "gate": None, "rule_id": None, "when": None,
                         "gate_mode": self.gate_mode}
                self.proposals.append(entry)
                if rule is not None:
                    entry["when"] = json.dumps(rule["when"])
                    self._queue.append((entry, rule))
        for _ in range(max(1, len(self._queue))):
            super().on_day_end(day, fills, rp)
            if self._current is not None:
                self._current["gate"] = self._outcome(self.traces[-1]["trace"])
                if self._current["gate"] in ("accepted", "rejected"):
                    self._current["rule_id"] = self.notebook.rules[-1]["rule_id"]

    @staticmethod
    def _outcome(trace) -> str:
        steps = {s["step"]: s["detail"] for s in trace}
        if "save_rule" in steps:
            return "accepted"
        if "reject_rule" in steps:
            return "deferred" if "not recorded" in steps["reject_rule"] else "rejected"
        return "dropped"

    def _review_offline(self, recent):
        """`recent` is already the selection window (parent.review). Emit the next queued DSL rule."""
        self._current = self._last = None
        if not self._queue:
            return None
        entry, rule = self._queue.pop(0)
        self._current = entry
        sit = self.with_situations(recent)
        hit = sit[[matches(rule, s) for s in sit.to_dict("records")]] if len(sit) else sit
        dollars = float((hit.clv * hit.contracts).sum()) if len(hit) else 0.0
        mean = float(hit.clv.mean()) if len(hit) else float("nan")
        self._last = self._rule(rule["when"], rule["do"], "model_wrong",
                                f"DSL ({self.proposer_name}): {entry['text']}; {len(hit)} selection trades, mean CLV "
                                f"{mean:+.3f} ({dollars:+.2f} dollars).", hit)
        return self._last

    def _review_llm(self, fills):
        """review() calls this after _review_offline when an LLM is on; the proposal is already chosen."""
        return self._last if self._current is not None else None
