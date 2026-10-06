"""Rule notebook: lessons the reviewer proposed and the gate approved.

Rules use a closed vocabulary so code can test the condition and apply the
action. Only the gate sets a rule active, and an active rule applies only to
decisions at or after its valid_from, so a back-test never uses a lesson from
the future. Rules are never deleted; rejected and retired rules stay as history.
"""
import copy
import json
from pathlib import Path

import pandas as pd

from nba_agent_paths import NOTEBOOK
SCHEMA_VERSION = "1.0"
LIMITS = {"max_active_rules": 20, "min_cases": 3, "default_expiry_days": 45, "retry_after_days": 21}

WHEN_EQUAL = {"team", "opponent", "ruled_out_player", "status", "news_type", "market_kind", "back_to_back"}
# side_price: price of the side the agent would buy; gap: edge after fees; market_move: how far the
# price moved toward that side since the anchor; model_shift: size of the model's news adjustment.
WHEN_RANGE = {"hours_to_tip", "news_age_minutes", "losing_sessions", "side_price", "gap", "market_move",
              "model_shift"}
WHEN_FIELDS = WHEN_EQUAL | {f"{f}_{end}" for f in WHEN_RANGE for end in ("min", "max")}
ACTIONS = {
    "minutes_share": "forecast", "minutes_cap": "forecast", "p_play_adjust": "forecast",
    "skip_market": "trader", "min_edge": "trader", "stake_scale": "trader",
}
STATUSES = ("proposed", "active", "rejected", "retired")


def validate(rule: dict) -> list:
    """Problems that stop a rule entering the notebook; empty means valid."""
    out = []
    unknown = set(rule.get("when", {})) - WHEN_FIELDS
    if unknown:
        out.append(f"unknown condition fields: {sorted(unknown)}")
    action = rule.get("do", {}).get("action")
    if action not in ACTIONS:
        out.append(f"unknown action: {action}")
    params = rule.get("do", {}).get("params", {})
    if action == "stake_scale" and not 0 < params.get("scale", 1) <= 1:
        out.append("stake_scale may only reduce stakes")
    if action == "min_edge" and params.get("edge", 0) <= 0:
        out.append("min_edge needs a positive edge")
    return out


def matches(rule: dict, situation: dict) -> bool:
    for key, want in rule.get("when", {}).items():
        if key in WHEN_EQUAL:
            if situation.get(key) != want:
                return False
            continue
        field, end = key.rsplit("_", 1)
        have = situation.get(field)
        if have is None or (end == "min" and have < want) or (end == "max" and have > want):
            return False
    return True


class Notebook:
    def __init__(self, rules=None, path: Path | None = None):
        self.rules = list(rules or [])
        self.path = path

    @classmethod
    def load(cls, path: Path = NOTEBOOK):
        if not path.exists():
            return cls(path=path)
        return cls(json.loads(path.read_text()).get("rules", []), path)

    def save(self, path: Path | None = None) -> Path:
        path = path or self.path or NOTEBOOK
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema_version": SCHEMA_VERSION, "limits": LIMITS, "rules": self.rules}
        path.write_text(json.dumps(body, indent=2, default=str))
        return path

    def active(self, now, kind: str | None = None) -> list:
        now = pd.Timestamp(now)
        out = []
        for r in self.rules:
            if r["status"] != "active" or (kind and ACTIONS[r["do"]["action"]] != kind):
                continue
            start = r.get("valid_from")
            if start is not None and pd.Timestamp(start) > now:
                continue
            expiry = r.get("expires_after_days")
            if start is not None and expiry and now > pd.Timestamp(start) + pd.Timedelta(days=expiry):
                continue
            out.append(r)
        return sorted(out, key=lambda r: -r.get("priority", 1))

    def matching(self, situation: dict, now, kind: str) -> list:
        return [r for r in self.active(now, kind) if matches(r, situation)]

    def get(self, rule_id: str):
        return next((r for r in self.rules if r["rule_id"] == rule_id), None)

    def next_id(self) -> str:
        return f"r{len(self.rules) + 1:03d}"

    def seen(self, rule: dict, now) -> bool:
        """True if the same condition and action is active now or was rejected recently.

        Expired rules may be renewed, and a rejected rule may be proposed again after
        retry_after_days, when the gate has more earlier days to test it on.
        """
        now = pd.Timestamp(now)
        live = {id(r) for r in self.active(now)}
        retry = pd.Timedelta(days=LIMITS["retry_after_days"])

        def recent_reject(r):
            return r["status"] == "rejected" and (r.get("proposed_at") is None
                                                  or now - pd.Timestamp(r["proposed_at"]) < retry)
        return any(r["when"] == rule["when"] and r["do"] == rule["do"] and (id(r) in live or recent_reject(r))
                   for r in self.rules)

    def record(self, rule: dict):
        problems = validate(rule)
        if problems:
            raise ValueError("; ".join(problems))
        if rule["status"] == "active" and len(self.active(rule["valid_from"])) >= LIMITS["max_active_rules"]:
            raise ValueError("notebook already holds the maximum number of active rules")
        self.rules.append(rule)

    def without(self, rule: dict) -> "Notebook":
        """Copy without any rule that has the candidate's condition and action (e.g. an expired version)."""
        return Notebook([copy.deepcopy(r) for r in self.rules if (r["when"], r["do"]) != (rule["when"], rule["do"])])

    def with_candidate(self, rule: dict) -> "Notebook":
        """Copy where the candidate counts as active on every day, for back-testing it."""
        trial = copy.deepcopy(rule)
        trial.update(status="active", valid_from=None)
        return Notebook(self.without(rule).rules + [trial])
