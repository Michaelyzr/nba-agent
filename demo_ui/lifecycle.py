"""Translate internal traces into safe, structured product stages."""
from __future__ import annotations


STAGES = (
    ("Observe", {"trigger", "investigate"}),
    ("Forecast", {"forecast"}),
    ("Compare market", {"analyse", "propose", "no_action"}),
    ("Safety check", {"checks", "risk", "blocked"}),
    ("Act / pass", {"confirm", "deliver"}),
    ("Market grade", set()),
    ("Review / learn", set()),
)


def execution_stages(trace: list[dict], phase: str, acted: bool) -> list[dict]:
    """Expose stage names and terse outcomes, never hidden reasoning or chain-of-thought."""
    seen = {item.get("step") for item in trace}
    out = []
    for index, (label, steps) in enumerate(STAGES):
        ran = bool(steps & seen)
        if index <= 4:
            state = "done" if ran else "pending"
            detail = "completed"
            if label == "Act / pass" and ran:
                detail = "paper action" if acted else "pass"
        elif label == "Market grade":
            state = "done" if phase == "After game" else "later"
            detail = ("settled against close" if acted else "no position to grade") if state == "done" else "after tip-off"
        else:
            state = "available" if phase == "After game" else "later"
            detail = "full-day reviewer is separate" if state == "available" else "after a replay day"
        out.append({"label": label, "state": state, "detail": detail})
    return out
