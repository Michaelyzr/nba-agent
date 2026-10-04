"""The one agent: a LangGraph checklist. It knows the order of steps and the stop rules, not basketball.

Run:  python graph.py "Did Marcus Hale's true shooting change after he was traded this season?"
      python graph.py --plant playoffs "..."   (inject a fault to show the retry loop)
"""
import argparse
import operator
import uuid
from typing import Annotated, Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from nba_agent.agents import steps
from nba_agent.data.tables import MAX_TRIES


class State(TypedDict, total=False):
    question: str
    answer: Optional[str]
    plant: Optional[str]
    mode: str
    form: dict
    missing: list
    followup: Optional[str]
    plan: dict
    ranked: list
    code: str
    result: dict
    features: Optional[dict]
    failures: list
    tries: int
    note: str
    note_result: dict
    note_failures: list
    note_tries: int
    status: str
    trace: Annotated[list, operator.add]


def after_clarify(state):
    if state["missing"]:
        return "wait"
    return "plan"


def after_plan(state):
    metric = state["form"]["metric"]
    if metric == "signing":
        return "build_features"
    if metric == "context":
        return "write_filter"
    return "rank_recaps"


def after_check(state):
    if not state["failures"]:
        return "write_note"
    if state["tries"] >= MAX_TRIES:
        return "unresolved"
    metric = state["form"]["metric"]
    if metric == "signing":
        return "build_features"
    if metric == "context":
        return "write_filter"
    return "write_pandas"


def after_support(state):
    if state.get("status") == "answered":
        return END
    if state["note_tries"] >= MAX_TRIES:
        return "unresolved"
    return "write_note"


def build():
    g = StateGraph(State)
    g.add_node("clarify", steps.clarify)
    g.add_node("wait", steps.wait)
    g.add_node("plan", steps.plan)
    g.add_node("rank_recaps", steps.rank_recaps)
    g.add_node("write_pandas", steps.write_code)
    g.add_node("run_pandas", steps.run_code)
    g.add_node("write_filter", steps.write_code)
    g.add_node("run_filter", steps.run_code)
    g.add_node("rank_paragraphs", steps.rank_paragraphs)
    g.add_node("build_features", steps.build_features)
    g.add_node("predict_pay", steps.predict_pay)
    g.add_node("check", steps.check)
    g.add_node("write_note", steps.write_note)
    g.add_node("support_check", steps.support_check)
    g.add_node("unresolved", steps.unresolved)

    g.add_edge(START, "clarify")
    g.add_conditional_edges("clarify", after_clarify, ["wait", "plan"])
    g.add_edge("wait", "clarify")
    g.add_conditional_edges("plan", after_plan, ["rank_recaps", "write_filter", "build_features"])
    g.add_edge("rank_recaps", "write_pandas")
    g.add_edge("write_pandas", "run_pandas")
    g.add_edge("run_pandas", "check")
    g.add_edge("write_filter", "run_filter")
    g.add_edge("run_filter", "rank_paragraphs")
    g.add_edge("rank_paragraphs", "check")
    g.add_edge("build_features", "predict_pay")
    g.add_edge("predict_pay", "check")
    g.add_conditional_edges("check", after_check,
                            ["write_note", "unresolved", "write_pandas", "write_filter", "build_features"])
    g.add_edge("write_note", "support_check")
    g.add_conditional_edges("support_check", after_support, [END, "unresolved", "write_note"])
    g.add_edge("unresolved", END)
    return g.compile(checkpointer=MemorySaver())


AGENT = build()


def start(question, plant=None, thread_id=None):
    """Run until the note is done or the agent pauses for a follow-up."""
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}, "recursion_limit": 60}
    AGENT.invoke({"question": question, "plant": plant, "tries": 0, "note_tries": 0, "trace": []}, config)
    return snapshot(config)


def reply(answer, config):
    AGENT.invoke(Command(resume=answer), config)
    return snapshot(config)


def snapshot(config) -> dict[str, Any]:
    s = AGENT.get_state(config)
    values = dict(s.values)
    values["config"] = config
    values["paused"] = bool(s.next)
    return values


def print_trace(trace, seen=0):
    for t in trace[seen:]:
        print(f"  {t['step']:<16} {t['detail']}")
    return len(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--plant", choices=sorted(steps.FAULTS), default=None)
    args = ap.parse_args()

    out = start(args.question, args.plant)
    seen = print_trace(out["trace"])
    while out["paused"]:
        print(f"\nAgent: {out['followup']}")
        out = reply(input("You: "), out["config"])
        seen = print_trace(out["trace"], seen)
    print("\n" + out["note"])


if __name__ == "__main__":
    main()
