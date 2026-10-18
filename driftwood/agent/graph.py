"""Collect, propose, stop, and only then act.

The compiled graph interrupts before ``execute``. The propose node returns
proposals and nothing else, so a model that emits ``approved_ids`` does not
write them into the state. A person sets that list with ``update_state``.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from driftwood.agent.actions import Toolset, run_actions
from driftwood.agent.context import build_context
from driftwood.agent.proposals import message_text, parse_proposals, system_prompt


class ResearchState(TypedDict, total=False):
    db_path: str
    model_name: str
    briefing: str
    as_of: str
    context: str
    proposals: list
    approved_ids: list
    notes: list
    results: list
    report: str
    report_path: str


def build_graph(model: object, tools: Toolset):
    """Compile the graph. ``model`` only needs an ``invoke`` method."""

    def collect(state: ResearchState) -> dict[str, object]:
        as_of, context = build_context(
            state["db_path"],
            str(state.get("model_name") or tools.model_name),
            str(state.get("briefing") or ""),
        )
        return {"as_of": as_of, "context": context}

    def propose(state: ResearchState) -> dict[str, object]:
        message = model.invoke(
            [
                SystemMessage(content=system_prompt()),
                HumanMessage(content=str(state.get("context") or "")),
            ]
        )
        # The return value is proposals only. approved_ids is not a model output.
        return {"proposals": parse_proposals(message_text(message))}

    def execute(state: ResearchState) -> dict[str, object]:
        return run_actions(
            as_of=str(state.get("as_of") or "unknown"),
            context=str(state.get("context") or ""),
            proposals=list(state.get("proposals") or []),
            approved_ids=[str(item) for item in state.get("approved_ids") or []],
            tools=tools,
        )

    graph = StateGraph(ResearchState)
    graph.add_node("collect", collect)
    graph.add_node("propose", propose)
    graph.add_node("execute", execute)
    graph.add_edge(START, "collect")
    graph.add_edge("collect", "propose")
    graph.add_edge("propose", "execute")
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["execute"])


def run_session(
    model: object,
    tools: Toolset,
    *,
    briefing: str = "",
    approve: list[str] | None = None,
    thread_id: str = "driftwood",
) -> dict[str, object]:
    """Run until the approval stop. Resume only when ``approve`` is a list.

    ``None`` means the person has not answered, so ``execute`` does not run.
    A list, including an empty one, is written with ``update_state`` and is
    the only source of ``approved_ids``.
    """

    graph = build_graph(model, tools)
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(
        {
            "db_path": tools.db_path,
            "model_name": tools.model_name,
            "briefing": briefing,
            "approved_ids": [],
            "proposals": [],
        },
        config,
    )
    snapshot = graph.get_state(config)
    if snapshot.next != ("execute",):
        raise RuntimeError("The agent did not stop before acting.")
    if approve is None:
        values = dict(snapshot.values)
        values.setdefault("results", [])
        values.setdefault("report", "")
        values.setdefault("report_path", "")
        return values
    graph.update_state(config, {"approved_ids": [str(item) for item in approve]})
    graph.invoke(None, config)
    return dict(graph.get_state(config).values)
