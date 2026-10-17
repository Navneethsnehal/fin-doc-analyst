"""LangGraph wiring. The picture in the README matches these edges."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from fin_doc_analyst.agent.nodes import build_nodes
from fin_doc_analyst.agent.state import AgentState


def build_graph(store: Any, llm: Any) -> Any:
    nodes = build_nodes(store, llm)
    graph = StateGraph(AgentState)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        _after_route,
        {"retrieve": "retrieve", "refuse": "refuse"},
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _after_grade,
        {
            "compute": "compute",
            "analyze_risk": "analyze_risk",
            "answer": "answer",
            "refuse": "refuse",
        },
    )
    graph.add_conditional_edges(
        "compute",
        _after_compute,
        {"answer": "answer", "refuse": "refuse"},
    )
    graph.add_edge("answer", "ground")
    graph.add_edge("analyze_risk", "ground")
    graph.add_conditional_edges(
        "ground",
        _after_ground,
        {"refuse": "refuse", "end": END},
    )
    graph.add_edge("refuse", END)
    return graph.compile()


def _after_route(state: AgentState) -> str:
    if state.get("route") == "refuse":
        return "refuse"
    return "retrieve"


def _after_grade(state: AgentState) -> str:
    # Compute and risk scan the in-scope corpus themselves. An empty graded
    # set must not block those paths: the question words often are not in the clauses.
    if state.get("route") == "compute":
        return "compute"
    if state.get("route") == "risk":
        return "analyze_risk"
    if not state.get("graded"):
        return "refuse"
    return "answer"


def _after_compute(state: AgentState) -> str:
    if state.get("refused") or not state.get("tool_results"):
        return "refuse"
    return "answer"


def _after_ground(state: AgentState) -> str:
    if state.get("grounded"):
        return "end"
    return "refuse"
