"""
User Agent LangGraph Definition
================================
Builds and compiles the StateGraph for user agent initialization.

Graph flow:
  load_priors → assign_communities → generate_plans
              → find_co_mobility → save_output → END
"""

from langgraph.graph import StateGraph, END
from model.state import UserInitState
from model.user_agent import (
    node_load_priors,
    node_assign_communities,
    node_generate_plans,
    node_find_co_mobility,
    node_save_output,
)


def build_user_init_graph():
    """
    Construct and compile the user initialization StateGraph.

    Returns a compiled LangGraph app that accepts an initial UserInitState
    and runs all five nodes in sequence.
    """
    graph = StateGraph(UserInitState)

    # ── Register nodes ────────────────────────────────────────────────
    graph.add_node("load_priors",         node_load_priors)
    graph.add_node("assign_communities",  node_assign_communities)
    graph.add_node("generate_plans",      node_generate_plans)
    graph.add_node("find_co_mobility",    node_find_co_mobility)
    graph.add_node("save_output",         node_save_output)

    # ── Define linear execution order ─────────────────────────────────
    graph.set_entry_point("load_priors")
    graph.add_edge("load_priors",        "assign_communities")
    graph.add_edge("assign_communities", "generate_plans")
    graph.add_edge("generate_plans",     "find_co_mobility")
    graph.add_edge("find_co_mobility",   "save_output")
    graph.add_edge("save_output",        END)

    return graph.compile()
