"""
User Agent — LangGraph Definitions
=====================================
Provides two compiled graphs:

  build_user_init_graph()      — 5-node initialization pipeline (prior-based, no LLM)
  build_user_inference_graph() — 4-node per-timestep inference (LLM-based)

Initialization graph (user_init.py nodes):
  load_priors → assign_communities → generate_plans → find_co_mobility → save_output

Inference graph (nodes defined in this file):
  check_plan → [STAY: apply_decision]
             → [MOVE: get_candidates → llm_decide → apply_decision]
  → END
"""

import sys
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.state import UserInitState, UserInferenceState
from model.user_init import (
    node_load_priors,
    node_assign_communities,
    node_generate_plans,
    node_find_co_mobility,
    node_save_output,
)


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION GRAPH  (unchanged logic from original user_graph.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_user_init_graph():
    """
    Build and compile the user initialization StateGraph.
    Runs five sequential nodes to produce 24h plan profiles for all test users.
    """
    graph = StateGraph(UserInitState)

    graph.add_node("load_priors",        node_load_priors)
    graph.add_node("assign_communities", node_assign_communities)
    graph.add_node("generate_plans",     node_generate_plans)
    graph.add_node("find_co_mobility",   node_find_co_mobility)
    graph.add_node("save_output",        node_save_output)

    graph.set_entry_point("load_priors")
    graph.add_edge("load_priors",        "assign_communities")
    graph.add_edge("assign_communities", "generate_plans")
    graph.add_edge("generate_plans",     "find_co_mobility")
    graph.add_edge("find_co_mobility",   "save_output")
    graph.add_edge("save_output",        END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE GRAPH NODES
# Each node receives the full UserInferenceState and returns a partial update.
# ═══════════════════════════════════════════════════════════════════════════════

def node_check_plan(state: UserInferenceState) -> Dict:
    """
    Node 1 — Check daily plan to determine action at current_hour.

    Finds the plan segment covering current_hour and sets:
      action       = "STAY" or "MOVE_AB"
      plan_segment = the matching segment dict
      move_purpose = target POI (only for MOVE_AB)
      move_dist    = distance label (only for MOVE_AB)

    For STAY: also sets next_location and next_hour immediately so
    apply_decision can be skipped via conditional edge.
    """
    hour    = state["current_hour"]
    plan    = state["user_profile"]["plan"]
    user_id = state["user_profile"]["user_id"][:8]

    # Find the segment that covers this hour
    segment = None
    for seg in plan:
        if seg["start_hour"] <= hour < seg["end_hour"]:
            segment = seg
            break

    if segment is None:
        # Past end of plan — stay put
        print(f"  [check_plan] user {user_id}: hour {hour} past plan end → STAY")
        return {
            "action":       "STAY",
            "plan_segment": {},
            "next_location": state["current_location"],
            "next_hour":     hour + 1,
        }

    action = segment["motif_type"]
    print(f"  [check_plan] user {user_id}: hour {hour:02d} → {action}", end="")

    result: Dict[str, Any] = {"action": action, "plan_segment": segment}

    if action == "STAY":
        print(f"  (stay until h{segment['end_hour']:02d}, poi={segment['poi_type']})")
        result["next_location"] = state["current_location"]
        result["next_hour"]     = segment["end_hour"]
        result["move_purpose"]  = ""
        result["move_dist"]     = ""
    else:  # MOVE_AB
        purpose = segment.get("to_poi", "Unknown")
        dist    = segment.get("dist_label", "1-2km")
        print(f"  (move to {purpose}, ~{dist})")
        result["move_purpose"] = purpose
        result["move_dist"]    = dist

    return result


def node_get_candidates(state: UserInferenceState) -> Dict:
    """
    Node 2 (MOVE branch only) — Spatial gravity search for candidate destinations.

    Calls spatial_gravity_search from util.common and enriches each candidate
    with its flow_out value from the flow_from lookup table.
    """
    from util.common import spatial_gravity_search

    current  = state["current_location"]
    user_id  = state["user_profile"]["user_id"][:8]

    candidates = spatial_gravity_search(
        current_loc=current,
        dist_label=state["move_dist"],
        purpose=state["move_purpose"],
        coord_map=state["coord_map"],
        poi_map=state["poi_map"],
        pop_map=state["pop_map"],
        cfg=state["cfg"],
    )

    # Enrich candidates with flow_out from origin perspective of destination
    flow_from = state.get("flow_from", {})
    for c in candidates:
        c["flow_out"] = round(sum(flow_from.get(str(c["loc_id"]), {}).values()), 2)

    print(f"  [get_candidates] user {user_id}: {len(candidates)} candidates "
          f"(purpose={state['move_purpose']}, dist={state['move_dist']})")
    for c in candidates:
        print(f"    loc {c['loc_id']:5d}  {c['poi_type']:32s}  "
              f"dist={c['dist_km']:.2f}km  gravity={c['gravity_score']:.3f}  "
              f"flow_out={c['flow_out']:.1f}")

    return {"candidates": candidates}


def node_llm_decide(state: UserInferenceState) -> Dict:
    """
    Node 3 (MOVE branch only) — Call LLM to choose next location.

    Reads LLM config from state['cfg']['llm'], builds the move-decision prompt,
    calls ChatOpenAI, and stores the raw response string.
    """
    from langchain_openai import ChatOpenAI
    from util.prompt import build_move_decision_prompt

    cfg     = state["cfg"]
    llm_cfg = cfg.get("llm", {})
    user_id = state["user_profile"]["user_id"][:8]

    llm = ChatOpenAI(
        model       = llm_cfg.get("model",       "gpt-4o-mini"),
        api_key     = llm_cfg.get("api_key"),
        base_url    = llm_cfg.get("base_url",    "https://api.openai-proxy.org/v1"),
        temperature = llm_cfg.get("temperature", 0.2),
        max_tokens  = llm_cfg.get("max_tokens",  150),
    )

    # Current location POI for context
    current_poi = state["poi_map"].get(state["current_location"], "Unknown")

    prompt = build_move_decision_prompt(
        city              = state["city"],
        user_label        = cfg.get("user_label", "Users"),
        community_profile = state["user_profile"].get("community_poi_profile", "Unknown"),
        from_poi          = current_poi,
        to_poi            = state["move_purpose"],
        dist_label        = state["move_dist"],
        candidates        = state.get("candidates", []),
        current_hour      = state["current_hour"],
    )

    print(f"  [llm_decide] user {user_id}: calling {llm_cfg.get('model', 'gpt-4o-mini')} ...")
    response = llm.invoke(prompt)
    raw      = response.content
    print(f"  [llm_decide] response: {raw}")

    return {"llm_response": raw}


def node_apply_decision(state: UserInferenceState) -> Dict:
    """
    Node 4 — Apply the decision and advance the simulation clock.

    For STAY  : next_location and next_hour were already set by check_plan.
    For MOVE_AB: parse LLM response via util.parser, validate against candidates.
    """
    from util.parser import parse_location_decision

    user_id = state["user_profile"]["user_id"][:8]

    if state.get("action") == "STAY":
        # Already resolved in check_plan; nothing more to do
        print(f"  [apply_decision] user {user_id}: STAY at "
              f"loc {state['next_location']} until h{state['next_hour']:02d}")
        return {}

    # --- Parse LLM output ---
    raw        = state.get("llm_response", "")
    candidates = state.get("candidates", [])
    decision   = parse_location_decision(raw, candidates)

    next_loc   = decision.get("next_location_id", state["current_location"])
    reason     = decision.get("reason", "")
    method     = decision.get("parse_method", "unknown")

    # Validate: if loc_id not in candidates, fall back to top candidate
    valid_ids = {c["loc_id"] for c in candidates}
    if next_loc not in valid_ids and candidates:
        print(f"  [apply_decision] user {user_id}: loc {next_loc} not in candidates "
              f"— falling back to top candidate")
        next_loc = candidates[0]["loc_id"]
        reason   = "corrected to top gravity candidate"

    print(f"  [apply_decision] user {user_id}: MOVE → loc {next_loc}  "
          f"({method}) reason: {reason}")

    return {
        "decision":      decision,
        "next_location": next_loc,
        "next_hour":     state["current_hour"] + 1,
    }


# ── Conditional router after check_plan ──────────────────────────────────────

def _route_after_check_plan(state: UserInferenceState) -> str:
    """Route to get_candidates (MOVE) or apply_decision (STAY)."""
    return "get_candidates" if state.get("action") == "MOVE_AB" else "apply_decision"


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_user_inference_graph():
    """
    Build and compile the per-timestep user inference StateGraph.

    Graph structure:
        check_plan
            ├─ (MOVE_AB) → get_candidates → llm_decide → apply_decision → END
            └─ (STAY)    → apply_decision → END
    """
    graph = StateGraph(UserInferenceState)

    graph.add_node("check_plan",     node_check_plan)
    graph.add_node("get_candidates", node_get_candidates)
    graph.add_node("llm_decide",     node_llm_decide)
    graph.add_node("apply_decision", node_apply_decision)

    graph.set_entry_point("check_plan")

    # Conditional branch after check_plan
    graph.add_conditional_edges(
        "check_plan",
        _route_after_check_plan,
        {
            "get_candidates": "get_candidates",
            "apply_decision": "apply_decision",
        },
    )

    graph.add_edge("get_candidates", "llm_decide")
    graph.add_edge("llm_decide",     "apply_decision")
    graph.add_edge("apply_decision", END)

    return graph.compile()
