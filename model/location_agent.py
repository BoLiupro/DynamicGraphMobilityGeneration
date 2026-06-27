"""
Location Agent — Node Functions and LangGraph Definition
=========================================================
Location agents reason from the "pull" side: given a location's POI profile,
hourly flow patterns, and the nearby users' plans, which users should move here?

This complements the user agent (which reasons from the user's own intent) and
produces an independent mobility graph. Both are later reconciled by the
reflection agent.

Pipeline:
  load_priors → select_active → process_locations → compile_mobility_graph → END
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from model.state import LocationAgentState


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS  (no LLM, pure data transformations)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_hourly_flow_in(
    pop_flow: List[Dict],
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, Dict[int, float]]]]:
    """
    Build per-hour flow-in structures from population_flow.json.

    Returns:
        hourly_flow_in:      {hour → {int(dst_loc): total_flow_in}}
        hourly_flow_sources: {hour → {int(dst_loc): {int(src_loc): flow}}}

    Note: population_flow.json uses string keys for loc_ids; converted to int here.
    """
    hourly_flow_in:      Dict[int, Dict[int, float]]                 = {}
    hourly_flow_sources: Dict[int, Dict[int, Dict[int, float]]]       = {}

    for slot in pop_flow:
        h   = int(slot["hour"])
        fin: Dict[int, float]                 = defaultdict(float)
        src: Dict[int, Dict[int, float]]      = defaultdict(dict)

        for edge_key, flow in slot["edge_flow_mean"].items():
            fr_str, to_str = edge_key.split(",")
            to_id, fr_id   = int(to_str), int(fr_str)
            fin[to_id]         += flow
            src[to_id][fr_id]   = flow

        hourly_flow_in[h]      = dict(fin)
        hourly_flow_sources[h] = {k: dict(v) for k, v in src.items()}

    return hourly_flow_in, hourly_flow_sources


def _get_plan_segment_at_hour(plan: List[Dict], hour: int) -> Optional[Dict]:
    """Return the plan segment whose [start_hour, end_hour) covers `hour`."""
    for seg in plan:
        if seg["start_hour"] <= hour < seg["end_hour"]:
            return seg
    return None


def _build_profile_lookup(profiles: List[Dict]) -> Dict[str, Dict]:
    """Return {user_id: profile} for O(1) lookup."""
    return {p["user_id"]: p for p in profiles}


def _build_location_context(
    loc_id: int,
    hour:   int,
    state:  "LocationAgentState",
) -> Dict:
    """
    Assemble the full context dict for a single location at `hour`.

    Context sections:
      poi_types          — top-k POI categories (from poi_multi_map)
      expected_pop       — mean population from hourly prior
      sim_flow_in        — count of users currently at source locations (simulation-state)
      hist_flow_in       — historical mean daily flow-in (from population_flow.json)
      flow_in_sources    — top-k source locations, sorted by flow, with pct and user count
      users_at_sources   — users currently at source locations, with their plan intent
      co_mobility_groups — pairs of co-mobile users both in the neighborhood
    """
    cfg            = state["cfg"]
    loc_cfg        = cfg.get("location_agent", {})
    max_sources    = loc_cfg.get("max_flow_sources",    5)
    max_users      = loc_cfg.get("max_neighbor_users", 20)

    poi_map        = state.get("poi_map",        {})
    poi_multi_map  = state.get("poi_multi_map",  {})
    hourly_pop     = state.get("hourly_pop",     {})
    hourly_src     = state.get("hourly_flow_sources", {})
    user_positions = state.get("user_positions", {})
    profiles       = state.get("all_user_profiles", [])

    # ── Multi-POI types ────────────────────────────────────────────────────
    poi_types    = poi_multi_map.get(loc_id) or [poi_map.get(loc_id, "Unknown")]
    expected_pop = hourly_pop.get(hour, {}).get(loc_id, 0.0)

    # ── Flow sources at this hour (historical) ─────────────────────────────
    src_flows         = hourly_src.get(hour, {}).get(loc_id, {})
    hist_flow_in      = sum(src_flows.values())
    total_flow_denom  = hist_flow_in or 1.0

    # Count users currently at each source location
    source_user_counts: Dict[int, int] = {}
    for src_id in src_flows:
        source_user_counts[src_id] = sum(
            1 for pos in user_positions.values() if pos == src_id
        )
    sim_flow_in = sum(source_user_counts.values())

    top_sources = sorted(src_flows.items(), key=lambda x: -x[1])[:max_sources]
    flow_in_sources = [
        {
            "from_loc":    src_id,
            "from_poi":    poi_map.get(src_id, "Unknown"),
            "hist_flow":   round(flow, 3),
            "pct":         round(flow / total_flow_denom * 100, 1),
            "users_now":   source_user_counts.get(src_id, 0),
        }
        for src_id, flow in top_sources
    ]

    # ── Users at source locations ──────────────────────────────────────────
    # These are users positioned at locations that historically send flow to loc_id
    neighbor_locs  = set(src_flows.keys())   # int loc_ids
    profile_by_id  = _build_profile_lookup(profiles)

    users_at_sources = []
    for uid, pos in user_positions.items():
        if pos not in neighbor_locs:
            continue
        profile = profile_by_id.get(uid)
        if not profile:
            continue
        seg = _get_plan_segment_at_hour(profile.get("plan", []), hour)
        plan_to_poi = seg.get("to_poi", "") if seg and seg["motif_type"] == "MOVE_AB" else ""
        users_at_sources.append({
            "user_id":       uid,
            "current_loc":   pos,
            "current_poi":   poi_map.get(pos, "Unknown"),
            "plan_action":   seg["motif_type"] if seg else "STAY",
            "plan_to_poi":   plan_to_poi,
            # True when user's plan destination matches any of this location's POI types
            "poi_match":     plan_to_poi in poi_types,
            "n_co_mobile":   len(profile.get("co_mobility_users", [])),
        })
        if len(users_at_sources) >= max_users:
            break

    # ── Co-mobility groups in the neighborhood ─────────────────────────────
    co_groups  = []
    seen_pairs = set()
    for u in users_at_sources:
        uid_a     = u["user_id"]
        profile_a = profile_by_id.get(uid_a, {})
        for co_info in profile_a.get("co_mobility_users", []):
            uid_b = co_info["user_id"]
            pair  = tuple(sorted([uid_a, uid_b]))   # canonical order
            if pair in seen_pairs:
                continue
            profile_b = profile_by_id.get(uid_b)
            if not profile_b:
                continue
            pos_b = user_positions.get(uid_b)
            if pos_b not in neighbor_locs:
                continue
            seen_pairs.add(pair)

            seg_a     = _get_plan_segment_at_hour(profile_a.get("plan", []), hour)
            seg_b     = _get_plan_segment_at_hour(profile_b.get("plan", []), hour)
            both_move = bool(
                seg_a and seg_a["motif_type"] == "MOVE_AB" and
                seg_b and seg_b["motif_type"] == "MOVE_AB"
            )
            co_groups.append({
                "user_a":     uid_a,
                "user_b":     uid_b,
                "similarity": co_info["similarity"],
                "loc_a":      u["current_loc"],
                "loc_b":      pos_b,
                "both_move":  both_move,
            })

    return {
        "loc_id":             loc_id,
        "poi_types":          poi_types,
        "expected_pop":       round(expected_pop, 2),
        "sim_flow_in":        sim_flow_in,
        "hist_flow_in":       round(hist_flow_in, 3),
        "total_flow_in":      hist_flow_in,
        "flow_in_sources":    flow_in_sources,
        "neighbor_users":     users_at_sources,
        "co_mobility_groups": co_groups,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def node_load_location_priors(state: LocationAgentState) -> Dict:
    """
    Node 1 — Load hourly population and flow priors from population_flow.json.

    Builds:
      flow_to             : aggregated dst→{src: total_flow}
      hourly_flow_in      : hour→{loc_id: total_flow_in}
      hourly_flow_sources : hour→{loc_id: {src_id: flow}}
      hourly_pop          : hour→{loc_id: population}
    """
    print("\n[LocNode 1] Loading location priors ...")
    cfg      = state["cfg"]
    data_dir = Path(cfg["paths"]["processed_dir"])
    pat_dir  = data_dir / "dynamic_graph" / "extracted_pattern"

    with open(pat_dir / "population_flow.json") as f:
        pop_flow = json.load(f)

    hourly_flow_in, hourly_sources = _build_hourly_flow_in(pop_flow)

    # hourly_pop: hour → {int(loc_id): population}
    hourly_pop: Dict[int, Dict[int, float]] = {}
    for slot in pop_flow:
        h = int(slot["hour"])
        hourly_pop[h] = {int(k): float(v) for k, v in slot["population_mean"].items()}

    active_hours = sum(1 for h in hourly_flow_in if hourly_flow_in[h])
    print(f"  Hours with flow : {active_hours} / {len(hourly_flow_in)}")
    print(f"  Hourly pop locs : {len(hourly_pop.get(8, {}))}")

    return {
        "hourly_flow_in":      hourly_flow_in,
        "hourly_flow_sources": hourly_sources,
        "hourly_pop":          hourly_pop,
    }


def node_select_active_locations(state: LocationAgentState) -> Dict:
    """
    Node 2 — Select top-m locations by current-simulation flow count at current_hour.

    Primary sort : sim_flow_in  = count of users currently positioned at source locations
                   (reflects actual simulation state rather than historical average)
    Secondary sort: hist_flow_in = historical mean daily flow from population_flow.json
                   (tiebreaker when sim counts are equal)

    Reads cfg['location_agent']['top_m'] (default 10).
    """
    h              = state["current_hour"]
    top_m          = state["cfg"].get("location_agent", {}).get("top_m", 10)
    user_positions = state.get("user_positions", {})
    hourly_src     = state.get("hourly_flow_sources", {})
    hourly_flow_in = state.get("hourly_flow_in", {})
    poi_map        = state.get("poi_map", {})

    src_at_h     = hourly_src.get(h, {})
    hist_flow_h  = hourly_flow_in.get(h, {})

    # Compute simulation-based flow_in for every location that has sources at hour h
    sim_flow: Dict[int, int] = {}
    for loc_id, src_flows in src_at_h.items():
        sim_flow[loc_id] = sum(
            1 for pos in user_positions.values() if pos in src_flows
        )

    # Rank: primary sim count, secondary historical flow
    all_locs = set(sim_flow.keys()) | set(hist_flow_h.keys())
    ranked   = sorted(
        all_locs,
        key=lambda loc: (-sim_flow.get(loc, 0), -hist_flow_h.get(loc, 0.0)),
    )[:top_m]

    print(f"\n[LocNode 2] Top-{top_m} active locations at hour {h:02d}:00")
    print(f"  {'loc':>5}  {'POI':32s}  {'sim_users':>9}  {'hist_flow':>9}")
    for loc_id in ranked:
        print(f"  {loc_id:5d}  {poi_map.get(loc_id, 'Unknown'):32s}  "
              f"{sim_flow.get(loc_id, 0):9d}  "
              f"{hist_flow_h.get(loc_id, 0.0):9.3f}")

    return {"active_locations": ranked}


def node_process_locations(state: LocationAgentState) -> Dict:
    """
    Node 3 — For each active location, build context and call LLM.

    For each location:
      1. Build context via _build_location_context()
      2. Skip LLM if no neighbor users
      3. Call ChatOpenAI with build_location_decision_prompt()
      4. Parse response via parse_user_selection()
    """
    from langchain_openai import ChatOpenAI
    from util.prompt import build_location_decision_prompt
    from util.parser import parse_user_selection

    cfg         = state["cfg"]
    llm_cfg     = cfg.get("llm", {})
    h           = state["current_hour"]
    active_locs = state.get("active_locations", [])
    poi_map     = state.get("poi_map", {})

    llm = ChatOpenAI(
        model       = llm_cfg.get("model",       "gpt-4o-mini"),
        api_key     = llm_cfg.get("api_key"),
        base_url    = llm_cfg.get("base_url",    "https://api.openai-proxy.org/v1"),
        temperature = llm_cfg.get("temperature", 0.2),
        max_tokens  = llm_cfg.get("max_tokens",  200),
    )

    print(f"\n[LocNode 3] Processing {len(active_locs)} locations at hour {h:02d}:00 ...")

    location_contexts:  Dict[int, Dict]       = {}
    location_decisions: Dict[int, List[str]]  = {}

    for i, loc_id in enumerate(active_locs):
        poi = poi_map.get(loc_id, "Unknown")
        print(f"\n  [{i+1}/{len(active_locs)}] loc {loc_id} ({poi})")

        # Build context
        ctx = _build_location_context(loc_id, h, state)
        location_contexts[loc_id] = ctx

        poi_str = " / ".join(ctx["poi_types"])
        print(f"    poi_types     : {poi_str}")
        print(f"    pop           : {ctx['expected_pop']:.1f}  "
              f"hist_flow_in={ctx['hist_flow_in']:.3f}  "
              f"sim_flow_in={ctx['sim_flow_in']} (users at source locs)")
        print(f"    source_locs   : {len(ctx['flow_in_sources'])}  "
              f"users_at_sources={len(ctx['neighbor_users'])} (prev-step positions)  "
              f"co_groups={len(ctx['co_mobility_groups'])}")

        # Skip LLM if no users are nearby to decide about
        if not ctx["neighbor_users"]:
            print(f"    (no neighbor users — skipping LLM)")
            location_decisions[loc_id] = []
            continue

        # Build prompt and call LLM
        prompt = build_location_decision_prompt(
            city            = state["city"],
            user_label      = cfg.get("user_label", "Users"),
            loc_id          = loc_id,
            poi_types       = ctx["poi_types"],
            current_hour    = h,
            expected_pop    = ctx["expected_pop"],
            sim_flow_in     = ctx["sim_flow_in"],
            flow_in_sources = ctx["flow_in_sources"],
            neighbor_users  = ctx["neighbor_users"],
            co_groups       = ctx["co_mobility_groups"],
        )

        print(f"    Calling {llm_cfg.get('model', 'gpt-4o-mini')} ...")
        response = llm.invoke(prompt)
        raw      = response.content
        print(f"    LLM: {raw}")

        # Parse and validate against known neighbor user_ids
        valid_ids = {u["user_id"] for u in ctx["neighbor_users"]}
        decision  = parse_user_selection(raw, valid_ids)
        location_decisions[loc_id] = decision["users_to_attract"]
        print(f"    Attracted ({decision['parse_method']}): "
              f"{len(decision['users_to_attract'])} users")

    return {
        "location_contexts":  location_contexts,
        "location_decisions": location_decisions,
    }


def node_compile_mobility_graph(state: LocationAgentState) -> Dict:
    """
    Node 4 — Aggregate all location decisions into the mobility graph.

    Conflict resolution (a user attracted by multiple locations):
      Priority 1 — Co-mobility convergence
          Prefer the location where the most co-mobile peers have already
          been assigned.  Users with more co-mobile peers are processed first
          so their choices create a signal for later users.
      Priority 2 — Flow-in attractiveness
          Among locations with equal co-peer count, prefer the one with the
          higher total_flow_in (historical pull strength).
      Priority 3 — Random
          Remaining ties are broken uniformly at random (seed from cfg or 42).

    Users not attracted by any location → STAY at current position.
    Each output record includes a "resolve_method" field for diagnostics.
    """
    rng = random.Random(state["cfg"].get("seed", 42))

    h                  = state["current_hour"]
    location_decisions = state.get("location_decisions", {})
    location_contexts  = state.get("location_contexts",  {})
    user_positions     = state.get("user_positions",     {})
    all_profiles       = state.get("all_user_profiles",  [])
    poi_map            = state.get("poi_map",            {})

    print(f"\n[LocNode 4] Compiling location-agent mobility graph ...")

    # ── Step 1: Build user → candidate list  {uid: [(loc_id, flow_in), ...]} ──
    user_to_candidates: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for loc_id, attracted in location_decisions.items():
        flow_in = location_contexts.get(loc_id, {}).get("total_flow_in", 0.0)
        for uid in attracted:
            user_to_candidates[uid].append((loc_id, flow_in))

    n_total      = len(user_to_candidates)
    n_conflicted = sum(1 for v in user_to_candidates.values() if len(v) > 1)
    print(f"  Users with attraction : {n_total}  (conflicts: {n_conflicted})")

    # ── Step 2: Profile lookup for co-mobility info ──────────────────────────
    profile_by_id = {p["user_id"]: p for p in all_profiles}

    # ── Step 3: Resolve assignments ──────────────────────────────────────────
    resolved:       Dict[str, int] = {}   # uid → assigned loc_id
    resolve_method: Dict[str, str] = {}   # uid → method string

    # Non-conflicted: assign immediately (no choice to make)
    for uid, cands in user_to_candidates.items():
        if len(cands) == 1:
            resolved[uid]       = cands[0][0]
            resolve_method[uid] = "unique"

    # Conflicted: process in descending order of co-mobile peer count so that
    # users with many social ties create a convergence signal for others
    conflicted_sorted = sorted(
        ((uid, cands) for uid, cands in user_to_candidates.items() if len(cands) > 1),
        key=lambda kv: -len(profile_by_id.get(kv[0], {}).get("co_mobility_users", [])),
    )

    n_by_co = n_by_flow = n_by_rand = 0
    for uid, candidates in conflicted_sorted:
        profile  = profile_by_id.get(uid, {})
        co_peers = {c["user_id"] for c in profile.get("co_mobility_users", [])}

        # Score each candidate as (co_peers_already_there, flow_in)
        scored: List[Tuple[int, float, int]] = []   # (co_count, flow_in, loc_id)
        for loc_id, flow_in in candidates:
            co_count = sum(1 for peer in co_peers if resolved.get(peer) == loc_id)
            scored.append((co_count, flow_in, loc_id))

        best_co   = max(s[0] for s in scored)
        best_flow = max(s[1] for s in scored if s[0] == best_co)
        best_locs = [s[2] for s in scored if s[0] == best_co and s[1] == best_flow]

        chosen = rng.choice(best_locs)
        resolved[uid] = chosen

        if best_co > 0:
            resolve_method[uid] = "co_mobility"
            n_by_co += 1
        elif len(best_locs) == 1:
            resolve_method[uid] = "flow_in"
            n_by_flow += 1
        else:
            resolve_method[uid] = "random"
            n_by_rand += 1

    if n_conflicted > 0:
        print(f"  Conflict resolution:")
        print(f"    co_mobility : {n_by_co}")
        print(f"    flow_in     : {n_by_flow}")
        print(f"    random      : {n_by_rand}")

    # ── Step 4: Build one record per user ─────────────────────────────────────
    mobility_graph: List[Dict] = []
    n_move = n_stay = 0

    for profile in all_profiles:
        uid      = profile["user_id"]
        from_loc = user_positions.get(uid, -1)

        if uid in resolved:
            to_loc = resolved[uid]
            action = "MOVE" if to_loc != from_loc else "STAY"
        else:
            to_loc = from_loc
            action = "STAY"

        if action == "MOVE":
            n_move += 1
        else:
            n_stay += 1

        mobility_graph.append({
            "user_id":        uid,
            "hour":           h,
            "from_loc":       from_loc,
            "to_loc":         to_loc,
            "action":         action,
            "from_poi":       poi_map.get(from_loc, "Unknown"),
            "to_poi":         poi_map.get(to_loc,   "Unknown"),
            "resolve_method": resolve_method.get(uid, "no_attraction"),
        })

    print(f"  Total users : {len(mobility_graph)}")
    print(f"  MOVE        : {n_move}")
    print(f"  STAY        : {n_stay}")

    return {"mobility_graph": mobility_graph}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_location_agent_graph():
    """
    Build and compile the location agent StateGraph.

    Graph:
        load_priors → select_active → process_locations → compile_graph → END
    """
    graph = StateGraph(LocationAgentState)

    graph.add_node("load_priors",       node_load_location_priors)
    graph.add_node("select_active",     node_select_active_locations)
    graph.add_node("process_locations", node_process_locations)
    graph.add_node("compile_graph",     node_compile_mobility_graph)

    graph.set_entry_point("load_priors")
    graph.add_edge("load_priors",       "select_active")
    graph.add_edge("select_active",     "process_locations")
    graph.add_edge("process_locations", "compile_graph")
    graph.add_edge("compile_graph",     END)

    return graph.compile()
