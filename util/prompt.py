#!/usr/bin/env python3
"""
Prompt templates for LLM-based mobility decisions.

Each function returns a plain string prompt ready to pass to a ChatLLM.
"""

from typing import List, Dict


def build_move_decision_prompt(
    city: str,
    user_label: str,
    community_profile: str,
    from_poi: str,
    to_poi: str,
    dist_label: str,
    candidates: List[Dict],
    current_hour: int,
) -> str:
    """
    Prompt for deciding the next location when a user executes a MOVE_AB.

    Args:
        city              : city name (e.g. "shanghai")
        user_label        : dataset-specific agent type from config
                            (e.g. "Vehicles" for car GPS, "Users" for mobile signaling)
        community_profile : dominant POI type of the user's home community
        from_poi          : POI type of the current (origin) location
        to_poi            : intended destination POI type (from daily plan)
        dist_label        : intended travel distance bin (e.g. "1-2km")
        candidates        : list of dicts from spatial_gravity_search, each with
                            {loc_id, poi_type, dist_km, gravity_score, flow_out}
        current_hour      : current simulation hour (0-23)

    Returns:
        Formatted prompt string.
    """
    # Build candidate table
    if candidates:
        rows = "\n".join(
            f"  {i+1}. loc_id={c['loc_id']}, poi={c['poi_type']}, "
            f"dist={c['dist_km']}km, gravity={c['gravity_score']}, "
            f"flow_out={c.get('flow_out', 0):.1f}"
            for i, c in enumerate(candidates)
        )
    else:
        rows = "  (no candidates available)"

    prompt = f"""You are simulating a mobility agent ({user_label}) in {city} at hour {current_hour:02d}:00.

Agent background: typically active in {community_profile} areas.
Current action: departing from a {from_poi} area, heading to a {to_poi} area (~{dist_label}).

Candidate destinations ranked by spatial attractiveness:
{rows}

Scoring criteria:
- gravity_score : attractiveness based on population density and proximity (higher = better)
- flow_out      : historical outbound traffic volume at this location (0 = unknown)

Choose the best destination, in order of priority:
1. POI type matches the intended purpose ({to_poi})
2. Highest gravity_score
3. Highest flow_out as tiebreaker

Reply with JSON only — no extra text:
{{"next_location_id": <integer loc_id from the list above>, "reason": "<max 25 words>"}}"""

    return prompt


def build_location_decision_prompt(
    city: str,
    user_label: str,
    loc_id: int,
    poi_types: List[str],
    current_hour: int,
    expected_pop: float,
    sim_flow_in: int,
    flow_in_sources: List[Dict],
    neighbor_users: List[Dict],
    co_groups: List[Dict],
) -> str:
    """
    Prompt for a location agent deciding which nearby users to attract.

    Args:
        city            : city name
        user_label      : "Vehicles" or "Users" (from config)
        loc_id          : this location's id
        poi_types       : this location's top-k POI categories (ranked by count)
        current_hour    : simulation hour (0-23)
        expected_pop    : mean expected population from flow prior
        sim_flow_in     : count of users currently at source locations (simulation state)
        flow_in_sources : list of {from_loc, from_poi, hist_flow, pct, users_now}
        neighbor_users  : list of {user_id, current_loc, current_poi,
                                   plan_action, plan_to_poi, poi_match, n_co_mobile}
        co_groups       : list of {user_a, user_b, similarity, loc_a, loc_b, both_move}

    Returns:
        Formatted prompt string.
    """
    MAX_USERS  = 15
    MAX_GROUPS = 5

    poi_str     = " / ".join(poi_types)
    primary_poi = poi_types[0] if poi_types else "Unknown"

    # ── Flow sources table ────────────────────────────────────────────────
    if flow_in_sources:
        src_rows = "\n".join(
            f"  {s['from_poi']:32s}  (loc {s['from_loc']})  "
            f"hist_flow={s['hist_flow']:.3f}  ({s['pct']}%)  "
            f"users_now={s['users_now']}"
            for s in flow_in_sources
        )
    else:
        src_rows = "  (no incoming flow at this hour)"

    # ── Neighbor users table ──────────────────────────────────────────────
    display_users = neighbor_users[:MAX_USERS]
    if display_users:
        user_rows = "\n".join(
            f"  {u['user_id'][:12]}  at {u['current_poi']:28s}  "
            f"plan={u['plan_action']}"
            + (f"→{u['plan_to_poi']}" if u.get("plan_to_poi") else "")
            + ("  [POI MATCH]" if u.get("poi_match") else "")
            + f"  co_links={u['n_co_mobile']}"
            for u in display_users
        )
        if len(neighbor_users) > MAX_USERS:
            user_rows += f"\n  ... and {len(neighbor_users) - MAX_USERS} more"
    else:
        user_rows = "  (none)"

    # ── Co-mobility groups ────────────────────────────────────────────────
    display_groups = co_groups[:MAX_GROUPS]
    if display_groups:
        group_rows = "\n".join(
            f"  {g['user_a'][:12]} + {g['user_b'][:12]}  "
            f"sim={g['similarity']:.2f}  both_moving={g['both_move']}"
            for g in display_groups
        )
        if len(co_groups) > MAX_GROUPS:
            group_rows += f"\n  ... and {len(co_groups) - MAX_GROUPS} more pairs"
    else:
        group_rows = "  (none)"

    # ── Assemble prompt ───────────────────────────────────────────────────
    prompt = f"""You are location {loc_id} ({poi_str}) in {city} at hour {current_hour:02d}:00.
Your role: decide which nearby {user_label} are likely to move to your location this hour.

Location profile:
  POI types     : {poi_str}  (ranked by POI density)
  Expected pop  : {expected_pop:.1f} (historical average)
  Sim flow_in   : {sim_flow_in} {user_label} currently at source locations
  Top flow sources (locations that historically send people here):
{src_rows}

Nearby {user_label} (currently at source locations in the previous timestep):
{user_rows}

Co-mobility groups in the neighborhood (pairs who tend to travel together):
{group_rows}

Decision criteria (in order of priority):
1. {user_label} whose plan_to_poi matches any of your POI types ({poi_str}) — marked [POI MATCH]
2. {user_label} from source locations with highest users_now count
3. Co-mobile pairs where both members are moving (attract both together)

Return a JSON object — no extra text:
{{"users_to_attract": ["<user_id>", ...], "reasoning": "<max 25 words>"}}

Only include user_ids from the nearby {user_label} list above."""

    return prompt
