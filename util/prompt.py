#!/usr/bin/env python3
"""
Prompt templates for LLM-based mobility decisions.

Each function returns a plain string prompt ready to pass to a ChatLLM.
"""

from typing import List, Dict


def build_move_decision_prompt(
    city: str,
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

    prompt = f"""You are simulating a car driver in {city} at hour {current_hour:02d}:00.

Driver background: typically active in {community_profile} areas.
Current action: leaving a {from_poi} area, heading to a {to_poi} area (~{dist_label}).

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
{{"next_location_id": <integer loc_id from the list above>, "reason": "<max 15 words>"}}"""

    return prompt
