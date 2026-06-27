"""
State Definitions
==================
Two TypedDicts for the two pipeline stages:

  UserInitState      — initialization pipeline (5 nodes, prior-based)
  UserInferenceState — per-timestep inference (LLM-based mobility decision)

All values use plain Python types so state is directly JSON-serializable.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class UserInitState(TypedDict, total=False):
    # ── Pipeline settings ────────────────────────────────────────────────
    city: str                          # 'shanghai' | 'shenzhen'
    cfg: Dict[str, Any]                # full YAML config
    seed: int                          # RNG seed for plan generation
    co_mobility_threshold: float       # Jaccard similarity threshold

    # ── Community priors (from community_fixed.json) ─────────────────────
    region_to_comm: Dict[str, int]     # str(region_id) → community_id
    comm_to_locs:   Dict[str, List[int]]  # str(comm_id) → [region_id,...]
    n_communities:  int
    poi_map:        Dict[int, str]     # region_id → dominant POI category
    coord_map:      Dict[int, List[float]]  # region_id → [lon, lat]

    # ── Motif priors (from motifs.json) ──────────────────────────────────
    # Global 2-state Markov chain: P(next_motif | current_motif)
    global_transition: Dict[str, Dict[str, float]]
    # Per-community stay duration distribution: comm_id(str) → {label: prob}
    comm_stay_probs: Dict[str, Dict[str, float]]
    # Per-community MOVE ratio: comm_id(str) → P(MOVE)
    comm_move_ratio: Dict[str, float]

    # ── Flow priors (from population_flow.json) ───────────────────────────
    # Aggregate flow across all hours: str(from_rid) → {str(to_rid): total_flow}
    flow_from: Dict[str, Dict[str, float]]
    # Per-community mean population: str(comm_id) → {str(region_id): mean_pop}
    comm_region_pop: Dict[str, Dict[str, float]]
    # Flat population map for gravity search: int(loc_id) → mean population
    pop_map: Dict[int, float]

    # ── Train user community assignments ─────────────────────────────────
    train_user_start_loc: Dict[str, int]   # user_id → region_id at hour 0
    train_user_community: Dict[str, int]   # user_id → community_id

    # ── P(community | start_location) from train data ────────────────────
    # str(loc_id) → {str(comm_id): probability}
    loc_comm_probs: Dict[str, Dict[str, float]]

    # ── Test users input ─────────────────────────────────────────────────
    # [{"user_id": ..., "start_location": int, "date": int}]
    test_users: List[Dict[str, Any]]

    # ── Generated output (populated by generate_plans + find_co_mobility) ─
    user_profiles: List[Dict[str, Any]]

    # ── Error signal ─────────────────────────────────────────────────────
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Per-timestep inference state — one user, one hour
# ─────────────────────────────────────────────────────────────────────────────

class UserInferenceState(TypedDict, total=False):
    # ── Static context (set once per inference call) ──────────────────────
    city: str
    cfg:  Dict[str, Any]
    user_profile: Dict[str, Any]     # single user's init-phase profile

    # ── Reference data (loaded from priors) ───────────────────────────────
    coord_map:  Dict[int, List[float]]        # loc_id → [lon, lat]
    poi_map:    Dict[int, str]                # loc_id → POI type
    pop_map:    Dict[int, float]              # loc_id → mean population
    flow_from:  Dict[str, Dict[str, float]]   # str(rid) → {str(rid): flow}

    # ── Per-step runtime ──────────────────────────────────────────────────
    current_location: int    # current location at start of this timestep
    current_hour:     int    # 0–23

    # ── Intermediate (filled by nodes) ────────────────────────────────────
    plan_segment: Dict[str, Any]  # plan segment covering current_hour
    action:       str             # "STAY" or "MOVE_AB"
    move_purpose: str             # target POI type (from MOVE segment)
    move_dist:    str             # dist_label from MOVE segment (e.g. "1-2km")
    candidates:   List[Dict]      # spatial gravity top-k with flow_out added
    llm_response: str             # raw LLM output string
    decision:     Dict[str, Any]  # parsed {"next_location_id": int, "reason": str}

    # ── Output ────────────────────────────────────────────────────────────
    next_location: int   # decided next location
    next_hour:     int   # hour after this action completes
    error:         Optional[str]
