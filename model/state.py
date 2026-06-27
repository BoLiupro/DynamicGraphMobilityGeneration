"""
User Agent State Definition
============================
Typed state dict passed between every LangGraph node in the
user-agent initialization pipeline.  All values use plain Python
types (dict/list/int/str/float) so the state is directly JSON-serializable.
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
