"""
User Initialization Node Functions
====================================
Each function is a LangGraph node that transforms the UserInitState.
Handles the 5-step initialization pipeline (no LLM, prior-based).

Pipeline order:
  load_priors          — load pattern files from extracted_pattern/
  assign_communities   — derive P(comm|start_loc) from train; assign to test users
  generate_plans       — generate 24h daily plans via Markov chain + motif stats
  find_co_mobility     — find users with similar plans within the same community
  save_output          — write user_profiles to JSON files
"""

import ast
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ── allow importing from project root ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from util.common import build_loc_lookup, build_loc_multi_poi, load_config, haversine, dist_bin
from model.state import UserInitState

# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_motif_key(raw: str):
    """Parse a string like \"('STAY', 'POI', '1h')\" into a tuple."""
    return ast.literal_eval(raw)


def _build_comm_stay_probs(motif_per_comm: Dict) -> Dict[str, Dict[str, float]]:
    """
    Per-community STAY duration distribution.
    Returns: { str(comm_id): { stay_label: probability } }
    """
    result = {}
    for comm_id, counts in motif_per_comm.items():
        stay_counts: Dict[str, int] = defaultdict(int)
        for raw_key, cnt in counts.items():
            mtype, _, label = _parse_motif_key(raw_key)
            if mtype == "STAY":
                stay_counts[label] += cnt
        total = sum(stay_counts.values())
        result[str(comm_id)] = {
            lbl: cnt / total for lbl, cnt in stay_counts.items()
        } if total > 0 else {"1h": 1.0}
    return result


def _build_comm_move_ratio(motif_per_comm: Dict) -> Dict[str, float]:
    """
    P(MOVE_AB) for each community, as a motif-level transition probability.
    Only counts MOVE_AB motifs (not complex multi-hop motifs like CHAIN_ABC, ROUND_ABA)
    so the probability matches the plan generator's simplified STAY/MOVE_AB model.
    """
    result = {}
    for comm_id, counts in motif_per_comm.items():
        n_stay   = sum(v for k, v in counts.items() if _parse_motif_key(k)[0] == "STAY")
        n_moveAB = sum(v for k, v in counts.items() if _parse_motif_key(k)[0] == "MOVE_AB")
        total    = n_stay + n_moveAB
        result[str(comm_id)] = n_moveAB / total if total > 0 else 0.05
    return result


def _build_flow_from(pop_flow: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate edge flow across all 24 hours.
    Returns: { str(from_rid): { str(to_rid): total_mean_flow } }
    """
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for slot in pop_flow:
        for edge_key, flow in slot["edge_flow_mean"].items():
            fr, to = edge_key.split(",")
            agg[fr][to] += flow
    return {k: dict(v) for k, v in agg.items()}


def _build_comm_region_pop(pop_flow: List[Dict],
                           region_to_comm: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Mean population per region, grouped by community.
    Returns: { str(comm_id): { str(region_id): mean_pop_over_24h } }
    """
    pop_sum:   Dict[str, float] = defaultdict(float)
    pop_count: Dict[str, int]   = defaultdict(int)
    for slot in pop_flow:
        for rid_str, pop in slot["population_mean"].items():
            pop_sum[rid_str]   += pop
            pop_count[rid_str] += 1
    mean_pop = {rid: pop_sum[rid] / pop_count[rid] for rid in pop_sum}

    result: Dict[str, Dict[str, float]] = defaultdict(dict)
    for rid_str, pop in mean_pop.items():
        cid = region_to_comm.get(rid_str)
        if cid is not None:
            result[str(cid)][rid_str] = pop
    return dict(result)


def _sample_stay_duration(comm_id: int,
                          comm_stay_probs: Dict[str, Dict[str, float]],
                          remaining: int,
                          rng: random.Random) -> int:
    """
    Sample a stay duration (hours) from the community's stay distribution.
    Caps at `remaining` so the plan doesn't exceed 24 hours.
    """
    label_to_range = {
        "1h":     (1, 1),
        "2-3h":   (2, 3),
        "4-12h":  (4, 12),
        "13-24h": (13, 24),
    }
    probs = comm_stay_probs.get(str(comm_id), {"1h": 1.0})
    labels = list(probs.keys())
    weights = [probs[l] for l in labels]
    chosen = rng.choices(labels, weights=weights, k=1)[0]
    lo, hi = label_to_range.get(chosen, (1, 1))
    hi_eff = min(hi, remaining)
    lo_eff = min(lo, hi_eff)          # ensure lo <= hi_eff
    dur = rng.randint(lo_eff, hi_eff) if lo_eff <= hi_eff else hi_eff
    return max(1, dur)


def _sample_destination(from_loc: int,
                        comm_id: int,
                        flow_from: Dict[str, Dict[str, float]],
                        comm_region_pop: Dict[str, Dict[str, float]],
                        comm_to_locs: Dict[str, List[int]],
                        rng: random.Random) -> int:
    """
    Sample next location for a MOVE step.
    Priority: flow-based neighbors within same community → population-weighted.
    Falls back to uniform random from community locations.
    """
    comm_locs = set(comm_to_locs.get(str(comm_id), []))
    if not comm_locs:
        return from_loc

    # Use flow data: destinations within same community
    from_flows = flow_from.get(str(from_loc), {})
    candidates = {
        int(to): w for to, w in from_flows.items()
        if int(to) in comm_locs and int(to) != from_loc
    }
    if candidates:
        locs, weights = zip(*candidates.items())
        return rng.choices(list(locs), weights=list(weights), k=1)[0]

    # Fallback: population-weighted from community locations
    pop_dict = comm_region_pop.get(str(comm_id), {})
    pop_candidates = {
        int(r): pop_dict.get(r, 1.0)
        for r in map(str, comm_locs) if int(r) != from_loc
    }
    if pop_candidates:
        locs, weights = zip(*pop_candidates.items())
        return rng.choices(list(locs), weights=list(weights), k=1)[0]

    # Last resort: any other location in community
    others = [l for l in comm_locs if l != from_loc]
    return rng.choice(others) if others else from_loc


def _sample_next_motif(current: str,
                       global_transition: Dict[str, Dict[str, float]],
                       comm_move_ratio: float,
                       rng: random.Random) -> str:
    """
    Sample next motif state (motif-level, not hourly).

    After STAY  → use community's observed MOVE_AB ratio directly as P(MOVE).
    After MOVE  → use global hourly transition (MOVE events are 1h, so hourly ≈ motif-level).
    """
    if current == "STAY":
        p_move = comm_move_ratio
        p_stay = 1.0 - p_move
        return rng.choices(["STAY", "MOVE_AB"], weights=[p_stay, p_move], k=1)[0]
    else:  # MOVE_AB
        row    = global_transition.get("MOVE_AB", {"STAY": 0.826, "MOVE_AB": 0.174})
        p_move = row.get("MOVE_AB", 0.174)
        p_stay = row.get("STAY",    0.826)
        return rng.choices(["STAY", "MOVE_AB"], weights=[p_stay, p_move], k=1)[0]


def _generate_single_plan(user_id: str,
                          start_loc: int,
                          comm_id: int,
                          state: "UserInitState",
                          rng: random.Random) -> List[Dict]:
    """
    Generate a 24h mobility plan for one user using a Markov-chain over
    STAY/MOVE motifs, with durations sampled from community statistics.

    Returns: list of segments covering exactly 24 hours.
    Each segment is a dict with:
      start_hour, end_hour, duration_hours, motif_type
      STAY → location, poi_type
      MOVE_AB → from_location, to_location, from_poi, to_poi
    """
    plan: List[Dict] = []
    hour     = 0
    loc      = start_loc
    motif    = "STAY"   # users start in STAY state at hour 0

    while hour < 24:
        remaining = 24 - hour

        if motif == "STAY":
            dur = _sample_stay_duration(
                comm_id, state["comm_stay_probs"], remaining, rng
            )
            plan.append({
                "start_hour":     hour,
                "end_hour":       hour + dur,
                "duration_hours": dur,
                "motif_type":     "STAY",
                "poi_type":       state["poi_map"].get(loc, "Unknown"),
            })
            hour += dur

        else:   # MOVE_AB — always 1 hour
            next_loc = _sample_destination(
                loc, comm_id,
                state["flow_from"], state["comm_region_pop"], state["comm_to_locs"],
                rng,
            )
            # Compute haversine distance between grid cell centroids
            coord = state.get("coord_map", {})
            src = coord.get(loc)
            dst = coord.get(next_loc)
            if src and dst:
                d_km = haversine(src[0], src[1], dst[0], dst[1])
            else:
                d_km = 0.0
            plan.append({
                "start_hour":     hour,
                "end_hour":       hour + 1,
                "duration_hours": 1,
                "motif_type":     "MOVE_AB",
                "from_poi":       state["poi_map"].get(loc, "Unknown"),
                "to_poi":         state["poi_map"].get(next_loc, "Unknown"),
                "dist_km":        round(d_km, 2),
                "dist_label":     dist_bin(d_km, state["cfg"]),
            })
            loc   = next_loc
            hour += 1

        # Transition to next motif state
        if hour < 24:
            comm_move = state["comm_move_ratio"].get(str(comm_id), 0.05)
            motif = _sample_next_motif(
                motif, state["global_transition"], comm_move, rng
            )

    return plan


def _plan_to_vector(plan: List[Dict]) -> np.ndarray:
    """Encode a plan as a 24-dim int vector: 0=STAY, 1=MOVE_AB."""
    vec = np.zeros(24, dtype=np.int8)
    for seg in plan:
        if seg["motif_type"] == "MOVE_AB":
            for h in range(seg["start_hour"], min(seg["end_hour"], 24)):
                vec[h] = 1
    return vec


def _jaccard_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Jaccard similarity on the set of MOVE hours (value == 1)."""
    both   = int(np.sum((v1 == 1) & (v2 == 1)))
    either = int(np.sum((v1 == 1) | (v2 == 1)))
    if either == 0:
        return 0.0   # both stay all day → no shared movement → not co-mobile
    return both / either


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE FUNCTIONS
# Each function receives the full state, returns a partial-update dict.
# ═══════════════════════════════════════════════════════════════════════════════

def node_load_priors(state: UserInitState) -> Dict:
    """
    Node 1 — Load all pattern files and build lookup structures.

    Reads:
      community_fixed.json, motifs.json, population_flow.json,
      location.csv, mobility_train.csv, mobility_test.csv
    """
    print("\n[Node 1] Loading priors ...")
    cfg  = state["cfg"]
    city = state["city"]

    data_dir  = Path(cfg["paths"]["processed_dir"])
    graph_dir = data_dir / cfg.get("paths", {}).get("graph_subdir", "dynamic_graph")
    pat_dir   = graph_dir / "extracted_pattern"

    # ── Community structure ────────────────────────────────────────────
    with open(pat_dir / "community_fixed.json") as f:
        cf = json.load(f)
    region_to_comm = {k: int(v) for k, v in cf["region_to_comm"].items()}
    comm_to_locs   = {k: [int(x) for x in v]
                      for k, v in cf["comm_to_locs"].items()}
    n_comm = cf["n_communities"]

    # ── POI and coordinate lookup ──────────────────────────────────────
    loc_df = pd.read_csv(data_dir / "location.csv")
    poi_map, coord_map = build_loc_lookup(loc_df, cfg)
    poi_multi_map = build_loc_multi_poi(loc_df, cfg, top_k=3)
    # coord_map values are tuples → convert to lists for JSON safety
    coord_map_serial = {rid: list(lonlat) for rid, lonlat in coord_map.items()}

    # ── Motif statistics ───────────────────────────────────────────────
    with open(pat_dir / "motifs.json") as f:
        motifs = json.load(f)
    global_transition = motifs["transition_matrix"]
    comm_stay_probs   = _build_comm_stay_probs(motifs["motif_per_community"])
    comm_move_ratio   = _build_comm_move_ratio(motifs["motif_per_community"])

    # ── Flow / population ──────────────────────────────────────────────
    with open(pat_dir / "population_flow.json") as f:
        pop_flow = json.load(f)
    flow_from       = _build_flow_from(pop_flow)
    comm_region_pop = _build_comm_region_pop(pop_flow, region_to_comm)

    # Flat population map: loc_id(int) → mean population across 24h
    pop_sum: Dict[str, float] = defaultdict(float)
    pop_cnt: Dict[str, int]   = defaultdict(int)
    for slot in pop_flow:
        for rid_str, pop in slot["population_mean"].items():
            pop_sum[rid_str] += pop
            pop_cnt[rid_str] += 1
    pop_map = {int(r): pop_sum[r] / pop_cnt[r] for r in pop_sum}

    # ── Train mobility (for community assignment) ──────────────────────
    mob_train = pd.read_csv(data_dir / "mobility_train.csv")
    # Start location = location at hour 0 of ANY day, per user
    hour0_train = (mob_train[mob_train["time"] == 0]
                   .groupby("user_id")["region_id"]
                   .first()
                   .to_dict())
    train_user_start_loc = {str(uid): int(rid) for uid, rid in hour0_train.items()}
    # Community assignment from start location
    train_user_community = {
        uid: region_to_comm.get(str(rid), 0)
        for uid, rid in train_user_start_loc.items()
    }

    # ── Test users (start location only) ──────────────────────────────
    mob_test  = pd.read_csv(data_dir / "mobility_test.csv")
    hour0_test = (mob_test[mob_test["time"] == 0]
                  .sort_values("date")
                  .groupby("user_id")
                  .first()
                  .reset_index()[["user_id", "date", "region_id"]])
    test_users = [
        {"user_id": str(row.user_id),
         "start_location": int(row.region_id),
         "date": int(row.date)}
        for row in hour0_test.itertuples()
    ]

    print(f"  communities  : {n_comm}")
    print(f"  train users  : {len(train_user_start_loc)}")
    print(f"  test  users  : {len(test_users)}")
    print(f"  flow edges   : {sum(len(v) for v in flow_from.values())}")
    print(f"  pop_map locs : {len(pop_map)}")

    return {
        "region_to_comm":       region_to_comm,
        "comm_to_locs":         comm_to_locs,
        "n_communities":        n_comm,
        "poi_map":              poi_map,
        "poi_multi_map":        poi_multi_map,
        "coord_map":            coord_map_serial,
        "global_transition":    global_transition,
        "comm_stay_probs":      comm_stay_probs,
        "comm_move_ratio":      comm_move_ratio,
        "flow_from":            flow_from,
        "comm_region_pop":      comm_region_pop,
        "pop_map":              pop_map,
        "train_user_start_loc": train_user_start_loc,
        "train_user_community": train_user_community,
        "test_users":           test_users,
    }


def node_assign_communities(state: UserInitState) -> Dict:
    """
    Node 2 — Assign a community to each test user.

    Method:
      Build P(community | start_location) from training users:
        - Count how many train users whose start-location == L are in each community.
        - Normalize to get a probability distribution.
      For each test user: sample community from P(comm | start_loc).
      Fallback: community size prior (larger communities are more probable).
    """
    print("\n[Node 2] Assigning communities to test users ...")

    region_to_comm      = state["region_to_comm"]
    train_user_start    = state["train_user_start_loc"]
    train_user_comm     = state["train_user_community"]
    comm_to_locs        = state["comm_to_locs"]
    n_comm              = state["n_communities"]
    rng                 = random.Random(state.get("seed", 42))

    # Build P(comm | start_loc) from training data
    loc_comm_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for uid, rid in train_user_start.items():
        cid = train_user_comm.get(uid, 0)
        loc_comm_counts[str(rid)][str(cid)] += 1

    loc_comm_probs: Dict[str, Dict[str, float]] = {}
    for loc_str, comm_cnt in loc_comm_counts.items():
        total = sum(comm_cnt.values())
        loc_comm_probs[loc_str] = {c: n / total for c, n in comm_cnt.items()}

    # Community size prior (fallback)
    comm_sizes  = {str(cid): len(locs) for cid, locs in comm_to_locs.items()}
    total_locs  = sum(comm_sizes.values())
    size_prior  = {cid: sz / total_locs for cid, sz in comm_sizes.items()}

    # Assign community to each test user
    test_users = state["test_users"]
    assigned_users: List[Dict] = []
    comm_dist_counts: Dict[int, int] = defaultdict(int)

    for user in test_users:
        loc_str = str(user["start_location"])
        probs   = loc_comm_probs.get(loc_str, size_prior)
        comms   = [int(c) for c in probs.keys()]
        weights = list(probs.values())
        comm    = rng.choices(comms, weights=weights, k=1)[0]
        assigned_users.append({**user, "assigned_community": comm})
        comm_dist_counts[comm] += 1

    print(f"  {len(assigned_users)} test users assigned to communities:")
    for cid in sorted(comm_dist_counts):
        print(f"    community {cid:2d}: {comm_dist_counts[cid]} users")

    return {
        "loc_comm_probs": loc_comm_probs,
        "test_users":     assigned_users,   # enriched with assigned_community
    }


def node_generate_plans(state: UserInitState) -> Dict:
    """
    Node 3 — Generate a 24h daily plan for each test user.

    Algorithm:
      1. Start in STAY state at start_location (hour 0).
      2. Markov chain: P(next_motif | current_motif) from global_transition,
         scaled by community's observed MOVE ratio.
      3. STAY duration: sampled from community's stay duration distribution.
      4. MOVE destination: sampled from within-community flow weights
         (falls back to population-weighted random).
      5. Repeat until 24 hours are covered.
    """
    print("\n[Node 3] Generating 24h daily plans ...")
    rng = random.Random(state.get("seed", 42))
    test_users = state["test_users"]
    profiles: List[Dict] = []

    for i, user in enumerate(test_users):
        uid      = user["user_id"]
        start    = user["start_location"]
        comm_id  = user["assigned_community"]

        plan = _generate_single_plan(uid, start, comm_id, state, rng)
        vec  = _plan_to_vector(plan).tolist()   # 24-dim int list

        # Community POI character (dominant category across community locations)
        comm_locs = state["comm_to_locs"].get(str(comm_id), [])
        poi_votes: Dict[str, int] = defaultdict(int)
        for loc in comm_locs:
            poi_votes[state["poi_map"].get(loc, "Unknown")] += 1
        comm_poi = max(poi_votes, key=poi_votes.get) if poi_votes else "Unknown"

        n_moves = int(sum(vec))
        profiles.append({
            "user_id":               uid,
            "city":                  state["city"],
            "start_location":        start,
            "start_poi":             state["poi_map"].get(start, "Unknown"),
            "assigned_community":    comm_id,
            "community_poi_profile": comm_poi,
            "plan":                  plan,
            "plan_vector":           vec,   # 0=STAY, 1=MOVE_AB per hour
            "n_moves":               n_moves,
            "co_mobility_users":     [],    # filled by next node
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(test_users):
            print(f"  {i+1:>4}/{len(test_users)} users planned")

    return {"user_profiles": profiles}


def node_find_co_mobility(state: UserInitState) -> Dict:
    """
    Node 4 — Find co-mobility users within the same community.

    For each pair of users in the same community:
      Compute Jaccard similarity on their 24h MOVE-hour sets.
      If similarity >= co_mobility_threshold → they are co-mobile.

    Result stored in profile["co_mobility_users"] as a list of
    {"user_id": ..., "similarity": float} dicts.
    """
    print("\n[Node 4] Finding co-mobility users ...")
    threshold = state.get("co_mobility_threshold", 0.60)
    profiles  = state["user_profiles"]

    # Group profiles by community
    comm_groups: Dict[int, List[int]] = defaultdict(list)  # comm → [profile_idx]
    for idx, p in enumerate(profiles):
        comm_groups[p["assigned_community"]].append(idx)

    vecs = [np.array(p["plan_vector"], dtype=np.int8) for p in profiles]

    n_pairs_found = 0
    for comm_id, idxs in comm_groups.items():
        for i, idx_a in enumerate(idxs):
            co = []
            for idx_b in idxs:
                if idx_b == idx_a:
                    continue
                sim = _jaccard_similarity(vecs[idx_a], vecs[idx_b])
                if sim >= threshold:
                    co.append({
                        "user_id":    profiles[idx_b]["user_id"],
                        "similarity": round(float(sim), 4),
                    })
            # Sort by similarity descending
            co.sort(key=lambda x: -x["similarity"])
            profiles[idx_a]["co_mobility_users"] = co
            n_pairs_found += len(co)

    print(f"  Co-mobility threshold : {threshold}")
    print(f"  Total co-mobility links: {n_pairs_found}")
    avg = n_pairs_found / len(profiles) if profiles else 0
    print(f"  Avg co-mobile peers/user: {avg:.1f}")

    return {"user_profiles": profiles}


def node_save_output(state: UserInitState) -> Dict:
    """
    Node 5 — Persist user profiles to JSON.

    Writes:
      output/user_init/{city}_user_profiles.json   — all users in one file
      output/user_init/{city}_summary.json         — aggregate statistics
    """
    print("\n[Node 5] Saving output ...")
    cfg      = state["cfg"]
    city     = state["city"]
    profiles = state["user_profiles"]

    proj_dir = Path(__file__).parent.parent
    out_dir  = proj_dir / "output" / "user_init"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── All user profiles ─────────────────────────────────────────────
    profiles_path = out_dir / f"{city}_user_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    # ── Summary statistics ─────────────────────────────────────────────
    n_total   = len(profiles)
    comm_dist = defaultdict(int)
    n_moves_total = 0
    n_co_total    = 0
    for p in profiles:
        comm_dist[p["assigned_community"]] += 1
        n_moves_total += p["n_moves"]
        n_co_total    += len(p["co_mobility_users"])

    summary = {
        "city":              city,
        "n_test_users":      n_total,
        "n_communities":     state["n_communities"],
        "co_mobility_threshold": state.get("co_mobility_threshold", 0.60),
        "community_distribution": {
            str(k): v for k, v in sorted(comm_dist.items())
        },
        "avg_moves_per_user":   round(n_moves_total / n_total, 2) if n_total else 0,
        "avg_co_mobile_peers":  round(n_co_total / n_total, 2) if n_total else 0,
    }

    summary_path = out_dir / f"{city}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Profiles saved → {profiles_path}")
    print(f"  Summary  saved → {summary_path}")
    print(f"\n  ── Summary ────────────────────────────────────────────")
    print(f"  Users        : {summary['n_test_users']}")
    print(f"  Avg moves/day: {summary['avg_moves_per_user']}")
    print(f"  Avg co-mobile: {summary['avg_co_mobile_peers']:.1f} peers")
    print(f"  Community distribution: {summary['community_distribution']}")

    return {}
