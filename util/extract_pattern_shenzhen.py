#!/usr/bin/env python3
"""
Pattern Mining for Shenzhen Private Car Mobility  (v3 — same logic as Shanghai)

Differences from the Shanghai version:
  - DATA_DIR points to shenzhen/
  - 500 location nodes (25×20 grid) instead of 400
  - Date range: 1101~1114 (1115 reserved for test)
  - user_id is integer (ObjectID) not a string
  - TRAIN_DATES: 1101~1114

All algorithm logic (evolving Leiden, population/flow, motifs) is identical.
"""

import json
import pickle
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path('/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration')
DATA_DIR    = PROJECT_DIR / 'data' / 'shenzhen'
GRAPH_DIR   = DATA_DIR / 'dyanmic_graph'
TRAIN_DIR   = GRAPH_DIR / 'train'
OUT_DIR     = GRAPH_DIR / 'extracted_pattern'
COMM_DIR    = OUT_DIR / 'communities'

# ============================================================
# HYPERPARAMETERS
# ============================================================

LEIDEN_SEED       = 42
LEIDEN_RESOLUTION = 0.20   # CPMVertexPartition; tune for community granularity
SMOOTH_ALPHA      = 0.4
# Use 1101~1114 only; 1115 reserved for test
TRAIN_DATES       = list(range(1101, 1115))
NUM_DAYS          = 14

STAY_BINS   = [0, 1, 3, 12, 24]
STAY_LABELS = ['1h', '2-3h', '4-12h', '13-24h']

DIST_BINS   = [0, 1, 2, 3, 5, float('inf')]
DIST_LABELS = ['0-1km', '1-2km', '2-3km', '3-5km', '>5km']

MOTIF_TYPES = ['STAY', 'MOVE_AB', 'ROUND_ABA', 'CHAIN_ABC', 'RETURN_ABCB', 'FULL_ABCBA']

POI_CATEGORIES = [
    'Transportation Facilities', 'Leisure & Entertainment', 'Companies & Enterprises',
    'Healthcare', 'Commercial & Residential', 'Tourist Attractions', 'Automotive',
    'Life Services', 'Science & Education & Culture', 'Shopping & Consumer Goods',
    'Sports & Fitness', 'Hotels & Accommodations', 'Financial Institutions', 'Dining & Cuisine',
]


# ============================================================
# HELPERS
# ============================================================

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dist_bin(d_km):
    for i, upper in enumerate(DIST_BINS[1:]):
        if d_km <= upper:
            return DIST_LABELS[i]
    return DIST_LABELS[-1]


def stay_bin(h):
    for i, upper in enumerate(STAY_BINS[1:]):
        if h <= upper:
            return STAY_LABELS[i]
    return STAY_LABELS[-1]


def load_train_graphs():
    files = sorted(TRAIN_DIR.glob('dynamic_graph_*.pkl'))
    graphs = []
    for fp in files:
        code = int(fp.stem.split('_')[-1])
        date, hour = code // 100, code % 100
        if date in TRAIN_DATES:
            with open(fp, 'rb') as f:
                g = pickle.load(f)
            graphs.append((date, hour, g))
    print(f"  Loaded {len(graphs)} graph files for dates 1101~1114")
    return graphs


def build_loc_lookup(loc_df):
    poi_map, coord_map = {}, {}
    cat_cols = [c for c in POI_CATEGORIES if c in loc_df.columns]
    for _, row in loc_df.iterrows():
        rid = int(row['region_id'])
        coord_map[rid] = (float(row['lon_center']), float(row['lat_center']))
        if cat_cols:
            vals = row[cat_cols]
            poi_map[rid] = vals.idxmax() if vals.max() > 0 else 'Unknown'
        else:
            poi_map[rid] = 'Unknown'
    return poi_map, coord_map


# ============================================================
# STEP 1 — POPULATION & EDGE-LEVEL FLOW
# ============================================================

def compute_population_flow(graphs):
    print("\n" + "=" * 60)
    print("STEP 1: Population & Edge-level Flow  (aggregated by hour-of-day)")
    print("=" * 60)

    pop_total  = defaultdict(lambda: defaultdict(int))
    edge_total = defaultdict(lambda: defaultdict(int))

    by_day = defaultdict(dict)
    for date, hour, g in graphs:
        by_day[date][hour] = g

    for date in sorted(by_day.keys()):
        prev_user_loc = {}
        for hour in range(24):
            g = by_day[date].get(hour)
            if g is None:
                prev_user_loc = {}
                continue
            user_locs = dict(zip(g['user_loc_user'], g['user_loc_loc'].tolist()))
            for uid, rid in user_locs.items():
                pop_total[hour][rid] += 1
                prev = prev_user_loc.get(uid)
                if prev is not None and prev != rid:
                    edge_total[hour][(prev, rid)] += 1
            prev_user_loc = user_locs

    result = []
    for hour in range(24):
        result.append({
            'hour':            hour,
            'population_mean': {str(r): round(c / NUM_DAYS, 4)
                                for r, c in pop_total[hour].items()},
            'edge_flow_mean':  {f"{e[0]},{e[1]}": round(c / NUM_DAYS, 4)
                                for e, c in edge_total[hour].items()},
        })

    total_edge_slots = sum(len(r['edge_flow_mean']) for r in result)
    print(f"  24 hour-of-day slots")
    print(f"  Avg active directed edges per hour: {total_edge_slots / 24:.0f}")

    with open(OUT_DIR / 'population_flow.json', 'w') as f:
        json.dump(result, f)
    print(f"  [OK] population_flow.json saved")
    return result


# ============================================================
# STEP 2 — EVOLVING LEIDEN WITH TIME SMOOTHING
# ============================================================

def build_igraph_snapshot(g_pkl, n_loc, loc_id, loc_idx):
    users    = g_pkl['user_nodes']
    n_usr    = len(users)
    usr_idx  = {uid: n_loc + i for i, uid in enumerate(users)}

    edges, weights = [], []
    for src, dst, w in zip(g_pkl['loc_loc_src'], g_pkl['loc_loc_dst'], g_pkl['loc_loc_w']):
        edges.append((loc_idx[src], loc_idx[dst]))
        weights.append(float(w))
    for uid, rid in zip(g_pkl['user_loc_user'], g_pkl['user_loc_loc']):
        edges.append((usr_idx[uid], loc_idx[rid]))
        weights.append(1.0)

    ig_g = ig.Graph(n=n_loc + n_usr, edges=edges, directed=False)
    ig_g.es['weight'] = weights
    ig_g.vs['ntype']  = ['loc'] * n_loc + ['user'] * n_usr
    user_rids = g_pkl['user_loc_loc'].tolist()
    return ig_g, users, user_rids


def build_seed_membership(prev_loc_mem, n_loc, user_rids, loc_idx):
    seed = list(prev_loc_mem)
    for rid in user_rids:
        idx = loc_idx.get(rid)
        seed.append(prev_loc_mem[idx] if idx is not None else 0)
    return seed


def apply_smooth(prev_loc_mem, new_loc_mem, alpha, rng):
    return [
        prev_loc_mem[i] if rng.random() < alpha else new_loc_mem[i]
        for i in range(len(prev_loc_mem))
    ]


def compute_evolving_communities(graphs, loc_df):
    print("\n" + "=" * 60)
    print("STEP 2: Evolving Leiden Community Detection (with time smoothing)")
    print(f"  Resolution={LEIDEN_RESOLUTION}  Alpha={SMOOTH_ALPHA}")
    print("=" * 60)

    COMM_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(LEIDEN_SEED)

    _, _, g0  = graphs[0]
    n_loc     = len(g0['loc_nodes'])
    loc_id    = g0['loc_nodes'].tolist()
    loc_idx   = {rid: i for i, rid in enumerate(loc_id)}

    # Cold-start: Leiden on static loc-loc graph
    print(f"  Cold-start: Leiden on static loc-loc graph ({n_loc} nodes) ...")
    edges_init, weights_init = [], []
    for src, dst, w in zip(g0['loc_loc_src'], g0['loc_loc_dst'], g0['loc_loc_w']):
        edges_init.append((loc_idx[src], loc_idx[dst]))
        weights_init.append(float(w))
    ig_init = ig.Graph(n=n_loc, edges=edges_init, directed=False)
    ig_init.es['weight'] = weights_init
    part0 = leidenalg.find_partition(
        ig_init, leidenalg.CPMVertexPartition,
        weights='weight', resolution_parameter=LEIDEN_RESOLUTION,
        seed=LEIDEN_SEED, n_iterations=20,
    )
    smoothed_loc_mem = list(part0.membership)
    print(f"  Cold-start communities: {len(set(smoothed_loc_mem))}")

    community_series = []
    for idx, (date, hour, g_pkl) in enumerate(graphs):
        ig_g, users, user_rids = build_igraph_snapshot(g_pkl, n_loc, loc_id, loc_idx)

        seed = build_seed_membership(smoothed_loc_mem, n_loc, user_rids, loc_idx)
        partition = leidenalg.find_partition(
            ig_g, leidenalg.CPMVertexPartition,
            weights='weight', resolution_parameter=LEIDEN_RESOLUTION,
            initial_membership=seed, seed=LEIDEN_SEED, n_iterations=5,
        )
        new_loc_mem      = partition.membership[:n_loc]
        smoothed_loc_mem = apply_smooth(smoothed_loc_mem, new_loc_mem, SMOOTH_ALPHA, rng)

        community_series.append((date, hour, g_pkl, users, user_rids,
                                  list(smoothed_loc_mem)))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(graphs):
            print(f"  [{idx+1:>4}/{len(graphs)}] date={date:04d} h={hour:02d}  "
                  f"loc-communities (smoothed)={len(set(smoothed_loc_mem))}")

    # Relabel converged communities 0..K-1
    final_loc_mem  = smoothed_loc_mem
    raw_comm_ids   = sorted(set(final_loc_mem))
    relabel        = {old: new for new, old in enumerate(raw_comm_ids)}
    final_loc_mem  = [relabel[c] for c in final_loc_mem]
    n_communities  = len(set(final_loc_mem))

    region_to_comm = {loc_id[i]: final_loc_mem[i] for i in range(n_loc)}
    comm_to_locs   = defaultdict(list)
    for rid, cid in region_to_comm.items():
        comm_to_locs[cid].append(rid)
    community_ids = sorted(comm_to_locs.keys())

    print(f"\n  Converged to {n_communities} communities (FIXED)")
    for cid in community_ids:
        print(f"    Community {cid:2d}: {len(comm_to_locs[cid]):3d} locations")

    fixed_out = {
        'n_communities':  n_communities,
        'community_ids':  community_ids,
        'resolution':     LEIDEN_RESOLUTION,
        'smooth_alpha':   SMOOTH_ALPHA,
        'region_to_comm': {str(k): v for k, v in region_to_comm.items()},
        'comm_to_locs':   {str(k): v for k, v in comm_to_locs.items()},
    }
    with open(OUT_DIR / 'community_fixed.json', 'w') as f:
        json.dump(fixed_out, f, indent=2)

    hour_comm_user_total = defaultdict(lambda: defaultdict(int))
    full_series = []
    for (date, hour, g_pkl, users, user_rids, _) in community_series:
        comm_users = {cid: [] for cid in community_ids}
        for uid, rid in zip(users, user_rids):
            cid = region_to_comm.get(rid)
            if cid is not None:
                comm_users[cid].append(uid)

        communities = {
            cid: {'loc_nodes': comm_to_locs[cid], 'user_nodes': comm_users[cid]}
            for cid in community_ids
        }
        entry = {'date': date, 'hour': hour,
                 'n_communities': n_communities, 'communities': communities}
        full_series.append(entry)

        for cid in community_ids:
            hour_comm_user_total[hour][cid] += len(comm_users[cid])

        fname = COMM_DIR / f"comm_{date:04d}{hour:02d}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = []
    for hour in range(24):
        summary.append({
            'hour': hour, 'n_communities': n_communities,
            'mean_users_per_comm': {
                str(cid): round(hour_comm_user_total[hour][cid] / NUM_DAYS, 2)
                for cid in community_ids},
        })
    with open(OUT_DIR / 'community_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [OK] {len(full_series)} community PKLs → {COMM_DIR}")
    print(f"  [OK] community_fixed.json + community_summary.json saved")
    return full_series, region_to_comm, comm_to_locs


# ============================================================
# STEP 3 — MOTIF MINING
# ============================================================

def extract_user_sequences_by_day(mob_train):
    mob = mob_train.sort_values(['user_id', 'date', 'time'])
    seq = defaultdict(dict)
    for (uid, date), grp in mob.groupby(['user_id', 'date']):
        seq[uid][date] = list(zip(grp['time'].tolist(), grp['region_id'].tolist()))
    return seq


def mine_motifs_for_day(day_seq, poi_map, coord_map):
    counts = defaultdict(int)
    runs = []
    i = 0
    while i < len(day_seq):
        rid = day_seq[i][1]
        j = i
        while j < len(day_seq) and day_seq[j][1] == rid:
            j += 1
        runs.append((rid, j - i))
        i = j

    for rid, dur in runs:
        counts[('STAY', poi_map.get(rid, 'Unknown'), stay_bin(dur))] += 1

    for k in range(len(runs) - 1):
        rA, _ = runs[k]
        rB, _ = runs[k + 1]
        if rA == rB:
            continue
        lonA, latA = coord_map.get(rA, (0.0, 0.0))
        lonB, latB = coord_map.get(rB, (0.0, 0.0))
        d_AB    = haversine(lonA, latA, lonB, latB)
        dbin_AB = dist_bin(d_AB)
        pA, pB  = poi_map.get(rA, 'Unknown'), poi_map.get(rB, 'Unknown')
        counts[('MOVE_AB', f'{pA}→{pB}', dbin_AB)] += 1

        if k + 2 < len(runs):
            rC, _ = runs[k + 2]
            if rC == rA:
                counts[('ROUND_ABA', f'{pA}→{pB}→{pA}', dbin_AB)] += 1
            elif rC != rB:
                lonC, latC = coord_map.get(rC, (0.0, 0.0))
                d_BC    = haversine(lonB, latB, lonC, latC)
                dbin_BC = dist_bin(d_BC)
                pC      = poi_map.get(rC, 'Unknown')
                counts[('CHAIN_ABC', f'{pA}→{pB}→{pC}', f'{dbin_AB}|{dbin_BC}')] += 1
                if k + 3 < len(runs):
                    rD, _ = runs[k + 3]
                    if rD == rB:
                        counts[('RETURN_ABCB',
                                f'{pA}→{pB}→{pC}→{pB}',
                                f'{dbin_AB}|{dbin_BC}|{dbin_BC}')] += 1
                        if k + 4 < len(runs):
                            rE, _ = runs[k + 4]
                            if rE == rA:
                                counts[('FULL_ABCBA',
                                        f'{pA}→{pB}→{pC}→{pB}→{pA}',
                                        f'{dbin_AB}|{dbin_BC}|{dbin_BC}|{dbin_AB}')] += 1
    return counts


def compute_motif_transition_matrix(user_day_seqs):
    trans = defaultdict(lambda: defaultdict(int))
    for uid, day_dict in user_day_seqs.items():
        for date, day_seq in day_dict.items():
            prev = None
            for i in range(len(day_seq) - 1):
                m = 'STAY' if day_seq[i][1] == day_seq[i + 1][1] else 'MOVE_AB'
                if prev is not None:
                    trans[prev][m] += 1
                prev = m
    all_m = sorted(set(list(trans) + [m for row in trans.values() for m in row]))
    matrix = {}
    for src in all_m:
        total = sum(trans[src].values())
        matrix[src] = {dst: round(trans[src][dst] / total, 6) if total else 0.0
                       for dst in all_m}
    return matrix


def mine_motifs(user_day_seqs, loc_df, region_to_comm):
    print("\n" + "=" * 60)
    print("STEP 3: Motif Mining (per-day, stay ≤ 24h)")
    print("=" * 60)

    poi_map, coord_map = build_loc_lookup(loc_df)
    global_counts = defaultdict(int)
    comm_counts   = defaultdict(lambda: defaultdict(int))

    total_user_days = 0
    for uid, day_dict in user_day_seqs.items():
        for date, day_seq in day_dict.items():
            if len(day_seq) < 2:
                continue
            total_user_days += 1
            c = mine_motifs_for_day(day_seq, poi_map, coord_map)
            for k, v in c.items():
                global_counts[k] += v
            loc_cnt = defaultdict(int)
            for h, r in day_seq:
                loc_cnt[r] += 1
            dom_r = max(loc_cnt, key=loc_cnt.get)
            cid   = region_to_comm.get(dom_r)
            if cid is not None:
                for k, v in c.items():
                    comm_counts[cid][k] += v

    type_counts = defaultdict(int)
    for (mtype, _, _), v in global_counts.items():
        type_counts[mtype] += v

    print(f"  Vehicle-days processed: {total_user_days:,}")
    print(f"  Total motif instances : {sum(type_counts.values()):,}")
    for mtype in MOTIF_TYPES:
        print(f"    {mtype:<15s}: {type_counts.get(mtype, 0):>8,}")

    trans_matrix = compute_motif_transition_matrix(user_day_seqs)
    init_probs   = {'STAY': 1.0}

    output = {
        'motif_counts':         {str(k): v for k, v in global_counts.items()},
        'motif_type_totals':    dict(type_counts),
        'motif_per_community':  {str(cid): {str(k): v for k, v in cmot.items()}
                                 for cid, cmot in comm_counts.items()},
        'transition_matrix':    trans_matrix,
        'initial_probs_hour0':  init_probs,
        'stay_bins':            STAY_LABELS,
        'dist_bins':            DIST_LABELS,
        'motif_types':          MOTIF_TYPES,
    }
    with open(OUT_DIR / 'motifs.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [OK] motifs.json saved")
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Pattern Mining — Shenzhen  (1101~1114, 500 loc nodes)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMM_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data ...")
    loc_df    = pd.read_csv(DATA_DIR / 'location.csv')
    mob_train = pd.read_csv(DATA_DIR / 'mobility_train.csv')
    mob_train = mob_train[mob_train['date'].astype(int) <= 1114].copy()
    print(f"  location rows : {len(loc_df)}")
    print(f"  mobility rows : {len(mob_train):,}  "
          f"| vehicles: {mob_train['user_id'].nunique():,}  "
          f"| dates: {mob_train['date'].min()}~{mob_train['date'].max()}")

    print("\nLoading dynamic graph PKLs (1101~1114) ...")
    graphs = load_train_graphs()

    pop_flow = compute_population_flow(graphs)

    community_series, region_to_comm, comm_to_locs = \
        compute_evolving_communities(graphs, loc_df)

    user_day_seqs = extract_user_sequences_by_day(mob_train)
    motifs = mine_motifs(user_day_seqs, loc_df, region_to_comm)

    n_comms = len(set(region_to_comm.values()))
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  population_flow.json : 24 hour-of-day slots")
    print(f"  community_fixed.json : {n_comms} converged communities")
    print(f"  communities/         : {len(community_series)} PKL files")
    print(f"  motifs.json          : "
          f"{sum(motifs['motif_type_totals'].values()):,} total motif instances")
    print("=" * 60)


if __name__ == '__main__':
    main()
