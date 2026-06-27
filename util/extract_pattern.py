#!/usr/bin/env python3
"""
Pattern Mining from Dynamic Mobility Graphs — city-agnostic.

Usage:
  python util/extract_pattern.py --city shanghai
  python util/extract_pattern.py --city shenzhen

All parameters are read from configs/{city}.yaml.

Community detection strategy — Evolving Leiden with time smoothing:
  At each timestep we build the full graph (N_loc + dynamic users).
  We seed Leiden from the smoothed previous location-node membership so
  that user-location edge dynamics inform community evolution while the
  location structure remains temporally stable.

  After all timesteps the smoothed location memberships converge to a stable
  partition.  We relabel communities 0..K-1 and use them as the fixed location
  community structure for all outputs.  User nodes are assigned dynamically
  (community of their current location).

Outputs (saved to {processed_dir}/dynamic_graph/extracted_pattern/):
  population_flow.json      -- 24 hour-of-day slots: mean pop + mean edge flow
  community_fixed.json      -- converged region->community mapping
  communities/              -- per-(date,hour) PKL: fixed loc + dynamic user comms
  community_summary.json    -- mean users per community per hour (24 slots)
  motifs.json               -- per-day motifs (stay <= 24h), transition matrix
"""

import argparse
import json
import pickle
import sys
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from util.common import load_config, haversine, dist_bin, stay_bin, build_loc_lookup


# ============================================================
# DATA LOADING
# ============================================================

def load_train_graphs(train_dir: Path, train_dates: list) -> list:
    """Load all pkl graph files whose date is in train_dates."""
    files  = sorted(train_dir.glob('dynamic_graph_*.pkl'))
    graphs = []
    for fp in files:
        code       = int(fp.stem.split('_')[-1])
        date, hour = code // 100, code % 100
        if date in train_dates:
            with open(fp, 'rb') as f:
                g = pickle.load(f)
            graphs.append((date, hour, g))
    print(f"  Loaded {len(graphs)} graph files  "
          f"(dates {min(train_dates)}~{max(train_dates)})")
    return graphs


# ============================================================
# STEP 1 — POPULATION & EDGE-LEVEL FLOW (aggregated by hour-of-day)
# ============================================================

def compute_population_flow(graphs: list, num_days: int, out_dir: Path) -> list:
    """
    For each hour-of-day h in [0,23]:
      population_mean[region_id]  = mean # users across num_days training days
      edge_flow_mean[from,to]     = mean # users who moved along edge per day
    Flow is intra-day only (no cross-day transitions).
    """
    print("\n" + "=" * 60)
    print("STEP 1: Population & Edge-level Flow  (aggregated by hour-of-day)")
    print("=" * 60)

    pop_total  = defaultdict(lambda: defaultdict(int))  # hour -> region -> total
    edge_total = defaultdict(lambda: defaultdict(int))  # hour -> (fr,to) -> total

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
            'population_mean': {str(r): round(c / num_days, 4)
                                for r, c in pop_total[hour].items()},
            'edge_flow_mean':  {f"{e[0]},{e[1]}": round(c / num_days, 4)
                                for e, c in edge_total[hour].items()},
        })

    total_edge_slots = sum(len(r['edge_flow_mean']) for r in result)
    print(f"  24 hour-of-day slots")
    print(f"  Avg active directed edges per hour: {total_edge_slots / 24:.0f}")

    with open(out_dir / 'population_flow.json', 'w') as f:
        json.dump(result, f)
    print(f"  [OK] population_flow.json saved")
    return result


# ============================================================
# STEP 2 — EVOLVING LEIDEN WITH TIME SMOOTHING
# ============================================================

def _build_igraph_snapshot(g_pkl, n_loc, loc_id, loc_idx):
    """Build igraph for one timestep: N_loc location nodes + dynamic user nodes."""
    users   = g_pkl['user_nodes']
    n_usr   = len(users)
    usr_idx = {uid: n_loc + i for i, uid in enumerate(users)}

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


def _build_seed_membership(prev_loc_mem, user_rids, loc_idx):
    """
    Seed membership for Leiden warm-start:
      - Location nodes: use converged previous membership.
      - User nodes    : community of their current location.
    """
    seed = list(prev_loc_mem)
    for rid in user_rids:
        idx = loc_idx.get(rid)
        seed.append(prev_loc_mem[idx] if idx is not None else 0)
    return seed


def _apply_smooth(prev_loc_mem, new_loc_mem, alpha, rng):
    """
    Stochastic time-smoothing: keep the PREVIOUS community assignment with
    probability alpha, else accept the new Leiden result.
    """
    return [
        prev_loc_mem[i] if rng.random() < alpha else new_loc_mem[i]
        for i in range(len(prev_loc_mem))
    ]


def compute_evolving_communities(graphs: list, loc_df: pd.DataFrame,
                                  cfg: dict, out_dir: Path,
                                  comm_dir: Path) -> tuple:
    """
    Runs evolving Leiden across all timesteps:
      1. Cold-start: Leiden on static loc-loc graph to initialise memberships.
      2. Evolving pass: warm-start from previous smoothed memberships, run Leiden,
         apply alpha-smoothing.
      3. Relabel converged loc communities 0..K-1 -> fixed community structure.
    """
    pcfg       = cfg['pattern']
    resolution = pcfg['leiden_resolution']
    seed       = pcfg['leiden_seed']
    alpha      = pcfg['leiden_smooth_alpha']
    n_iter_c   = pcfg['leiden_n_iter_cold']
    n_iter_w   = pcfg['leiden_n_iter_warm']
    num_days   = len(cfg['time']['train_dates'])

    print("\n" + "=" * 60)
    print("STEP 2: Evolving Leiden Community Detection (with time smoothing)")
    print(f"  Resolution={resolution}  Alpha={alpha}  Seed={seed}")
    print(f"  Cold-start iters={n_iter_c}  Warm iters={n_iter_w}")
    print("=" * 60)

    comm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    _, _, g0  = graphs[0]
    n_loc     = len(g0['loc_nodes'])
    loc_id    = g0['loc_nodes'].tolist()
    loc_idx   = {rid: i for i, rid in enumerate(loc_id)}

    # -- Cold-start: Leiden on static loc-loc graph --
    print(f"  Cold-start: Leiden on static loc-loc graph ({n_loc} nodes) ...")
    edges_init, weights_init = [], []
    for src, dst, w in zip(g0['loc_loc_src'], g0['loc_loc_dst'], g0['loc_loc_w']):
        edges_init.append((loc_idx[src], loc_idx[dst]))
        weights_init.append(float(w))
    ig_init = ig.Graph(n=n_loc, edges=edges_init, directed=False)
    ig_init.es['weight'] = weights_init
    part0 = leidenalg.find_partition(
        ig_init, leidenalg.CPMVertexPartition,
        weights='weight', resolution_parameter=resolution,
        seed=seed, n_iterations=n_iter_c,
    )
    smoothed_loc_mem = list(part0.membership)
    print(f"  Cold-start communities: {len(set(smoothed_loc_mem))}")

    # -- Evolving pass --
    community_series = []
    for idx, (date, hour, g_pkl) in enumerate(graphs):
        ig_g, users, user_rids = _build_igraph_snapshot(g_pkl, n_loc, loc_id, loc_idx)
        init_mem = _build_seed_membership(smoothed_loc_mem, user_rids, loc_idx)

        partition = leidenalg.find_partition(
            ig_g, leidenalg.CPMVertexPartition,
            weights='weight', resolution_parameter=resolution,
            initial_membership=init_mem, seed=seed, n_iterations=n_iter_w,
        )
        new_loc_mem      = partition.membership[:n_loc]
        smoothed_loc_mem = _apply_smooth(smoothed_loc_mem, new_loc_mem, alpha, rng)

        community_series.append((date, hour, g_pkl, users, user_rids,
                                  list(smoothed_loc_mem)))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(graphs):
            n_comms = len(set(smoothed_loc_mem))
            print(f"  [{idx+1:>4}/{len(graphs)}] date={date:04d} h={hour:02d}  "
                  f"loc-communities (smoothed)={n_comms}")

    # -- Relabel converged location communities 0..K-1 --
    final_loc_mem = smoothed_loc_mem
    raw_ids       = sorted(set(final_loc_mem))
    relabel       = {old: new for new, old in enumerate(raw_ids)}
    final_loc_mem = [relabel[c] for c in final_loc_mem]
    n_communities = len(set(final_loc_mem))

    region_to_comm = {loc_id[i]: final_loc_mem[i] for i in range(n_loc)}
    comm_to_locs   = defaultdict(list)
    for rid, cid in region_to_comm.items():
        comm_to_locs[cid].append(rid)
    community_ids = sorted(comm_to_locs.keys())

    print(f"\n  Converged to {n_communities} communities (FIXED for output)")
    for cid in community_ids:
        print(f"    Community {cid:2d}: {len(comm_to_locs[cid]):3d} locations")

    # -- Save fixed community structure --
    fixed_out = {
        'n_communities':  n_communities,
        'community_ids':  community_ids,
        'resolution':     resolution,
        'smooth_alpha':   alpha,
        'region_to_comm': {str(k): v for k, v in region_to_comm.items()},
        'comm_to_locs':   {str(k): v for k, v in comm_to_locs.items()},
    }
    with open(out_dir / 'community_fixed.json', 'w') as f:
        json.dump(fixed_out, f, indent=2)
    print(f"  [OK] community_fixed.json saved")

    # -- Save per-timestep PKLs and community summary --
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
        entry = {
            'date': date, 'hour': hour,
            'n_communities': n_communities,
            'communities': communities,
        }
        full_series.append(entry)
        for cid in community_ids:
            hour_comm_user_total[hour][cid] += len(comm_users[cid])

        fname = comm_dir / f"comm_{date:04d}{hour:02d}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = []
    for hour in range(24):
        summary.append({
            'hour':                hour,
            'n_communities':       n_communities,
            'mean_users_per_comm': {
                str(cid): round(hour_comm_user_total[hour][cid] / num_days, 2)
                for cid in community_ids
            },
        })
    with open(out_dir / 'community_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [OK] {len(full_series)} community PKLs saved -> {comm_dir}")
    print(f"  [OK] community_summary.json saved")
    return full_series, region_to_comm, comm_to_locs


# ============================================================
# STEP 3 — MOTIF MINING (per-day, stay <= 24h)
# ============================================================

MOTIF_TYPES = ['STAY', 'MOVE_AB', 'ROUND_ABA', 'CHAIN_ABC', 'RETURN_ABCB', 'FULL_ABCBA']


def _extract_user_sequences_by_day(mob_train: pd.DataFrame) -> dict:
    mob = mob_train.sort_values(['user_id', 'date', 'time'])
    seq = defaultdict(dict)
    for (uid, date), grp in mob.groupby(['user_id', 'date']):
        seq[uid][date] = list(zip(grp['time'].tolist(), grp['region_id'].tolist()))
    return seq


def _mine_motifs_for_day(day_seq: list, poi_map: dict,
                          coord_map: dict, cfg: dict) -> dict:
    """Extract motif counts for a single user-day sequence."""
    counts = defaultdict(int)

    # Compress consecutive same-location records into (region, duration) runs
    runs, i = [], 0
    while i < len(day_seq):
        rid = day_seq[i][1]
        j   = i
        while j < len(day_seq) and day_seq[j][1] == rid:
            j += 1
        runs.append((rid, j - i))
        i = j

    # STAY motifs
    for rid, dur in runs:
        counts[('STAY', poi_map.get(rid, 'Unknown'), stay_bin(dur, cfg))] += 1

    # Movement motifs: A->B, A->B->A, A->B->C, A->B->C->B, A->B->C->B->A
    for k in range(len(runs) - 1):
        rA, _ = runs[k]
        rB, _ = runs[k + 1]
        if rA == rB:
            continue

        lonA, latA = coord_map.get(rA, (0.0, 0.0))
        lonB, latB = coord_map.get(rB, (0.0, 0.0))
        d_AB    = haversine(lonA, latA, lonB, latB)
        dbin_AB = dist_bin(d_AB, cfg)
        pA, pB  = poi_map.get(rA, 'Unknown'), poi_map.get(rB, 'Unknown')

        counts[('MOVE_AB', f'{pA}->{pB}', dbin_AB)] += 1

        if k + 2 < len(runs):
            rC, _ = runs[k + 2]
            if rC == rA:
                counts[('ROUND_ABA', f'{pA}->{pB}->{pA}', dbin_AB)] += 1
            elif rC != rB:
                lonC, latC = coord_map.get(rC, (0.0, 0.0))
                d_BC    = haversine(lonB, latB, lonC, latC)
                dbin_BC = dist_bin(d_BC, cfg)
                pC      = poi_map.get(rC, 'Unknown')
                counts[('CHAIN_ABC', f'{pA}->{pB}->{pC}',
                         f'{dbin_AB}|{dbin_BC}')] += 1

                if k + 3 < len(runs):
                    rD, _ = runs[k + 3]
                    if rD == rB:
                        counts[('RETURN_ABCB',
                                f'{pA}->{pB}->{pC}->{pB}',
                                f'{dbin_AB}|{dbin_BC}|{dbin_BC}')] += 1
                        if k + 4 < len(runs):
                            rE, _ = runs[k + 4]
                            if rE == rA:
                                counts[('FULL_ABCBA',
                                        f'{pA}->{pB}->{pC}->{pB}->{pA}',
                                        f'{dbin_AB}|{dbin_BC}|{dbin_BC}|{dbin_AB}')] += 1
    return counts


def _compute_motif_transition_matrix(user_day_seqs: dict) -> dict:
    """Compute STAY <-> MOVE_AB transition probabilities across all user-days."""
    trans = defaultdict(lambda: defaultdict(int))
    for uid, day_dict in user_day_seqs.items():
        for date, day_seq in day_dict.items():
            prev = None
            for i in range(len(day_seq) - 1):
                m = 'STAY' if day_seq[i][1] == day_seq[i + 1][1] else 'MOVE_AB'
                if prev is not None:
                    trans[prev][m] += 1
                prev = m
    all_m  = sorted(set(list(trans) + [m for row in trans.values() for m in row]))
    matrix = {}
    for src in all_m:
        total = sum(trans[src].values())
        matrix[src] = {dst: round(trans[src][dst] / total, 6) if total else 0.0
                       for dst in all_m}
    return matrix


def mine_motifs(user_day_seqs: dict, loc_df: pd.DataFrame,
                region_to_comm: dict, cfg: dict, out_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print("STEP 3: Motif Mining (per-day, stay <= 24h)")
    print("=" * 60)

    poi_map, coord_map = build_loc_lookup(loc_df, cfg)
    global_counts = defaultdict(int)
    comm_counts   = defaultdict(lambda: defaultdict(int))

    total_user_days = 0
    for uid, day_dict in user_day_seqs.items():
        for date, day_seq in day_dict.items():
            if len(day_seq) < 2:
                continue
            total_user_days += 1
            c = _mine_motifs_for_day(day_seq, poi_map, coord_map, cfg)
            for k, v in c.items():
                global_counts[k] += v

            # Assign user-day to community of their most-visited location
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

    print(f"  User-days processed   : {total_user_days:,}")
    print(f"  Total motif instances : {sum(type_counts.values()):,}")
    for mtype in MOTIF_TYPES:
        print(f"    {mtype:<15s}: {type_counts.get(mtype, 0):>8,}")

    pcfg = cfg['pattern']
    output = {
        'motif_counts':        {str(k): v for k, v in global_counts.items()},
        'motif_type_totals':   dict(type_counts),
        'motif_per_community': {str(cid): {str(k): v for k, v in cmot.items()}
                                for cid, cmot in comm_counts.items()},
        'transition_matrix':   _compute_motif_transition_matrix(user_day_seqs),
        'initial_probs_hour0': {'STAY': 1.0},
        'stay_bins':           pcfg['stay_labels'],
        'dist_bins':           pcfg['dist_labels'],
        'motif_types':         MOTIF_TYPES,
    }
    with open(out_dir / 'motifs.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [OK] motifs.json saved")
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Mine patterns from dynamic mobility graphs.')
    parser.add_argument('--city', required=True, choices=['shanghai', 'shenzhen'],
                        help='City to process -- determines which config file to load.')
    args = parser.parse_args()

    cfg = load_config(args.city)

    processed_dir = Path(cfg['paths']['processed_dir'])
    graph_dir     = processed_dir / 'dynamic_graph'
    train_dir     = graph_dir / 'train'
    out_dir       = graph_dir / 'extracted_pattern'
    comm_dir      = out_dir / 'communities'

    train_dates = cfg['time']['train_dates']
    num_days    = len(train_dates)
    max_date    = max(train_dates)

    print("=" * 60)
    print(f"Pattern Mining Pipeline  [{cfg['city']}]  "
          f"(dates {min(train_dates)}~{max_date})")
    print("=" * 60)

    out_dir.mkdir(parents=True, exist_ok=True)
    comm_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading data ...")
    loc_df    = pd.read_csv(processed_dir / 'location.csv')
    mob_train = pd.read_csv(processed_dir / 'mobility_train.csv')
    mob_train = mob_train[mob_train['date'].astype(int) <= max_date].copy()
    print(f"  location rows : {len(loc_df)}")
    print(f"  mobility rows : {len(mob_train):,}  "
          f"| users: {mob_train['user_id'].nunique():,}  "
          f"| dates: {mob_train['date'].min()}~{mob_train['date'].max()}")

    print("\nLoading dynamic graph PKLs ...")
    graphs = load_train_graphs(train_dir, train_dates)

    pop_flow = compute_population_flow(graphs, num_days, out_dir)

    community_series, region_to_comm, comm_to_locs = \
        compute_evolving_communities(graphs, loc_df, cfg, out_dir, comm_dir)

    user_day_seqs = _extract_user_sequences_by_day(mob_train)
    motifs = mine_motifs(user_day_seqs, loc_df, region_to_comm, cfg, out_dir)

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
