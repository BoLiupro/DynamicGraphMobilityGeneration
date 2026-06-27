#!/usr/bin/env python3
"""
Unified Dynamic Graph Construction — city-agnostic.

Usage:
  python scripts/build_dynamic_graph.py --city shanghai
  python scripts/build_dynamic_graph.py --city shenzhen

All parameters are read from configs/{city}.yaml.

Graph structure at each timestep (date, hour):
  Nodes:
    - Location nodes : all grid regions (static; 400 for Shanghai / 500 for Shenzhen)
    - User nodes     : users/vehicles present at this timestep

  Edges:
    - loc-loc : static edges derived from training mobility.
                w(i→j) = exp(-dist/sigma_km) × (1 + alpha_flow × flow_norm)
                Sparsified: keep the top flow_topk fraction by weight.
    - user-loc: dynamic edges — weight = 1.0

Outputs (under {processed_dir}/dynamic_graph/):
  loc_loc_edges.csv
  train/dynamic_graph_{mmdd}{hh}.pkl
  test/ dynamic_graph_{mmdd}{hh}.pkl

Each pkl is a dict:
  {
    'date'          : int  (e.g. 601 for June 1)
    'hour'          : int  (0–23)
    'loc_nodes'     : np.int32[N_loc]        all region IDs
    'user_nodes'    : list                   user IDs active at this timestep
    'loc_loc_src'   : np.int32[E_ll]         loc-loc source region IDs
    'loc_loc_dst'   : np.int32[E_ll]         loc-loc dest   region IDs
    'loc_loc_w'     : np.float32[E_ll]       loc-loc weights
    'user_loc_user' : list                   user IDs (parallel to user_loc_loc)
    'user_loc_loc'  : np.int32[E_ul]         region IDs (one per user)
    'user_loc_w'    : np.float32[E_ul]       all 1.0
  }
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path so util.common is importable
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from util.common import load_config, haversine_matrix


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data(cfg: dict) -> tuple:
    processed_dir = Path(cfg['paths']['processed_dir'])
    user_label    = cfg.get('user_label', 'Users')

    print("=" * 60)
    print(f"STEP 1: Loading data  [{cfg['city']}]")
    print("=" * 60)

    loc       = pd.read_csv(processed_dir / 'location.csv')
    mob_train = pd.read_csv(processed_dir / 'mobility_train.csv')
    mob_test  = pd.read_csv(processed_dir / 'mobility_test.csv')

    print(f"  location.csv   : {len(loc)} regions")
    print(f"  mobility_train : {len(mob_train):,} rows | "
          f"{mob_train['user_id'].nunique()} {user_label} | "
          f"dates {mob_train['date'].min()}~{mob_train['date'].max()}")
    print(f"  mobility_test  : {len(mob_test):,} rows | "
          f"{mob_test['user_id'].nunique()} {user_label} | "
          f"dates {mob_test['date'].min()}~{mob_test['date'].max()}")

    return loc, mob_train, mob_test


# ============================================================
# STEP 2 — LOC-LOC EDGES  (derived from training mobility only)
# ============================================================

def compute_loc_loc_edges(mob_train: pd.DataFrame,
                          loc: pd.DataFrame,
                          cfg: dict) -> pd.DataFrame:
    """
    Method B: distance-base + flow bonus.
      Weight  : w(i→j) = exp(-dist/sigma_km) × (1 + alpha_flow × flow_norm)
      Candidates: all (i,j) pairs with dist < max_dist_km.
      Sparsify  : keep top flow_topk fraction by combined weight.
    """
    gcfg        = cfg['graph']
    sigma_km    = gcfg['sigma_km']
    alpha_flow  = gcfg['alpha_flow']
    max_dist_km = gcfg['max_dist_km']
    flow_topk   = gcfg['flow_topk']

    print("\n" + "=" * 60)
    print("STEP 2: Computing loc-loc edges  (Method B)")
    print(f"  sigma={sigma_km}km  alpha={alpha_flow}  "
          f"max_dist={max_dist_km}km  top-{flow_topk * 100:.0f}%")
    print("=" * 60)

    # ── 2a. Count consecutive location transitions in training data ──
    print("  Counting location transitions in training data ...")
    mob = mob_train.sort_values(['user_id', 'date', 'time']).copy()
    mob['next_region'] = mob.groupby('user_id')['region_id'].shift(-1)
    mob['next_date']   = mob.groupby('user_id')['date'].shift(-1)
    mob['next_time']   = mob.groupby('user_id')['time'].shift(-1)

    # Within-day consecutive transitions (hour t → t+1, same date)
    same_day_next_hour = (
        (mob['next_date'] == mob['date']) &
        (mob['next_time'] == mob['time'] + 1)
    )
    # Day-boundary transitions (hour 23 → hour 0 of the next calendar day)
    # Works for datasets within a single month (no month-end crossing).
    day_boundary = (
        (mob['next_date'] == mob['date'] + 1) &
        (mob['time'] == 23) & (mob['next_time'] == 0)
    )

    moved = mob['region_id'] != mob['next_region']
    trans = mob[same_day_next_hour | day_boundary][moved].dropna(
        subset=['next_region']).copy()
    trans['next_region'] = trans['next_region'].astype(int)

    flow_df  = (trans.groupby(['region_id', 'next_region'])
                     .size()
                     .reset_index(name='flow'))
    max_flow = flow_df['flow'].max() if len(flow_df) > 0 else 1

    print(f"  Transition events  : {len(trans):,}")
    print(f"  Unique flow pairs  : {len(flow_df):,}  (max flow = {max_flow})")

    # ── 2b. Distance matrix + candidate pairs ───────────────────────
    N        = len(loc)
    print(f"  Computing pairwise distance matrix ({N}×{N}) ...")
    loc_s    = loc.sort_values('region_id').reset_index(drop=True)
    rids     = loc_s['region_id'].values
    dist_mat = haversine_matrix(loc_s['lon_center'].values, loc_s['lat_center'].values)

    si, di = np.where((dist_mat < max_dist_km) & (np.eye(N) == 0))
    print(f"  Candidate pairs within {max_dist_km}km: {len(si):,}  "
          f"(avg {len(si) / N:.0f} neighbors per region)")

    candidates = pd.DataFrame({
        'region_id':   rids[si],
        'next_region': rids[di],
        'dist_km':     dist_mat[si, di],
    })

    # ── 2c. Merge flow counts and compute combined weight ────────────
    candidates = candidates.merge(flow_df, on=['region_id', 'next_region'], how='left')
    candidates['flow']       = candidates['flow'].fillna(0).astype(int)
    candidates['flow_norm']  = candidates['flow'] / max_flow
    candidates['dist_decay'] = np.exp(-candidates['dist_km'] / sigma_km)
    candidates['weight']     = (candidates['dist_decay']
                                * (1 + alpha_flow * candidates['flow_norm']))

    n_with_flow = (candidates['flow'] > 0).sum()
    print(f"  Pairs with observed flow: {n_with_flow:,} / {len(candidates):,} "
          f"({100 * n_with_flow / len(candidates):.1f}%)")

    # ── 2d. Sparsify: keep top flow_topk fraction by weight ─────────
    threshold = np.percentile(candidates['weight'].values, (1 - flow_topk) * 100)
    sparse    = candidates[candidates['weight'] >= threshold].copy().reset_index(drop=True)

    print(f"  Weight threshold (top {flow_topk * 100:.0f}%): {threshold:.6f}")
    print(f"  Edges: {len(candidates):,} → {len(sparse):,}")
    print(f"  Distance stats (km): mean={sparse['dist_km'].mean():.2f}  "
          f"max={sparse['dist_km'].max():.2f}")

    return sparse[['region_id', 'next_region', 'flow', 'dist_km', 'weight']]


# ============================================================
# STEP 3 — BUILD AND SAVE DYNAMIC GRAPHS
# ============================================================

def build_graphs(mobility: pd.DataFrame,
                 loc_loc_edges: pd.DataFrame,
                 loc: pd.DataFrame,
                 out_dir: Path,
                 split_name: str) -> int:
    """
    For each (date, hour) in mobility, build and save a pkl graph snapshot.
    loc-loc edges are static (same for every timestep).
    user-loc edges are dynamic (reflect user positions at this hour).
    """
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Building dynamic graphs  [{split_name}]")
    print(f"{'=' * 60}")

    ll_src = loc_loc_edges['region_id'].values.astype(np.int32)
    ll_dst = loc_loc_edges['next_region'].values.astype(np.int32)
    ll_w   = loc_loc_edges['weight'].values.astype(np.float32)

    all_loc_nodes = loc['region_id'].values.astype(np.int32)

    timesteps = (mobility[['date', 'time']]
                 .drop_duplicates()
                 .sort_values(['date', 'time'])
                 .reset_index(drop=True))
    n = len(timesteps)
    print(f"  Timesteps: {n}  "
          f"(dates {timesteps['date'].min()}~{timesteps['date'].max()}, hours 0-23)")

    for i, row in timesteps.iterrows():
        date = int(row['date'])
        hour = int(row['time'])

        snap     = mobility[(mobility['date'] == date) & (mobility['time'] == hour)]
        user_ids = snap['user_id'].tolist()
        loc_ids  = snap['region_id'].values.astype(np.int32)

        graph = {
            'date':          date,
            'hour':          hour,
            'loc_nodes':     all_loc_nodes,
            'user_nodes':    user_ids,
            'loc_loc_src':   ll_src,
            'loc_loc_dst':   ll_dst,
            'loc_loc_w':     ll_w,
            'user_loc_user': user_ids,
            'user_loc_loc':  loc_ids,
            'user_loc_w':    np.ones(len(snap), dtype=np.float32),
        }

        fname = out_dir / f"dynamic_graph_{date:04d}{hour:02d}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{split_name}] {i+1:>4}/{n}  date={date:04d} h={hour:02d}  "
                  f"users={len(user_ids):>5}  loc-loc={len(ll_src):,}")

    print(f"  [OK] {n} graph files saved → {out_dir}")
    return n


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build dynamic mobility graphs from processed data.')
    parser.add_argument('--city', required=True, choices=['shanghai', 'shenzhen'],
                        help='City to process — determines which config file to load.')
    args = parser.parse_args()

    cfg = load_config(args.city)

    processed_dir = Path(cfg['paths']['processed_dir'])
    graph_dir     = processed_dir / 'dynamic_graph'
    train_dir     = graph_dir / 'train'
    test_dir      = graph_dir / 'test'

    print(f"Dynamic Graph Construction Pipeline — {cfg['city']}")
    print("=" * 60)

    for d in [graph_dir, train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    loc, mob_train, mob_test = load_data(cfg)
    loc_loc_edges = compute_loc_loc_edges(mob_train, loc, cfg)

    edge_csv = graph_dir / 'loc_loc_edges.csv'
    loc_loc_edges.to_csv(edge_csv, index=False)
    print(f"\n  [OK] loc_loc_edges.csv saved → {edge_csv}")

    n_train = build_graphs(mob_train, loc_loc_edges, loc, train_dir, 'train')
    n_test  = build_graphs(mob_test,  loc_loc_edges, loc, test_dir,  'test')

    gcfg = cfg['graph']
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  loc_loc_edges : {len(loc_loc_edges)} edges  "
          f"(sigma={gcfg['sigma_km']}km, top-{gcfg['flow_topk'] * 100:.0f}%)")
    print(f"  train graphs  : {n_train} files → {train_dir}")
    print(f"  test  graphs  : {n_test}  files → {test_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
