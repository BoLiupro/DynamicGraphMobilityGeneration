#!/usr/bin/env python3
"""
Dynamic Graph Construction for Shenzhen Private Car Mobility

Identical structure to Shanghai's version; adapted for:
  - 500 location nodes  (25 × 20 grid, 113.90~114.15E, 22.50~22.70N)
  - Shenzhen data paths

Graph structure at each timestep (date, hour):
  Nodes:
    - Location nodes : all 500 grid regions (static)
    - User nodes     : vehicles present in this split

  Edges:
    - loc-loc : static, distance-decay × flow-bonus (Method B)
                w(i→j) = exp(-dist/SIGMA_KM) × (1 + ALPHA_FLOW × flow_norm)
                Sparsified: keep top FLOW_TOPK fraction by weight
    - user-loc: dynamic, weight = 1.0

Output files:
  dyanmic_graph/
    loc_loc_edges.csv
    train/dynamic_graph_{mmdd}{hh}.pkl
    test/ dynamic_graph_{mmdd}{hh}.pkl

Each pkl dict structure is identical to Shanghai's.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR     = Path('/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration')
DATA_DIR        = PROJECT_DIR / 'data' / 'shenzhen'
GRAPH_DIR       = DATA_DIR / 'dyanmic_graph'
TRAIN_GRAPH_DIR = GRAPH_DIR / 'train'
TEST_GRAPH_DIR  = GRAPH_DIR / 'test'

# ============================================================
# HYPERPARAMETERS  (same as Shanghai)
# ============================================================

SIGMA_KM    = 3.0    # distance decay: exp(-d / sigma)
ALPHA_FLOW  = 1.0    # flow bonus weight
MAX_DIST_KM = 10.0   # candidate edge distance cutoff
FLOW_TOPK   = 0.20   # keep top 20% edges by combined weight


# ============================================================
# HAVERSINE DISTANCE  (vectorized N×N)
# ============================================================

def haversine_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    R = 6371.0
    lons_r = np.radians(lons)
    lats_r = np.radians(lats)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    loc       = pd.read_csv(DATA_DIR / 'location.csv')
    mob_train = pd.read_csv(DATA_DIR / 'mobility_train.csv')
    mob_test  = pd.read_csv(DATA_DIR / 'mobility_test.csv')

    print(f"  location.csv   : {len(loc)} regions")
    print(f"  mobility_train : {len(mob_train):,} rows | "
          f"{mob_train['user_id'].nunique()} vehicles | "
          f"dates {mob_train['date'].min()}~{mob_train['date'].max()}")
    print(f"  mobility_test  : {len(mob_test):,} rows | "
          f"{mob_test['user_id'].nunique()} vehicles | "
          f"dates {mob_test['date'].min()}~{mob_test['date'].max()}")
    return loc, mob_train, mob_test


# ============================================================
# STEP 2 — LOC-LOC EDGES  (derived from training mobility only)
# ============================================================

def compute_loc_loc_edges(mob_train: pd.DataFrame,
                          loc: pd.DataFrame) -> pd.DataFrame:
    """
    Method B: distance-base + flow-bonus.
    w(i→j) = exp(-dist/SIGMA_KM) × (1 + ALPHA_FLOW × flow_norm)
    Candidate pairs: all (i,j) with dist < MAX_DIST_KM.
    Sparsify: keep top FLOW_TOPK fraction by weight.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Computing loc-loc edges  (Method B)")
    print("=" * 60)

    # ── 2a. Count transitions from training mobility ────────────────
    mob = mob_train.sort_values(['user_id', 'date', 'time']).copy()
    mob['next_region'] = mob.groupby('user_id')['region_id'].shift(-1)
    mob['next_date']   = mob.groupby('user_id')['date'].shift(-1)
    mob['next_time']   = mob.groupby('user_id')['time'].shift(-1)

    same_day_next_hour = (
        (mob['date'] == mob['next_date']) &
        (mob['next_time'] == mob['time'] + 1)
    )
    day_boundary = (
        (mob['next_date'].astype(str) ==
         (pd.to_datetime(mob['date'].astype(str), format='%m%d') +
          pd.Timedelta(days=1)).dt.strftime('%m%d')) &
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
    print(f"  Transition events:  {len(trans):,}")
    print(f"  Unique flow pairs:  {len(flow_df):,}  (max flow = {max_flow})")

    # ── 2b. Distance matrix ─────────────────────────────────────────
    print(f"  Computing pairwise distance matrix ({len(loc)}×{len(loc)}) ...")
    loc_s    = loc.sort_values('region_id').reset_index(drop=True)
    rids     = loc_s['region_id'].values
    dist_mat = haversine_matrix(loc_s['lon_center'].values, loc_s['lat_center'].values)

    N = len(loc_s)
    si, di = np.where((dist_mat < MAX_DIST_KM) & (np.eye(N) == 0))
    print(f"  Candidate pairs within {MAX_DIST_KM} km: {len(si):,}  "
          f"(avg {len(si)/N:.0f} neighbors per region)")

    candidates = pd.DataFrame({
        'region_id':   rids[si],
        'next_region': rids[di],
        'dist_km':     dist_mat[si, di],
    })

    # ── 2c. Merge flow + compute weight ─────────────────────────────
    candidates = candidates.merge(flow_df, on=['region_id', 'next_region'], how='left')
    candidates['flow']      = candidates['flow'].fillna(0).astype(int)
    candidates['flow_norm'] = candidates['flow'] / max_flow
    candidates['dist_decay'] = np.exp(-candidates['dist_km'] / SIGMA_KM)
    candidates['weight']    = (candidates['dist_decay']
                               * (1 + ALPHA_FLOW * candidates['flow_norm']))

    # ── 2d. Sparsify ─────────────────────────────────────────────────
    threshold = np.percentile(candidates['weight'].values, (1 - FLOW_TOPK) * 100)
    sparse    = candidates[candidates['weight'] >= threshold].copy().reset_index(drop=True)

    print(f"  Edges {len(candidates):,} → {len(sparse):,}  "
          f"(top {FLOW_TOPK*100:.0f}%, threshold={threshold:.6f})")
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
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Building dynamic graphs  [{split_name}]")
    print(f"{'=' * 60}")
    print(f"  Output directory: {out_dir}")

    ll_src = loc_loc_edges['region_id'].values.astype(np.int32)
    ll_dst = loc_loc_edges['next_region'].values.astype(np.int32)
    ll_w   = loc_loc_edges['weight'].values.astype(np.float32)

    all_loc_nodes = loc['region_id'].values.astype(np.int32)   # 500 nodes

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
        ul_w     = np.ones(len(snap), dtype=np.float32)

        graph = {
            'date':           date,
            'hour':           hour,
            'loc_nodes':      all_loc_nodes,
            'user_nodes':     user_ids,
            'loc_loc_src':    ll_src,
            'loc_loc_dst':    ll_dst,
            'loc_loc_w':      ll_w,
            'user_loc_user':  user_ids,
            'user_loc_loc':   loc_ids,
            'user_loc_w':     ul_w,
        }

        fname = out_dir / f"dynamic_graph_{date:04d}{hour:02d}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{split_name}] {i+1:>4}/{n}  date={date:04d} h={hour:02d}  "
                  f"vehicles={len(user_ids):>5}  "
                  f"loc-loc={len(ll_src):,}  user-loc={len(user_ids):>5}")

    print(f"  [OK] {n} graph files saved → {out_dir}")
    return n


# ============================================================
# MAIN
# ============================================================

def main():
    print("Dynamic Graph Construction Pipeline — Shenzhen")
    print("=" * 60)

    for d in [GRAPH_DIR, TRAIN_GRAPH_DIR, TEST_GRAPH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    loc, mob_train, mob_test = load_data()

    loc_loc_edges = compute_loc_loc_edges(mob_train, loc)

    edge_csv = GRAPH_DIR / 'loc_loc_edges.csv'
    loc_loc_edges.to_csv(edge_csv, index=False)
    print(f"\n  [OK] loc_loc_edges.csv saved → {edge_csv}")

    n_train = build_graphs(mob_train, loc_loc_edges, loc, TRAIN_GRAPH_DIR, 'train')
    n_test  = build_graphs(mob_test,  loc_loc_edges, loc, TEST_GRAPH_DIR,  'test')

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  loc_loc_edges : {len(loc_loc_edges)} edges")
    print(f"  train graphs  : {n_train} files → {TRAIN_GRAPH_DIR}")
    print(f"  test  graphs  : {n_test}  files → {TEST_GRAPH_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
