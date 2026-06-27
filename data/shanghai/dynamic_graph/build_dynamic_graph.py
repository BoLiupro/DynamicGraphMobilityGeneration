#!/usr/bin/env python3
"""
Dynamic Graph Construction for Human Mobility Generation

Graph structure at each timestep (date, hour):
  Nodes:
    - Location nodes : all 400 grid regions (static set)
    - User nodes     : users present in this split (1010 train / 253 test)

  Edges:
    - loc-loc : static edges derived from training mobility transitions.
                weight(i→j) = flow_norm(i→j) × exp(-dist(i,j) / SIGMA_KM)
                Sparsified by keeping the top FLOW_TOPK fraction by weight.
    - user-loc: dynamic edges — user u → region r if u is at r at this timestep.
                weight = 1.

Output files:
  dyanmic_graph/
    loc_loc_edges.csv          (edge table for inspection)
    train/dynamic_graph_{mmdd}{hh}.pkl
    test/ dynamic_graph_{mmdd}{hh}.pkl

Each pkl file is a dict:
  {
    'date'          : int  (e.g. 601 for June 1)
    'hour'          : int  (0-23)
    'loc_nodes'     : np.int32[400]        all region IDs
    'user_nodes'    : list[str]            user IDs active at this timestep
    'loc_loc_src'   : np.int32[E_ll]       loc-loc source region IDs
    'loc_loc_dst'   : np.int32[E_ll]       loc-loc dest   region IDs
    'loc_loc_w'     : np.float32[E_ll]     loc-loc weights
    'user_loc_user' : list[str]            user IDs   (parallel to user_loc_loc)
    'user_loc_loc'  : np.int32[E_ul]       region IDs (one per user)
    'user_loc_w'    : np.float32[E_ul]     all 1.0
  }
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR     = Path('/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration')
DATA_DIR        = PROJECT_DIR / 'data' / 'shanghai'
GRAPH_DIR       = DATA_DIR / 'dyanmic_graph'       # keep original folder name
TRAIN_GRAPH_DIR = GRAPH_DIR / 'train'
TEST_GRAPH_DIR  = GRAPH_DIR / 'test'

# ============================================================
# HYPERPARAMETERS
# ============================================================

SIGMA_KM    = 3.0   # distance decay scale: exp(-d / sigma); ~3 km ≈ 3 grid-cell widths
ALPHA_FLOW  = 1.0   # flow bonus: w = exp(-d/σ) × (1 + α × flow_norm), α=0 → pure distance
MAX_DIST_KM = 10.0  # candidate cutoff: only consider region pairs within this distance
FLOW_TOPK   = 0.20  # keep the top 20% of candidate edges by combined weight


# ============================================================
# HAVERSINE DISTANCE  (vectorized)
# ============================================================

def haversine_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Return N×N great-circle distance matrix in km."""
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

    print(f"  location.csv   : {len(loc)} regions, columns={loc.columns.tolist()[:4]}...")
    print(f"  mobility_train : {len(mob_train):,} rows | "
          f"{mob_train['user_id'].nunique()} users | "
          f"dates {mob_train['date'].min()}~{mob_train['date'].max()}")
    print(f"  mobility_test  : {len(mob_test):,} rows | "
          f"{mob_test['user_id'].nunique()} users | "
          f"dates {mob_test['date'].min()}~{mob_test['date'].max()}")
    return loc, mob_train, mob_test


# ============================================================
# STEP 2 — LOC-LOC EDGES  (derived from training mobility only)
# ============================================================

def compute_loc_loc_edges(mob_train: pd.DataFrame,
                          loc: pd.DataFrame) -> pd.DataFrame:
    """
    Build the static loc-loc edge table using Method B:

      Candidate generation:
        All ordered region pairs (i, j) with dist(i,j) < MAX_DIST_KM.
        This gives O(400 × K_neighbors) candidates, much denser than
        the flow-only approach.

      Weight formula:
        w(i→j) = exp(-dist(i,j) / SIGMA_KM) × (1 + ALPHA_FLOW × flow_norm(i→j))
        - Base term  exp(-dist/σ)     : purely distance-driven; all nearby pairs
          get a non-zero weight even if no user ever moved between them.
        - Bonus term (1 + α·flow_norm): pairs with high observed flow get
          an additional multiplicative boost (up to ×2 when α=1).
        - flow_norm = flow / max_flow  ∈ [0, 1]; 0 for pairs with no transitions.

      Sparsification:
        Keep edges with weight ≥ percentile(1 − FLOW_TOPK) to retain the top
        FLOW_TOPK fraction by weight.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Computing loc-loc edges  (Method B: distance-base + flow bonus)")
    print("=" * 60)

    # ── 2a. Count transitions i → j from training mobility ───
    print("  Counting consecutive location transitions in training data ...")

    mob = mob_train.sort_values(['user_id', 'date', 'time']).copy()
    mob['next_region'] = mob.groupby('user_id')['region_id'].shift(-1)
    mob['next_date']   = mob.groupby('user_id')['date'].shift(-1)
    mob['next_time']   = mob.groupby('user_id')['time'].shift(-1)

    same_day_next_hour = (
        (mob['date'] == mob['next_date']) &
        (mob['next_time'] == mob['time'] + 1)
    )
    day_boundary = (
        (mob['next_date'] == mob['date'] + 1) &
        (mob['time'] == 23) & (mob['next_time'] == 0)
    )
    moved = mob['region_id'] != mob['next_region']
    trans = mob[same_day_next_hour | day_boundary][moved].dropna(subset=['next_region']).copy()
    trans['next_region'] = trans['next_region'].astype(int)

    flow_df = (trans.groupby(['region_id', 'next_region'])
                    .size()
                    .reset_index(name='flow'))
    max_flow = flow_df['flow'].max() if len(flow_df) > 0 else 1

    print(f"  Transition events:      {len(trans):,}")
    print(f"  Unique flow pairs:      {len(flow_df):,}  (max flow = {max_flow})")

    # ── 2b. Distance matrix + candidate pairs ────────────────
    print(f"  Computing pairwise distance matrix (400×400) ...")
    loc_s   = loc.sort_values('region_id').reset_index(drop=True)
    rids    = loc_s['region_id'].values
    dist_mat = haversine_matrix(loc_s['lon_center'].values, loc_s['lat_center'].values)

    # All (i→j) pairs within MAX_DIST_KM, excluding self-loops
    N = len(loc_s)
    si, di = np.where((dist_mat < MAX_DIST_KM) & (np.eye(N) == 0))
    print(f"  Candidate pairs within {MAX_DIST_KM} km: {len(si):,}  "
          f"(avg {len(si)/N:.0f} neighbors per region)")

    candidates = pd.DataFrame({
        'region_id':   rids[si],
        'next_region': rids[di],
        'dist_km':     dist_mat[si, di],
    })

    # ── 2c. Merge flow counts (0 for pairs with no transitions) ─
    candidates = candidates.merge(flow_df, on=['region_id', 'next_region'], how='left')
    candidates['flow'] = candidates['flow'].fillna(0).astype(int)

    # ── 2d. Combined weight ───────────────────────────────────
    # w = exp(-dist/σ) × (1 + α × flow_norm)
    candidates['flow_norm']  = candidates['flow'] / max_flow
    candidates['dist_decay'] = np.exp(-candidates['dist_km'] / SIGMA_KM)
    candidates['weight']     = (candidates['dist_decay']
                                * (1 + ALPHA_FLOW * candidates['flow_norm']))

    n_with_flow = (candidates['flow'] > 0).sum()
    print(f"  Pairs with observed flow: {n_with_flow:,} / {len(candidates):,} "
          f"({100*n_with_flow/len(candidates):.1f}%)")

    # ── 2e. Sparsify ──────────────────────────────────────────
    threshold = np.percentile(candidates['weight'].values, (1 - FLOW_TOPK) * 100)
    sparse    = candidates[candidates['weight'] >= threshold].copy().reset_index(drop=True)

    print(f"  Weight threshold (top {FLOW_TOPK*100:.0f}%): {threshold:.6f}")
    print(f"  Edges before → after sparsification: {len(candidates):,} → {len(sparse):,}")
    print(f"  Distance stats (km): mean={sparse['dist_km'].mean():.2f}  "
          f"max={sparse['dist_km'].max():.2f}")
    print(f"  Weight  stats:       mean={sparse['weight'].mean():.4f}  "
          f"max={sparse['weight'].max():.4f}")

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
    For each (date, hour) in mobility, build a graph and save as pkl.

    The loc-loc edges are static (same for every timestep).
    The user-loc edges are dynamic (reflect user positions at this hour).
    """
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Building dynamic graphs  [{split_name}]")
    print(f"{'=' * 60}")
    print(f"  Output directory: {out_dir}")

    # Static arrays — reused for every graph (no copy needed, pkl uses refs)
    ll_src = loc_loc_edges['region_id'].values.astype(np.int32)
    ll_dst = loc_loc_edges['next_region'].values.astype(np.int32)
    ll_w   = loc_loc_edges['weight'].values.astype(np.float32)

    # All 400 location nodes are always present (fixed vocabulary)
    all_loc_nodes = loc['region_id'].values.astype(np.int32)

    # All (date, hour) pairs in sorted order
    timesteps = (mobility[['date', 'time']]
                 .drop_duplicates()
                 .sort_values(['date', 'time'])
                 .reset_index(drop=True))
    n = len(timesteps)
    print(f"  Timesteps to process: {n}  "
          f"(dates {timesteps['date'].min()}~{timesteps['date'].max()}, "
          f"hours 0-23)")

    for i, row in timesteps.iterrows():
        date = int(row['date'])
        hour = int(row['time'])

        # Snapshot: all users and their location at (date, hour)
        snap = mobility[(mobility['date'] == date) & (mobility['time'] == hour)]

        user_ids  = snap['user_id'].tolist()
        loc_ids   = snap['region_id'].values.astype(np.int32)
        ul_w      = np.ones(len(snap), dtype=np.float32)

        graph = {
            'date':           date,
            'hour':           hour,
            # ── Node sets ──────────────────────────────────────
            'loc_nodes':      all_loc_nodes,    # shape (400,)
            'user_nodes':     user_ids,         # list of user_id strings
            # ── Loc-loc edges (static, directed COO) ───────────
            'loc_loc_src':    ll_src,           # shape (E_ll,)
            'loc_loc_dst':    ll_dst,
            'loc_loc_w':      ll_w,
            # ── User-loc edges (dynamic) ────────────────────────
            'user_loc_user':  user_ids,         # user_id for each edge
            'user_loc_loc':   loc_ids,          # region_id for each edge
            'user_loc_w':     ul_w,             # all 1.0
        }

        fname = out_dir / f"dynamic_graph_{date:04d}{hour:02d}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Progress print every 50 steps and at the end
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{split_name}] {i+1:>4}/{n}  "
                  f"date={date:04d} h={hour:02d}  "
                  f"users={len(user_ids):>4}  "
                  f"loc-loc={len(ll_src):,}  user-loc={len(user_ids):>4}")

    print(f"  [OK] {n} graph files saved to {out_dir}")
    return n


# ============================================================
# MAIN
# ============================================================

def main():
    print("Dynamic Graph Construction Pipeline")
    print("=" * 60)

    for d in [GRAPH_DIR, TRAIN_GRAPH_DIR, TEST_GRAPH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    loc, mob_train, mob_test = load_data()

    # Loc-loc edges derived solely from training mobility
    loc_loc_edges = compute_loc_loc_edges(mob_train, loc)

    # Save edge table for inspection / debugging
    edge_csv = GRAPH_DIR / 'loc_loc_edges.csv'
    loc_loc_edges.to_csv(edge_csv, index=False)
    print(f"\n  [OK] loc_loc_edges.csv saved → {edge_csv}")

    # Build and save all dynamic graphs
    n_train = build_graphs(mob_train, loc_loc_edges, loc, TRAIN_GRAPH_DIR, 'train')
    n_test  = build_graphs(mob_test,  loc_loc_edges, loc, TEST_GRAPH_DIR,  'test')

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  loc_loc_edges : {len(loc_loc_edges)} edges  (sigma={SIGMA_KM}km, top-{FLOW_TOPK*100:.0f}%)")
    print(f"  train graphs  : {n_train} files → {TRAIN_GRAPH_DIR}")
    print(f"  test  graphs  : {n_test}  files → {TEST_GRAPH_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
