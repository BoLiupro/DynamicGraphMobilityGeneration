#!/usr/bin/env python3
"""
Shared utilities for the mobility pipeline.

Exports:
    load_config(city)                -- load configs/{city}.yaml
    get_project_dir()                -- DynamicGraphMobilityGeneration root path
    haversine_matrix(lons, lats)     -- N×N distance matrix (km)
    haversine(lon1,lat1,lon2,lat2)   -- single pair distance (km)
    dist_bin(d_km, cfg)              -- bin distance using config's dist_bins/labels
    stay_bin(h, cfg)                 -- bin hours using config's stay_bins/labels
    build_loc_lookup(loc_df, cfg)    -- poi_map and coord_map dicts
    spatial_gravity_search(...)      -- top-k candidates by gravity model
"""

import numpy as np
import yaml
from pathlib import Path


def get_project_dir() -> Path:
    return Path(__file__).parent.parent   # util/../ = DynamicGraphMobilityGeneration/


def load_config(city: str) -> dict:
    path = get_project_dir() / 'configs' / f'{city}.yaml'
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


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


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two (lon, lat) points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dist_bin(d_km: float, cfg: dict) -> str:
    """Bin distance value into a label string using pattern config."""
    bins   = cfg['pattern']['dist_bins']   # e.g. [0, 1, 2, 3, 5]
    labels = cfg['pattern']['dist_labels'] # e.g. ['0-1km', ..., '>5km']
    for i, upper in enumerate(bins[1:]):
        if d_km <= upper:
            return labels[i]
    return labels[-1]


def stay_bin(h: int, cfg: dict) -> str:
    """Bin stay duration (hours) into a label string using pattern config."""
    bins   = cfg['pattern']['stay_bins']
    labels = cfg['pattern']['stay_labels']
    for i, upper in enumerate(bins[1:]):
        if h <= upper:
            return labels[i]
    return labels[-1]


def spatial_gravity_search(
    current_loc: int,
    dist_label: str,
    purpose: str,
    coord_map: dict,
    poi_map: dict,
    pop_map: dict,
    cfg: dict,
) -> list:
    """
    Gravity-model candidate search for a user about to move.

    Args:
        current_loc : origin location id
        dist_label  : intended travel distance label (e.g. "1-2km"), from the
                      daily plan MOVE segment; used to filter candidates to the
                      matching distance range in cfg['pattern']['dist_bins/labels']
        purpose     : target POI type from the daily plan MOVE segment
        coord_map   : {int(loc_id): [lon, lat]}
        poi_map     : {int(loc_id): str POI type}
        pop_map     : {int(loc_id): float mean population}
        cfg         : full city config dict

    Returns:
        List of up to cfg['gravity']['top_k'] dicts, sorted by gravity score:
          {loc_id, poi_type, dist_km, gravity_score}
        POI-matching candidates receive a boost multiplier.

    Gravity formula:  score = pop^alpha / max(dist, min_dist)^beta
    """
    gcfg     = cfg.get("gravity", {})
    alpha    = gcfg.get("alpha",       1.0)
    beta     = gcfg.get("beta",        2.0)
    top_k    = gcfg.get("top_k",       5)
    poi_boost = gcfg.get("poi_boost",  2.0)
    min_dist = gcfg.get("min_dist_km", 0.1)

    # Resolve distance range from label
    d_labels = cfg["pattern"]["dist_labels"]
    d_bins   = cfg["pattern"]["dist_bins"]
    if dist_label in d_labels:
        idx = d_labels.index(dist_label)
        lo  = d_bins[idx]
        hi  = d_bins[idx + 1] if idx + 1 < len(d_bins) else float("inf")
    else:
        lo, hi = 0.0, float("inf")

    src = coord_map.get(current_loc)
    if src is None:
        return []

    results = []
    for loc_id, lonlat in coord_map.items():
        if loc_id == current_loc:
            continue
        d_km = haversine(src[0], src[1], lonlat[0], lonlat[1])
        if d_km < lo or d_km > hi:
            continue

        pop   = pop_map.get(loc_id, 1.0)
        score = (pop ** alpha) / max(d_km, min_dist) ** beta

        # Boost score when candidate POI matches the intended purpose
        loc_poi = poi_map.get(loc_id, "Unknown")
        if purpose and loc_poi == purpose:
            score *= poi_boost

        results.append({
            "loc_id":        loc_id,
            "poi_type":      loc_poi,
            "dist_km":       round(d_km, 2),
            "gravity_score": round(score, 4),
        })

    results.sort(key=lambda x: -x["gravity_score"])
    return results[:top_k]


def build_loc_lookup(loc_df, cfg: dict) -> tuple:
    """
    Build poi_map (region_id → dominant POI category label)
    and coord_map (region_id → (lon, lat)) from location.csv.
    """
    poi_map, coord_map = {}, {}
    poi_cats = cfg['poi']['categories']
    cat_cols = [c for c in poi_cats if c in loc_df.columns]
    for _, row in loc_df.iterrows():
        rid = int(row['region_id'])
        coord_map[rid] = (float(row['lon_center']), float(row['lat_center']))
        if cat_cols:
            vals = row[cat_cols]
            poi_map[rid] = vals.idxmax() if vals.max() > 0 else 'Unknown'
        else:
            poi_map[rid] = 'Unknown'
    return poi_map, coord_map
