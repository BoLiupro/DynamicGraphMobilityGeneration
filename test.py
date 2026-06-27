#!/usr/bin/env python3
"""
Single-User Inference Test
============================
Tests one timestep of LLM-based mobility decision for a single user.
Uses the pre-computed user profiles from the initialization phase.

Usage:
    cd DynamicGraphMobilityGeneration
    python3 test.py --city shanghai
    python3 test.py --city shenzhen --user-idx 5 --hour 17
    python3 test.py --city shanghai --user-idx 0 --hour auto   # first MOVE hour
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from util.common import load_config
from model.user_agent import build_user_inference_graph
from model.user_init import node_load_priors


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_user_profiles(city: str) -> list:
    """Load pre-computed user profiles from the init phase."""
    path = Path(__file__).parent / "output" / "user_init" / f"{city}_user_profiles.json"
    if not path.exists():
        raise FileNotFoundError(
            f"User profiles not found: {path}\n"
            f"Run 'python3 main.py --city {city}' first."
        )
    with open(path) as f:
        return json.load(f)


def find_users_with_moves(profiles: list) -> list:
    """Return indices of users that have at least one MOVE in their plan."""
    return [i for i, p in enumerate(profiles) if p["n_moves"] >= 1]


def find_first_move_hour(plan: list) -> int:
    """Return the start hour of the first MOVE_AB segment, or 8 if none."""
    for seg in plan:
        if seg["motif_type"] == "MOVE_AB":
            return seg["start_hour"]
    return 8


def print_plan(plan: list, highlight_hour: int):
    """Pretty-print a 24h daily plan, highlighting the active segment."""
    print("  Daily plan:")
    for seg in plan:
        active = " ◀" if seg["start_hour"] <= highlight_hour < seg["end_hour"] else "  "
        if seg["motif_type"] == "STAY":
            print(f"    h{seg['start_hour']:02d}-{seg['end_hour']:02d}  STAY    "
                  f"poi={seg['poi_type']}{active}")
        else:
            print(f"    h{seg['start_hour']:02d}-{seg['end_hour']:02d}  MOVE_AB "
                  f"{seg['from_poi']} → {seg['to_poi']} ({seg['dist_label']}){active}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Single-user inference test")
    p.add_argument("--city",     choices=["shanghai", "shenzhen"], default="shanghai")
    p.add_argument("--user-idx", type=int, default=None,
                   help="Index into user profiles list (default: first user with moves)")
    p.add_argument("--hour",     type=int, default=None,
                   help="Simulate from this hour (default: first MOVE hour of selected user)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("User Agent — Single-Step Inference Test")
    print("=" * 60)

    cfg = load_config(args.city)

    # ── Select user ───────────────────────────────────────────────────────────
    profiles   = load_user_profiles(args.city)
    movers     = find_users_with_moves(profiles)

    if args.user_idx is not None:
        user_idx = args.user_idx
    elif movers:
        user_idx = movers[0]
        print(f"  (no --user-idx given; using first user with moves: idx={user_idx})")
    else:
        user_idx = 0
        print("  WARNING: no users with moves found; using idx=0 (will test STAY)")

    if user_idx >= len(profiles):
        print(f"  ERROR: --user-idx {user_idx} out of range (total={len(profiles)})")
        sys.exit(1)

    profile = profiles[user_idx]

    # ── Select test hour ──────────────────────────────────────────────────────
    test_hour = args.hour if args.hour is not None else find_first_move_hour(profile["plan"])

    # ── Print user info ───────────────────────────────────────────────────────
    print(f"\n  city       : {args.city}")
    print(f"  user_idx   : {user_idx}")
    print(f"  user_id    : {profile['user_id'][:16]}...")
    print(f"  community  : {profile['assigned_community']} "
          f"({profile['community_poi_profile']})")
    print(f"  n_moves    : {profile['n_moves']}")
    print(f"  test_hour  : {test_hour:02d}:00\n")
    print_plan(profile["plan"], test_hour)

    # ── Load reference data (re-uses node_load_priors) ────────────────────────
    print("\nLoading reference data ...")
    priors = node_load_priors({"city": args.city, "cfg": cfg, "seed": 42})
    print(f"  coord_map: {len(priors['coord_map'])} locations")
    print(f"  pop_map  : {len(priors['pop_map'])} locations")
    print(f"  flow edges: {sum(len(v) for v in priors['flow_from'].values())}")

    # ── Run inference graph ───────────────────────────────────────────────────
    app = build_user_inference_graph()

    initial_state = {
        "city":             args.city,
        "cfg":              cfg,
        "user_profile":     profile,
        "current_location": profile["start_location"],
        "current_hour":     test_hour,
        "coord_map":        priors["coord_map"],
        "poi_map":          priors["poi_map"],
        "pop_map":          priors["pop_map"],
        "flow_from":        priors["flow_from"],
    }

    print("\nRunning inference graph ...")
    print("-" * 60)
    result = app.invoke(initial_state)
    print("-" * 60)

    # ── Print result ──────────────────────────────────────────────────────────
    print("\nResult:")
    print(f"  action        : {result.get('action')}")
    print(f"  next_location : {result.get('next_location')} "
          f"(poi={priors['poi_map'].get(result.get('next_location'), 'Unknown')})")
    print(f"  next_hour     : {result.get('next_hour')}")
    if result.get("decision"):
        d = result["decision"]
        print(f"  parse_method  : {d.get('parse_method', '-')}")
        print(f"  reason        : {d.get('reason', '-')}")
    if result.get("error"):
        print(f"  ERROR         : {result['error']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
