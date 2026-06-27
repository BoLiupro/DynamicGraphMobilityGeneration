#!/usr/bin/env python3
"""
Human Mobility Generation Framework — Main Entry Point
=======================================================
Runs the user agent initialization pipeline for a given city.

Usage:
    cd DynamicGraphMobilityGeneration
    python3 main.py --city shanghai
    python3 main.py --city shenzhen --seed 123 --co-threshold 0.65

Pipeline steps:
  1. load_priors         — load pattern files (community, motifs, flow)
  2. assign_communities  — assign community to each test user via P(comm|start_loc)
  3. generate_plans      — generate 24h daily plan per user (Markov chain)
  4. find_co_mobility    — find users with similar plans (Jaccard >= threshold)
  5. save_output         — write JSON to output/user_init/

Output:
  output/user_init/{city}_user_profiles.json
  output/user_init/{city}_summary.json
"""

import argparse
import sys
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from util.common import load_config
from model.user_agent import build_user_init_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Human Mobility Generation — User Agent Initialization"
    )
    parser.add_argument(
        "--city",
        choices=["shanghai", "shenzhen"],
        default="shanghai",
        help="City to process (default: shanghai)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for plan generation (default: 42)",
    )
    parser.add_argument(
        "--co-threshold",
        type=float,
        default=0.60,
        dest="co_threshold",
        help="Jaccard similarity threshold for co-mobility detection (default: 0.60)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Human Mobility Generation — User Agent Initialization")
    print("=" * 60)
    print(f"  city           : {args.city}")
    print(f"  seed           : {args.seed}")
    print(f"  co-threshold   : {args.co_threshold}")

    # ── Load city config ─────────────────────────────────────────────
    cfg = load_config(args.city)

    # ── Build LangGraph app ───────────────────────────────────────────
    app = build_user_init_graph()

    # ── Initial state ─────────────────────────────────────────────────
    initial_state = {
        "city":                  args.city,
        "cfg":                   cfg,
        "seed":                  args.seed,
        "co_mobility_threshold": args.co_threshold,
    }

    # ── Run pipeline ──────────────────────────────────────────────────
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    if final_state.get("error"):
        print(f"[ERROR] {final_state['error']}")
        sys.exit(1)

    profiles = final_state.get("user_profiles", [])
    print(f"  Initialized {len(profiles)} user agents")
    print(f"  Output: output/user_init/{args.city}_user_profiles.json")


if __name__ == "__main__":
    main()
