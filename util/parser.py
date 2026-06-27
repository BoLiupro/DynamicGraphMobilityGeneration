#!/usr/bin/env python3
"""
Parsers for LLM output strings.

parse_location_decision: extract {next_location_id, reason} from raw LLM text.
"""

import json
import re
from typing import Any, Dict, List, Optional


def parse_location_decision(
    raw: str,
    candidates: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Parse the LLM's move-decision output into a structured dict.

    Expected LLM format:
        {"next_location_id": <int>, "reason": "<string>"}

    Fallback strategy (in order):
        1. Direct JSON parse of the full string
        2. Regex extraction of the first JSON block containing next_location_id
        3. Regex extraction of just the integer value after "next_location_id"
        4. Return the top gravity candidate (if provided)
        5. Return sentinel {next_location_id: -1}

    Args:
        raw        : raw string returned by the LLM
        candidates : gravity-search candidates list (used as fallback)

    Returns:
        {"next_location_id": int, "reason": str, "parse_method": str}
    """
    raw = (raw or "").strip()

    # --- Strategy 1: direct JSON parse ---
    try:
        obj = json.loads(raw)
        if "next_location_id" in obj:
            return {
                "next_location_id": int(obj["next_location_id"]),
                "reason":           str(obj.get("reason", "")),
                "parse_method":     "direct_json",
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Strategy 2: extract first {...} block that contains the key ---
    match = re.search(r'\{[^{}]*"next_location_id"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return {
                "next_location_id": int(obj["next_location_id"]),
                "reason":           str(obj.get("reason", "")),
                "parse_method":     "regex_json_block",
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Strategy 3: just extract the integer value ---
    match = re.search(r'"next_location_id"\s*:\s*(\d+)', raw)
    if match:
        return {
            "next_location_id": int(match.group(1)),
            "reason":           "extracted from partial response",
            "parse_method":     "regex_int",
        }

    # --- Strategy 4: fallback to top gravity candidate ---
    if candidates:
        return {
            "next_location_id": candidates[0]["loc_id"],
            "reason":           "fallback: top gravity candidate (LLM parse failed)",
            "parse_method":     "gravity_fallback",
        }

    # --- Strategy 5: sentinel ---
    return {
        "next_location_id": -1,
        "reason":           "parse failed, no candidates available",
        "parse_method":     "failed",
    }
