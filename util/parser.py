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


def parse_user_selection(
    raw: str,
    valid_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Parse the location agent's LLM output into a structured dict.

    Expected LLM format:
        {"users_to_attract": ["<uid>", ...], "reasoning": "<string>"}

    Fallback strategy (in order):
        1. Direct JSON parse, filter user_ids against valid_ids
        2. Regex extraction of a JSON block containing users_to_attract
        3. Regex extraction of just the list [...] after users_to_attract
        4. Safe empty fallback: users_to_attract = []

    Args:
        raw       : raw LLM output string
        valid_ids : set of known-valid user_id strings; invalid ids are dropped

    Returns:
        {"users_to_attract": list[str], "reasoning": str, "parse_method": str}
    """
    raw = (raw or "").strip()

    def _filter(ids: list) -> list:
        if valid_ids is None:
            return [str(i) for i in ids]
        return [str(i) for i in ids if str(i) in valid_ids]

    # --- Strategy 1: direct JSON parse ---
    try:
        obj = json.loads(raw)
        if "users_to_attract" in obj:
            return {
                "users_to_attract": _filter(obj["users_to_attract"]),
                "reasoning":        str(obj.get("reasoning", "")),
                "parse_method":     "direct_json",
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Strategy 2: extract first {...} block containing the key ---
    match = re.search(r'\{[^{}]*"users_to_attract"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return {
                "users_to_attract": _filter(obj["users_to_attract"]),
                "reasoning":        str(obj.get("reasoning", "")),
                "parse_method":     "regex_json_block",
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Strategy 3: extract the list value ---
    match = re.search(r'"users_to_attract"\s*:\s*(\[[^\]]*\])', raw, re.DOTALL)
    if match:
        try:
            ids = json.loads(match.group(1))
            return {
                "users_to_attract": _filter(ids),
                "reasoning":        "extracted list from partial response",
                "parse_method":     "regex_list",
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Strategy 4: safe empty fallback ---
    return {
        "users_to_attract": [],
        "reasoning":        "parse failed — no users attracted",
        "parse_method":     "failed",
    }
