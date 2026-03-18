"""
03_merge_events_json_to_csv.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

EVENTS_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\events_output"
)
OUTPUT_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\merged_events.csv"
)

VERBOSE = True


# =============================================================================
# Helper functions for reading different input formats
# =============================================================================

def read_events_from_json(path: Path) -> List[Dict[str, Any]]:
    """
    Read events from a legacy accident-level JSON file.

    Expected structure:
    {
      "accident_id": "...",
      "num_events": N,
      "events": [ {event}, {event}, ... ]
    }

    Returns:
        List of event dictionaries.
        If the file is malformed, returns an empty list.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if VERBOSE:
            print(f"[WARN] Failed to parse JSON file: {path.name}")
        return []

    events = data.get("events", [])
    if not isinstance(events, list):
        if VERBOSE:
            print(f"[WARN] 'events' is not a list in {path.name}")
        return []

    # Attach accident_id defensively if missing
    acc_id = data.get("accident_id", path.stem)
    for ev in events:
        if isinstance(ev, dict) and "accident_id" not in ev:
            ev["accident_id"] = acc_id

    return [ev for ev in events if isinstance(ev, dict)]


def read_events_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read events from a JSON Lines (.jsonl) file.

    Expected structure:
      One JSON object per line, each representing ONE event.

    Returns:
        List of event dictionaries parsed line-by-line.
        Invalid lines are skipped safely.
    """
    events: List[Dict[str, Any]] = []
    acc_id = path.stem

    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue

        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            if VERBOSE:
                print(
                    f"[WARN] Invalid JSON line skipped "
                    f"(file={path.name}, line={lineno})"
                )
            continue

        if not isinstance(ev, dict):
            continue

        # Ensure accident_id exists (downstream relies on it)
        if "accident_id" not in ev:
            ev["accident_id"] = acc_id

        events.append(ev)

    return events


# =============================================================================
# Main merge logic
# =============================================================================

def main():
    """
    Main entry point.

    Iterates over all files in EVENTS_DIR, reads events from each file
    (JSON or JSONL), and merges them into a single CSV.
    """

    all_events: List[Dict[str, Any]] = []
    all_keys = set()

    if VERBOSE:
        print(f"[INFO] Reading events from: {EVENTS_DIR}")

    for path in sorted(EVENTS_DIR.iterdir()):

        if path.suffix == ".json":
            events = read_events_from_json(path)

        elif path.suffix == ".jsonl":
            events = read_events_from_jsonl(path)

        else:
            continue

        if not events:
            if VERBOSE:
                print(f"[INFO] No valid events found in {path.name}")
            continue

        all_events.extend(events)
        for ev in events:
            all_keys.update(ev.keys())

        if VERBOSE:
            print(f"[OK] {path.name}: {len(events)} events")

    if not all_events:
        print("[WARN] No events found. CSV will not be written.")
        return

    # -------------------------------------------------------------------------
    # Build DataFrame with discovered schema
    # -------------------------------------------------------------------------
    columns = sorted(all_keys)

    rows = []
    for ev in all_events:
        row = {k: ev.get(k) for k in columns}
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n[DONE] Merged {len(df)} events → {OUTPUT_CSV}")
    print("[DONE] Program 03 completed successfully")


# =============================================================================
# Script entry
# =============================================================================

if __name__ == "__main__":
    main()
