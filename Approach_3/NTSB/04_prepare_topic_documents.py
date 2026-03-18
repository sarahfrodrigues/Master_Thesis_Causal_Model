"""
04_prepare_topic_documents.py

"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Input directory produced by Program 02
EVENTS_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\events_output"
)

# Output CSV for BERTopic
OUT_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_docs.csv"
)

# Minimum length for evidence_text to be considered meaningful
MIN_TEXT_LEN = 30

VERBOSE = True


# =============================================================================
# Helper functions: reading events from different formats
# =============================================================================

def read_events_from_json(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Read events from a legacy accident-level JSON file.

    Expected structure:
    {
        "accident_id": "...",
        "events": [ {event}, {event}, ... ]
    }

    Returns:
        (accident_id, list_of_event_dicts)

    Failure behavior:
        - Returns (None, []) if file is malformed
        - NEVER raises an exception upstream
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if VERBOSE:
            print(f"[WARN] Failed to parse JSON file: {path.name}")
        return None, []

    acc_id = data.get("accident_id", path.stem)
    events = data.get("events", [])

    if not isinstance(events, list):
        if VERBOSE:
            print(f"[WARN] 'events' is not a list in {path.name}")
        return acc_id, []

    # Keep only dict-like events
    events = [ev for ev in events if isinstance(ev, dict)]
    return acc_id, events


def read_events_from_jsonl(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Read events from a JSON Lines (*.jsonl) file.

    Expected structure:
        One JSON object per line, each representing ONE event.

    Returns:
        (accident_id, list_of_event_dicts)

    Notes:
        - accident_id is inferred from filename stem
        - Invalid lines are skipped safely
    """
    acc_id = path.stem
    events: List[Dict[str, Any]] = []

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

        if isinstance(ev, dict):
            events.append(ev)

    return acc_id, events


# =============================================================================
# Main logic
# =============================================================================

def main():
    """
    Main entry point.

    Iterates over all files in EVENTS_DIR, reads events from each file
    (JSON or JSONL), and converts valid event texts into topic documents.
    """

    rows = []
    doc_id = 0

    if VERBOSE:
        print(f"[INFO] Reading events from: {EVENTS_DIR}")

    if not EVENTS_DIR.exists():
        print(f"[WARN] Events directory not found: {EVENTS_DIR}")
        return

    for path in sorted(EVENTS_DIR.iterdir()):

        # --------------------------------------------------------------
        # Determine input format
        # --------------------------------------------------------------
        if path.suffix == ".jsonl":
            acc_id, events = read_events_from_jsonl(path)

        elif path.suffix == ".json":
            acc_id, events = read_events_from_json(path)

        else:
            continue  # Ignore unrelated files

        if not events:
            if VERBOSE:
                print(f"[INFO] No valid events in {path.name}")
            continue

        # --------------------------------------------------------------
        # Convert events to topic documents
        # --------------------------------------------------------------
        for ev in events:
            # add an condition, if event_polarity is not "NEGATIVE" then skip
            if ev.get("event_polarity") != "NEGATIVE":
                continue
            text = ev.get("evidence_text", "")

            # Defensive text filtering
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < MIN_TEXT_LEN:
                continue

            rows.append({
                "doc_id": doc_id,
                "accident_id": acc_id,
                "text": text,
                "event_polarity": ev.get("event_polarity")
            })

            doc_id += 1

        if VERBOSE:
            print(f"[OK] {path.name}: {len(events)} events scanned")

    # --------------------------------------------------------------
    # Write output
    # --------------------------------------------------------------
    df = pd.DataFrame(rows)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Topic documents written → {OUT_CSV}")
    print(f"[INFO] Documents: {len(df)}")
    print("[DONE] Program 04 completed successfully")


# =============================================================================
# Script entry
# =============================================================================

if __name__ == "__main__":
    main()
