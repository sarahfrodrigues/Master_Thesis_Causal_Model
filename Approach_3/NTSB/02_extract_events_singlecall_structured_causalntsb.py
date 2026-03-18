from __future__ import annotations
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import traceback
import dashscope
from dashscope import Generation

# =============================================================================
# Configuration
# =============================================================================

# Input/output paths
INPUT_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\narratives_output"
)
OUTPUT_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\events_output"
)
STATE_FILE = OUTPUT_DIR / "processing_state.json"  # For crash recovery
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = "qwen3-max"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 800

# Segmentation parameters
SEGMENT_CHARS = 4500
SEGMENT_OVERLAP = 300

# Reliability knobs
MAX_RETRIES_PER_SEGMENT = 3  # Increased from 2 for better resilience
INITIAL_RETRY_SLEEP = 0.8    # Initial sleep time (exponential backoff)
MAX_RETRY_SLEEP = 10         # Max sleep time between retries
API_TIMEOUT = 30             # Timeout for API calls (seconds)

# Progress tracking
VERBOSE = True
dashscope.api_key = ""  # Replace with actual key

# Valid vocabularies (keep your original config)
VALID_EVENT_POLARITIES = {"NEGATIVE", "NEUTRAL", "POSITIVE"}

# =============================================================================
# Progress Tracking (Crash Recovery)
# =============================================================================


@dataclass
class ProcessingState:
    """Track progress to resume after crashes"""
    last_processed_file: str = ""
    last_completed_segment: int = -1
    completed_files: List[str] = None

    def __post_init__(self):
        if self.completed_files is None:
            self.completed_files = []

    def save(self):
        """Save state to JSON file"""
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump({
                "last_processed_file": self.last_processed_file,
                "last_completed_segment": self.last_completed_segment,
                "completed_files": self.completed_files
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "ProcessingState":
        """Load state from JSON file (or create new if missing)"""
        if not STATE_FILE.exists():
            return cls()

        try:
            with STATE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                last_processed_file=data.get("last_processed_file", ""),
                last_completed_segment=data.get("last_completed_segment", -1),
                completed_files=data.get("completed_files", [])
            )
        except (json.JSONDecodeError, KeyError):
            # Corrupted state file - start fresh
            if VERBOSE:
                print("[WARN] Corrupted state file - starting fresh")
            return cls()

# =============================================================================
# Text Segmentation (Unchanged)
# =============================================================================


def split_text(text: str) -> List[str]:
    """Split long narrative into overlapping character-based segments"""
    segments = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + SEGMENT_CHARS)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)

        if end >= n:
            break
        start = end - SEGMENT_OVERLAP

    return segments

# =============================================================================
# Prompt Construction (Unchanged)
# =============================================================================


def build_prompt(segment_text: str) -> str:
    """Build prompt for one segment (JSON Lines output)"""
    return f"""
You are an aviation accident analyst extracting EVENTS for CAUSAL ANALYSIS.

The text below is ONE CONTIGUOUS SEGMENT from an aviation accident narrative.
Treat it independently.

STRICT RULES (VERY IMPORTANT):
1. Extract ONLY events explicitly supported by the text.
2. Do NOT infer causality beyond what is explicitly stated.
3. Do NOT merge information across segments.
5. OUTPUT FORMAT IS CRITICAL (see below).

DEFINITION OF AN EVENT:
A concrete action, failure, decision, maneuver, communication,
system behavior, environmental condition, or physical occurrence.

FIELDS FOR EACH EVENT (ALL REQUIRED):
actor
action
object
outcome
phase
evidence_text
event_polarity

CONTROLLED VOCABULARIES:
event_polarity:
NEGATIVE | NEUTRAL | POSITIVE

OUTPUT FORMAT (MANDATORY):
- Output ONE JSON OBJECT PER LINE
- NO surrounding array
- NO markdown
- NO commentary

Example:
{{"actor": "...", "action": "...", ...}}
{{"actor": "...", "action": "...", ...}}

[SEGMENT TEXT]
{segment_text}
""".strip()

# =============================================================================
# Event Normalization (Unchanged)
# =============================================================================


def normalize_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize event fields (replace invalid values with None)"""
    if ev.get("event_polarity") not in VALID_EVENT_POLARITIES:
        ev["event_polarity"] = None
    return ev

# =============================================================================
# Robust LLM Call with Retries & Error Handling
# =============================================================================


def extract_events_from_segment(
    segment_text: str,
    acc_id: str,
    seg_id: int,
) -> List[Dict[str, Any]]:
    """
    Call LLM with robust error handling and exponential backoff retries
    Returns list of valid events (empty if all retries fail)
    """
    prompt = build_prompt(segment_text)
    last_error = None
    last_raw = ""

    for attempt in range(1, MAX_RETRIES_PER_SEGMENT + 1):
        try:
            # Add jitter to avoid rate limiting
            if attempt > 1:
                backoff = INITIAL_RETRY_SLEEP * (2 ** (attempt - 1))
                sleep_time = min(
                    backoff + random.uniform(0, 1),
                    MAX_RETRY_SLEEP
                )
                if VERBOSE:
                    print(f"[RETRY] acc={acc_id} seg={seg_id} attempt={attempt} - sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

            # Make API call with timeout
            resp = Generation.call(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": "You extract structured accident events."},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                timeout=API_TIMEOUT
            )

            # Check API response status
            if resp.status_code != 200:
                last_error = f"API Error: {resp.code} - {resp.message}"
                if VERBOSE:
                    print(f"[WARN] acc={acc_id} seg={seg_id} attempt={attempt}: {last_error}")
                continue

            # Parse JSON Lines output
            raw = resp.output.choices[0].message.content.strip()
            last_raw = raw
            events: List[Dict[str, Any]] = []

            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if isinstance(ev, dict):
                        ev["_segment_id"] = seg_id
                        events.append(normalize_event(ev))
                except json.JSONDecodeError:
                    continue  # Skip invalid lines

            if events:
                return events
            else:
                last_error = "No valid events parsed"
                if VERBOSE:
                    print(f"[WARN] acc={acc_id} seg={seg_id} attempt={attempt}: {last_error}")

        except Exception as e:
            # Catch ALL exceptions (network, SSL, timeout, etc.)
            last_error = f"{type(e).__name__}: {str(e)}"
            if VERBOSE:
                print(f"[ERROR] acc={acc_id} seg={seg_id} attempt={attempt}: {last_error}")
                print(f"[TRACE] {traceback.format_exc()[:500]}")

    # All retries failed
    if VERBOSE:
        print(f"[FAIL] Segment failed after {MAX_RETRIES_PER_SEGMENT} attempts: acc={acc_id} seg={seg_id}")
        if last_raw:
            print(f"[RAW] Last output: {last_raw[:500]}")
    return []

# =============================================================================
# Accident Processing with Progress Tracking
# =============================================================================


def process_accident(txt_path: Path, state: ProcessingState) -> bool:
    """
    Process a single accident with crash recovery
    Returns True if completed successfully, False if failed/crashed
    """
    acc_id = txt_path.stem

    # Skip if already completed
    if acc_id in state.completed_files:
        if VERBOSE:
            print(f"[SKIP] Already processed: {acc_id}")
        return True

    # Check if we need to resume from a specific segment
    resume_segment = -1
    if acc_id == state.last_processed_file:
        resume_segment = state.last_completed_segment
        if VERBOSE:
            print(f"[RESUME] acc={acc_id} from segment {resume_segment + 1}")

    try:
        # Read input text
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        segments = split_text(text)
        out_path = OUTPUT_DIR / f"{acc_id}.jsonl"

        # If resuming, use append mode; else write new file
        write_mode = "a" if (resume_segment >= 0 and out_path.exists()) else "w"
        n_events = 0

        with out_path.open(write_mode, encoding="utf-8") as fout:
            for seg_id, seg_text in enumerate(segments, start=1):
                # Skip segments already processed
                if seg_id <= resume_segment:
                    if VERBOSE:
                        print(f"[SKIP] acc={acc_id} seg={seg_id}/{len(segments)} (already processed)")
                    continue

                if VERBOSE:
                    print(f"[SEGMENT] acc={acc_id} seg={seg_id}/{len(segments)} chars={len(seg_text)}")

                # Update state before processing segment (crash safety)
                state.last_processed_file = acc_id
                state.last_completed_segment = seg_id - 1  # Last completed
                state.save()

                # Process segment
                events = extract_events_from_segment(seg_text, acc_id, seg_id)

                # Write events (atomic line-by-line)
                for ev in events:
                    fout.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    n_events += 1

                # Flush to disk to prevent data loss
                fout.flush()

        # Mark as completed
        state.last_processed_file = acc_id
        state.last_completed_segment = len(segments)
        state.completed_files.append(acc_id)
        state.save()

        if VERBOSE:
            print(f"[OK] Accident {acc_id}: {n_events} events written → "
                  f"{out_path}")
        return True

    except Exception as e:
        # Fatal error for this accident - save state and return False
        if VERBOSE:
            print(f"[CRASH] Processing failed for {acc_id}: {type(e).__name__}: {str(e)}")
            print(f"[TRACE] {traceback.format_exc()[:500]}")
        # Save current state to resume later
        state.save()
        return False

# =============================================================================
# Main Entry Point with Recovery
# =============================================================================


def main():
    # Load progress state
    state = ProcessingState.load()
    if VERBOSE:
        print(f"[START] Resuming from: {state.last_processed_file} "
              f"(segment {state.last_completed_segment})")
        print(f"[INFO] Already completed: {len(state.completed_files)} files")

    # Get list of input files (sorted for consistency)
    input_files = sorted(INPUT_DIR.glob("*.txt"))
    total_files = len(input_files)

    # Process files (skip completed ones)
    for idx, txt_path in enumerate(input_files):
        acc_id = txt_path.stem

        # Skip if we haven't reached the resume point yet
        if state.last_processed_file and acc_id < state.last_processed_file:
            if VERBOSE:
                print(f"[SKIP] Not yet at resume point: {acc_id}")
            continue

        if VERBOSE:
            print(f"\n[PROCESS] [{idx+1}/{total_files}] Accident {acc_id}")

        # Process accident (returns False if crashed)
        success = process_accident(txt_path, state)

        # Stop if processing failed (crash)
        if not success:
            print(f"\n[STOP] Processing failed at {acc_id} - "
                  "resume possible from this point")
            break

    # Clean up state if all files completed
    if len(state.completed_files) == total_files:
        if VERBOSE:
            print("\n[DONE] All files processed - clearing state")
        state = ProcessingState()  # Reset state
        state.save()

    print("[DONE] Program 02 completed successfully")


if __name__ == "__main__":
    main()
