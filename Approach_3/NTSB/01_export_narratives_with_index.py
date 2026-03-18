"""
01_export_narratives_with_index.py
"""

import os
import csv
import pyodbc
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

# Path to AVALL Access database
AVALL_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\raw_data\avall.mdb"
)

# Output directory for TXT files and index.csv
OUTPUT_DIR = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\narratives_output"
)

# Index mapping filename
INDEX_CSV_NAME = "index.csv"

# Narrative length threshold
MIN_NARR_LEN = 50

# FAR parts typically associated with commercial operations
FAR_PARTS = ("121", "129")

# =============================================================================
# Helper functions
# =============================================================================


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def safe_strip(value) -> str:
    """
    Convert a value to string and strip whitespace.
    Returns an empty string if the value is None.
    """
    return str(value).strip() if value is not None else ""


def format_date_for_csv(dt_value) -> str:
    """
    Convert a pyodbc datetime value into a stable ISO date string (YYYY-MM-DD).
    """
    if dt_value is None:
        return ""
    if isinstance(dt_value, datetime):
        return dt_value.strftime("%Y-%m-%d")
    try:
        return dt_value.strftime("%Y-%m-%d")
    except Exception:
        return str(dt_value)

# =============================================================================
# Main export logic
# =============================================================================


def main():
    ensure_dir(OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # 1) Connect to Access database
    # -------------------------------------------------------------------------
    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        fr"DBQ={AVALL_PATH};"
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # -------------------------------------------------------------------------
    # 2) Query candidate narratives (SQL-safe for Memo fields)
    # -------------------------------------------------------------------------
    # IMPORTANT:
    # - NO LEN(), DISTINCT, or string functions on narr_accp
    # - Filtering and de-duplication are handled in Python
    #
    far_parts_sql = ",".join([f"'{p}'" for p in FAR_PARTS])

    sql = f"""
    SELECT
        e.ev_id,
        e.ev_date,
        n.narr_accp
    FROM
        (events AS e
         INNER JOIN aircraft AS a ON e.ev_id = a.ev_id)
         INNER JOIN narratives AS n ON e.ev_id = n.ev_id
    WHERE
        a.far_part IN ({far_parts_sql})
        AND n.narr_accp IS NOT NULL
        AND e.ev_date IS NOT NULL
    ORDER BY
        e.ev_date ASC;
    """

    cursor.execute(sql)
    rows = cursor.fetchall()

    print(f"[INFO] Rows returned by SQL (pre-dedup): {len(rows)}")

    # -------------------------------------------------------------------------
    # 3) Accident-level de-duplication + narrative length filtering (Python)
    # -------------------------------------------------------------------------
    # Guarantee: exactly one narrative per ev_id
    #
    seen_ev_ids = set()
    unique_rows = []

    for r in rows:
        ev_id = safe_strip(r.ev_id)
        if not ev_id:
            continue

        # Enforce accident-level uniqueness
        if ev_id in seen_ev_ids:
            continue

        narrative = safe_strip(r.narr_accp)
        if not narrative:
            continue

        # Apply narrative length threshold
        if len(narrative) <= MIN_NARR_LEN:
            continue

        seen_ev_ids.add(ev_id)
        unique_rows.append((ev_id, r.ev_date, narrative))

    print(f"[INFO] Unique accidents after ev_id dedup: {len(unique_rows)}")

    # -------------------------------------------------------------------------
    # 4) Write TXT narratives and index.csv
    # -------------------------------------------------------------------------
    index_path = os.path.join(OUTPUT_DIR, INDEX_CSV_NAME)

    with open(index_path, "w", encoding="utf-8-sig", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "idx",
            "ev_id",
            "ev_date",
            "narrative_len",
            "txt_filename"
        ])

        for idx, (ev_id, ev_date, narrative) in enumerate(
                unique_rows, start=1):
            txt_filename = f"{idx}.txt"
            txt_path = os.path.join(OUTPUT_DIR, txt_filename)

            # Write full narrative text (no truncation)
            with open(txt_path, "w", encoding="utf-8") as ftxt:
                ftxt.write(narrative)

            writer.writerow([
                idx,
                ev_id,
                format_date_for_csv(ev_date),
                len(narrative),
                txt_filename
            ])

    print("[OK] Export completed successfully.")
    print(f"[OK] Narratives written: {len(unique_rows)}")
    print(f"[OK] Output directory: {OUTPUT_DIR}")
    print(f"[OK] Index file: {index_path}")
    print("[DONE] Program 01 completed successfully")
    # -------------------------------------------------------------------------
    # 5) Clean up database resources
    # -------------------------------------------------------------------------
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
