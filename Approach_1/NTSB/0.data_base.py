"""
01_export_narratives_to_csv.py

Exports NTSB accident narratives from both:
- avall.mdb (post-2008)
- Pre2008.mdb (pre-2008)

Applies:
- FAR Part filtering
- minimum narrative length
- deduplication by ev_id

Outputs:
- NTSB_ALL.csv
"""

import os
import csv
import pyodbc
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

MDB_PATHS = [
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_RAW_DATA\avall.mdb",
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_RAW_DATA\Pre2008.mdb"
]

OUTPUT_DIR = r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common"
OUTPUT_CSV = "NTSB_ALL.csv"

# =====================================================
# PARAMETERS
# =====================================================
MIN_NARR_CHAR = 500
MIN_NARR_WORDS = 100
MAX_NARR_WORDS = 600
FAR_PARTS = ("121", "129")

# =============================================================================
# Helper functions
# =============================================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_strip(value) -> str:
    return str(value).strip() if value is not None else ""


def format_date(dt_value) -> str:
    if dt_value is None:
        return ""
    if isinstance(dt_value, datetime):
        return dt_value.strftime("%Y-%m-%d")
    try:
        return dt_value.strftime("%Y-%m-%d")
    except Exception:
        return str(dt_value)


# =============================================================================
# Database extraction
# =============================================================================

def extract_from_mdb(mdb_path):
    print(f"[INFO] Reading database: {os.path.basename(mdb_path)}")

    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        fr"DBQ={mdb_path};"
    )

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

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

    cursor.close()
    conn.close()

    print(f"[INFO] Rows extracted: {len(rows)}")

    return rows


# =============================================================================
# Main logic
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # 1) Extract from all MDB files
    # -------------------------------------------------------------------------
    all_rows = []

    for mdb_path in MDB_PATHS:
        rows = extract_from_mdb(mdb_path)
        all_rows.extend(rows)

    print(f"[INFO] Total rows before filtering: {len(all_rows)}")
    # -------------------------------------------------------------------------
    # 2) Deduplication + filtering
    # -------------------------------------------------------------------------
    seen_ev_ids = set()
    final_rows = []

    for r in all_rows:
        ev_id = safe_strip(r.ev_id)

        if not ev_id or ev_id in seen_ev_ids:
            continue

        narrative = safe_strip(r.narr_accp)

        # -------------------------------
        # Character filter
        # -------------------------------
        if len(narrative) < MIN_NARR_CHAR:
            continue

        # -------------------------------
        # Word-count filter
        # -------------------------------
        word_count = len(narrative.split())

        if word_count < MIN_NARR_WORDS or word_count > MAX_NARR_WORDS:
            continue

        seen_ev_ids.add(ev_id)

        final_rows.append([
            ev_id,
            format_date(r.ev_date),
            narrative
        ])

    # -------------------------------------------------------------------------
    # 3) Write CSV
    # -------------------------------------------------------------------------
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ev_id", "date", "narrative"])
        writer.writerows(final_rows)

    print("[OK] CSV export completed successfully.")
    print(f"[OK] File saved at: {output_path}")
    print("[DONE] Program completed successfully.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
