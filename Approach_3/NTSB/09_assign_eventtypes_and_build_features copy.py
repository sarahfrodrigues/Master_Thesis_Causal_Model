"""
09_assign_eventtypes_and_build_features.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DOC_TOPICS = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\doc_topics.csv"
)
TOPIC_INFO = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_info.csv"
)
ACCIDENT_TOPIC_MATRIX_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\accident_topic_matrix_labeled.csv"
)

# Load data
df_doc_topics = pd.read_csv(DOC_TOPICS)
df_topic_info = pd.read_csv(TOPIC_INFO)

EXCLUDED_TOPICS = {-1}

df_doc_topics = df_doc_topics[
    ~df_doc_topics["topic_id"].isin(EXCLUDED_TOPICS)
]

# -------------------------------------------------------------------------
# Core Logic: Create Labeled Accident-Topic Matrix
# -------------------------------------------------------------------------


# Merge topic labels
df_merged = df_doc_topics.merge(
    df_topic_info,
    on="topic_id",
    how="left"
)

# Create binary accident-topic matrix
accident_topic_matrix = (
    df_merged
    .assign(value=1)
    .pivot_table(
        index="accident_id",
        columns="Name",
        values="value",
        aggfunc="max",
        fill_value=0
    )
    .reset_index()
)

# Save result
accident_topic_matrix.to_csv(ACCIDENT_TOPIC_MATRIX_OUT, index=False)

print("[DONE] Program 09 completed successfully")
