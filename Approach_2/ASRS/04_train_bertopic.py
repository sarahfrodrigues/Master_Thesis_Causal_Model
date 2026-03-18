"""
08_train_bertopic_model.py
Added BERTopic visualizations & automatic HTML export
All visualizations are interactive, browser-friendly HTML files
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from umap import UMAP
from hdbscan import HDBSCAN


# Download stopwords silently if not present
nltk.download('stopwords', quiet=True)

# =============================================================================
# Configuration
# =============================================================================


stopwords_additional = {"airplane", "aircraft", "acft", "flight", "flt",
                        "time", "fly", "airline", "plane", "airport",
                        "ntsb", "report", "faa"
                        }

# Rename to avoid conflict with scikit-learn stopwords parameter
custom_stopwords = (list(stopwords.words('english')) +
                    list(stopwords_additional))

# Output paths for model and results
MODEL_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\bertopic_model.pkl"
)
DOC_TOPICS_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\doc_topics.csv"
)
TOPIC_INFO_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\topic_info.csv"
)
TOPIC_KEYWORDS_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\topic_keywords.json"
)
# Dedicated folder for BERTopic visualizations
VISUALIZATION_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\visualizations"
)
LOCAL_MODEL_PATH = Path(r"C:\models\all-MiniLM-L6-v2")
GLOSSARY_PATH = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)
TOPIC_DOCS_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_PREPROCESSED.csv"
)

# -------------------------------
# Run Mode Switch
# -------------------------------
# MODE = "EXPLORATORY"
MODE = "FULL_DATA"

# =============================================================================
# Mode-Specific Hyperparameters
# =============================================================================
if MODE == "EXPLORATORY":
    MIN_DOCS_REQUIRED = 30
    MIN_TOPIC_SIZE = 10
    NR_TOPICS = None

elif MODE == "FULL_DATA":
    MIN_DOCS_REQUIRED = 300
    MIN_TOPIC_SIZE = 180
    NR_TOPICS = None

else:
    raise ValueError(f"Unknown MODE selected: {MODE}")

# =============================================================================
# Helper Functions
# =============================================================================


def expand_abbreviations(text: str | float, abbr_dict: dict) -> str | float:
    """Expand abbreviations in text using a glossary dictionary."""
    if pd.isna(text):
        return text
    text = str(text)
    for abbr, full_form in abbr_dict.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
    return text


def convert_keys_to_string(data: dict | list) -> dict | list:
    """Convert dictionary keys to strings for valid JSON serialization."""
    if isinstance(data, dict):
        return {str(k): convert_keys_to_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_string(item) for item in data]
    else:
        return data


def save_bertopic_visuals(model: BERTopic, docs: list[str], save_dir: Path) -> None:
    """
    Generate and save core BERTopic visualizations as interactive HTML files.
    Handles exceptions to avoid crashing the pipeline if some plots fail.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Generating BERTopic visualizations → {save_dir}")

    try:
        # 1. Topic hierarchy dendrogram
        hierarchy = model.visualize_hierarchy()
        hierarchy.write_html(save_dir / "topic_hierarchy.html")
        print("[OK] Saved topic hierarchy plot")

        # 2. 2D UMAP projection of topics
        topics_2d = model.visualize_topics()
        topics_2d.write_html(save_dir / "topic_2d_projection.html")
        print("[OK] Saved 2D topic projection")

        # 3. Top terms barchart per topic
        barchart = model.visualize_barchart(top_n_topics=15, n_words=10)
        barchart.write_html(save_dir / "topic_term_barchart.html")
        print("[OK] Saved topic term barchart")

        # 4. Topic similarity heatmap
        heatmap = model.visualize_heatmap()
        heatmap.write_html(save_dir / "topic_similarity_heatmap.html")
        print("[OK] Saved topic similarity heatmap")

        # 5. Document-topic distribution (hide annotations for large datasets)
        doc_dist = model.visualize_documents(docs, hide_annotations=True)
        doc_dist.write_html(save_dir / "document_topic_distribution.html")
        print("[OK] Saved document-topic distribution plot")

        print("[INFO] All visualizations generated successfully")

    except Exception as e:
        print(f"[WARNING] Visualization generation failed: {str(e)}")
        print("Some plots may fail with few topics or high outlier rates.")

# =============================================================================
# Main Training Pipeline
# =============================================================================


def main():
    # Validate input file exists
    if not TOPIC_DOCS_CSV.exists():
        print("[SKIP] Input CSV not found. Run Program 04 first.")
        return

    # Load and preprocess documents
    df = pd.read_csv(TOPIC_DOCS_CSV)
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        abbreviations = json.load(f)


    df["Report 1_Narrative"] = df["Report 1_Narrative"].apply(lambda x: expand_abbreviations(x, abbreviations))
    docs = df["Report 1_Narrative"].astype(str).tolist()
    n_docs = len(docs)

    print(f"[INFO] Running in {MODE} mode")
    print(f"[INFO] Total documents loaded: {n_docs}")

    # Minimum document threshold check
    if n_docs < MIN_DOCS_REQUIRED:
        print(f"[SKIP] Insufficient documents: {n_docs} < {MIN_DOCS_REQUIRED}")
        return

    # Load embedding model (local if available)
    if LOCAL_MODEL_PATH.exists():
        embedding_model = SentenceTransformer(str(LOCAL_MODEL_PATH))
        print(f"[INFO] Using local embedding model: {LOCAL_MODEL_PATH}")
    else:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Local model not found; using Hugging Face all-MiniLM-L6-v2")

    # Initialize BERTopic components
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=custom_stopwords
    )
    umap_model = UMAP(
        n_neighbors=25,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=80,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # Initialize and train BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="english",
        nr_topics=NR_TOPICS,
        calculate_probabilities=True,
        verbose=True
    )
    print("[INFO] Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(docs)

    # Extract topic metadata and keywords
    topic_info = topic_model.get_topic_info()
    topic_keywords = {}
    for topic_id in topic_info["Topic"].unique():
        topic_id_py = int(topic_id)
        if topic_id_py >= 0:
            keywords = topic_model.get_topic(topic_id_py)
            topic_keywords[topic_id_py] = [term for term, _ in keywords[:10]]
        else:
            topic_keywords[topic_id_py] = []

    # Save topic keywords to JSON
    with open(TOPIC_KEYWORDS_OUT, "w", encoding="utf-8") as f:
        json.dump(convert_keys_to_string(topic_keywords), f, indent=2)

    # Print dataset statistics
    n_valid_topics = int((topic_info["Topic"] >= 0).sum())
    n_outliers = len([t for t in topics if t == -1])
    print(f"[INFO] Valid topics (excluding outliers): {n_valid_topics}")
    print(f"[INFO] Outlier documents: {n_outliers} / {n_docs}")

    # Quality guard for full data mode
    if MODE == "FULL_DATA" and n_valid_topics < 3:
        print("[SKIP] Too few valid topics (<3) in FULL_DATA mode; aborting save")
        return

    # Generate and save all visualizations
    save_bertopic_visuals(topic_model, docs, VISUALIZATION_DIR)

    # Create topics vs documents probs binary matrix
    if probs is not None:
        # Convert probabilities to binary matrix 
        binary_matrix = (probs > 0.1).astype(int)
        # Save binary matrix to CSV
        binary_df = pd.DataFrame(binary_matrix, columns=[f"topic_{i}" for i in range(binary_matrix.shape[1])])
        binary_df.to_csv(r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\binary_matrix.csv", index=False, encoding="utf-8")
        print("[OK] Binary document-topic matrix saved")

    # Create clean topic labels
    topic_info["Name"] = [f"topic_{int(t)}" if t >= 0 else "outlier" for t in topic_info["Topic"]]

    # Create output directories
    for path in [MODEL_OUT, DOC_TOPICS_OUT, TOPIC_INFO_OUT, TOPIC_KEYWORDS_OUT]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Save trained model and outputs
    topic_model.save(MODEL_OUT, serialization="safetensors")
    print(f"[OK] BERTopic model saved to: {MODEL_OUT}")

    df_with_topics = df.assign(topic_id=[int(t) for t in topics])
    df_with_topics.to_csv(DOC_TOPICS_OUT, index=False, encoding="utf-8")
    print(f"[OK] Document-topic assignments saved to: {DOC_TOPICS_OUT}")

    topic_info_clean = topic_info.rename(columns={"Topic": "topic_id"})
    topic_info_clean["topic_id"] = topic_info_clean["topic_id"].astype(int)
    topic_info_clean.to_csv(TOPIC_INFO_OUT, index=False, encoding="utf-8")
    print(f"[OK] Topic metadata saved to: {TOPIC_INFO_OUT}")

    print("[DONE] BERTopic training and visualization pipeline completed successfully")

# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    main()
