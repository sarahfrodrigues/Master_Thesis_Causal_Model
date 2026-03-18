"""
01_hyperparams_effect.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import warnings
import time
import json
warnings.filterwarnings("ignore")


# =============================================================================
# NLTK setup
# =============================================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =============================================================================
# Configuration
# =============================================================================
# INPUT:
TOPIC_DOCS_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_PREPROCESSED.csv"
)
LOCAL_MODEL_PATH = "all-MiniLM-L6-v2"
STOPWORDS_LIST = list(stopwords.words("english"))
GLOSSARY_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)

# OUTPUT:
VISUALIZATION_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\Plots"
)
VISUALIZATION_DIR.mkdir(exist_ok=True)

# Ranges for testing
NUM_NEIGHBORS_RANGE = [10, 15, 20, 25, 30, 35, 40, 45]
MIN_TOPIC_SIZE_RANGE = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
MIN_SAMPLE_SIZE_RANGE = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

# =============================================================================
# Utility Functions
# =============================================================================


def expand_abbreviations(text, abbr_dict):
    if pd.isna(text):
        return text

    text = str(text)

    for abbr, full in abbr_dict.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)

    return text


def load_and_preprocess(csv_path: Path, glossary) -> List[str]:
    """Load and preprocess text data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    with open(glossary, "r", encoding="utf-8") as f:
        abreviations = json.load(f)

    print("Expanding abbreviations...")
    df["Report 1_Narrative"] = df["Report 1_Narrative"].astype(str).apply(
        lambda x: expand_abbreviations(str(x), abreviations)
    )

    documents = df["Report 1_Narrative"].astype(str).tolist()
    documents = [doc.strip() for doc in documents if len(doc.strip()) > 10]

    print(f"Loaded {len(documents)} documents")
    return documents


# =============================================================================
# Coherence
# =============================================================================


def calculate_topic_coherence_fast(topic_model, docs, topn=10):

    topics = topic_model.get_topics()
    topic_words = []

    for topic_id in sorted(topics.keys()):
        if topic_id == -1:
            continue
        words = [word for word, _ in topics[topic_id][:topn]]
        if words:
            topic_words.append(words)

    if len(topic_words) < 2:
        return 0.0

    tokenized_docs = [
        [
            word.lower()
            for word in doc.split()
            if word.isalnum() and len(word) > 2
        ]
        for doc in docs
    ]

    dictionary = Dictionary(tokenized_docs)

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v"
    )

    return coherence_model.get_coherence()


# =============================================================================
# Parameter Testing Functions
# =============================================================================


def test_num_neighbors_optimized(
    docs: List[str],
    embeddings: np.ndarray,
    base_params: Dict,
    embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """Test UMAP n_neighbors parameter."""

    results = []
    total_start = time.time()

    for num_neighbors in NUM_NEIGHBORS_RANGE:
        iter_start = time.time()
        print(f"\nTesting n_neighbors={num_neighbors}...")

        # Create UMAP model with current n_neighbors
        umap_model = UMAP(
            n_neighbors=num_neighbors,
            n_components=3,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )

        # HDBSCAN with fixed parameters
        hdbscan_model = HDBSCAN(
            min_cluster_size=base_params["MIN_TOPIC_SIZE"],
            min_samples=base_params["MIN_SAMPLE_SIZE"],
            metric="euclidean",
            prediction_data=True
        )

        # Vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=STOPWORDS_LIST
        )

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            verbose=False,
            calculate_probabilities=False,
            nr_topics="auto"
        )

        # Fit with pre-computed embeddings
        topics, _ = topic_model.fit_transform(docs, embeddings)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        n_topics = int((topic_info["Topic"] >= 0).sum())
        outliers = list(topics).count(-1)

        # Calculate coherence (skip if too few/many topics)
        if 3 <= n_topics <= 50:
            coherence = calculate_topic_coherence_fast(topic_model, docs)
        else:
            coherence = 0.0

        # Store results
        outlier_pct = (
            (outliers / len(docs)) * 100 if len(docs) > 0 else 0
        )
        results.append({
            "num_neighbors": num_neighbors,
            "coherence": coherence,
            "n_valid_topics": n_topics,
            "n_outlier_docs": outliers,
            "outlier_percentage": outlier_pct
        })

        iter_time = time.time() - iter_start
        print(f"  Results: Topics={n_topics}, Coherence={coherence:.4f}, "
              f"Outliers={outliers} ({results[-1]['outlier_percentage']:.1f}%)"
              f"Time={iter_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\nCompleted n_neighbors test in {total_time:.1f} seconds")

    return pd.DataFrame(results)


def test_min_topic_size_optimized(
    docs: List[str],
    embeddings: np.ndarray,
    base_params: Dict,
    embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """Test HDBSCAN min_cluster_size parameter."""

    results = []
    total_start = time.time()

    for size in MIN_TOPIC_SIZE_RANGE:
        iter_start = time.time()
        print(f"\nTesting min_topic_size={size}...")

        # UMAP with fixed parameters
        umap_model = UMAP(
            n_neighbors=base_params["NUM_NEIGHBORS"],
            n_components=3,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )

        # HDBSCAN with current min_cluster_size
        hdbscan_model = HDBSCAN(
            min_cluster_size=size,
            min_samples=base_params["MIN_SAMPLE_SIZE"],
            metric="euclidean",
            prediction_data=True
        )

        # Vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=STOPWORDS_LIST
        )

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            verbose=False,
            calculate_probabilities=False
        )

        # Fit with pre-computed embeddings
        topics, _ = topic_model.fit_transform(docs, embeddings)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        n_topics = int((topic_info["Topic"] >= 0).sum())
        outliers = list(topics).count(-1)

        # Calculate coherence
        if 3 <= n_topics <= 50:
            coherence = calculate_topic_coherence_fast(topic_model, docs)
        else:
            coherence = 0.0

        # Store results
        outlier_pct = (
            (outliers / len(docs)) * 100 if len(docs) > 0 else 0
        )
        results.append({
            "min_topic_size": size,
            "coherence": coherence,
            "n_valid_topics": n_topics,
            "n_outlier_docs": outliers,
            "outlier_percentage": outlier_pct
        })

        iter_time = time.time() - iter_start
        print(f"  Results: Topics={n_topics}, Coherence={coherence:.4f}, "
              f"Outliers={outliers} ({results[-1]['outlier_percentage']:.1f}%)"
              f"Time={iter_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\nCompleted min_topic_size test in {total_time:.1f} seconds")

    return pd.DataFrame(results)


def test_min_sample_size_optimized(
    docs: List[str],
    embeddings: np.ndarray,
    base_params: Dict,
    embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """Test HDBSCAN min_samples parameter."""

    results = []
    total_start = time.time()

    for size in MIN_SAMPLE_SIZE_RANGE:
        iter_start = time.time()
        print(f"\nTesting min_sample_size={size}...")

        # UMAP with fixed parameters
        umap_model = UMAP(
            n_neighbors=base_params["NUM_NEIGHBORS"],
            n_components=3,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )

        # HDBSCAN with current min_samples
        hdbscan_model = HDBSCAN(
            min_cluster_size=base_params["MIN_TOPIC_SIZE"],
            min_samples=size,
            metric="euclidean",
            prediction_data=True
        )

        # Vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=STOPWORDS_LIST
        )

        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            verbose=False,
            calculate_probabilities=False
        )

        # Fit with pre-computed embeddings
        topics, _ = topic_model.fit_transform(docs, embeddings)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        n_topics = int((topic_info["Topic"] >= 0).sum())
        outliers = list(topics).count(-1)

        # Calculate coherence
        if 3 <= n_topics <= 50:
            coherence = calculate_topic_coherence_fast(topic_model, docs)
        else:
            coherence = 0.0

        # Store results
        outlier_pct = (
            (outliers / len(docs)) * 100 if len(docs) > 0 else 0
        )
        results.append({
            "min_sample_size": size,
            "coherence": coherence,
            "n_valid_topics": n_topics,
            "n_outlier_docs": outliers,
            "outlier_percentage": outlier_pct
        })

        iter_time = time.time() - iter_start
        print(f"  Results: Topics={n_topics}, Coherence={coherence:.4f}, "
              f"Outliers={outliers} ({results[-1]['outlier_percentage']:.1f}%)"
              f"Time={iter_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\nCompleted min_sample_size test in {total_time:.1f} seconds")

    return pd.DataFrame(results)


# =============================================================================
# Thesis / Scientific Visualizations
# =============================================================================

def set_thesis_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300
    })


def plot_coherence_topics(
    x,
    coherence,
    topics,
    xlabel,
    title,
    filename
):
    set_thesis_style()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        x, coherence,
        marker="o",
        linewidth=2,
        label="Coherence (c_v)"
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Coherence score")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x, topics,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="orange",
        label="Number of topics"
    )
    ax2.set_ylabel("Number of topics")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / filename, dpi=300)
    plt.show()


def plot_outliers_subplot(
    df_neighbors,
    df_topic_size,
    df_sample_size
):
    set_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    axes[0].plot(
        df_neighbors["num_neighbors"],
        df_neighbors["outlier_percentage"],
        marker="x",
        color="green",
        linewidth=2
    )
    axes[0].set_title("Outliers vs Number of Neighbors")
    axes[0].set_xlabel("Number of neighbors")
    axes[0].set_ylabel("Outlier documents (%)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        df_topic_size["min_topic_size"],
        df_topic_size["outlier_percentage"],
        marker="x",
        color="green",
        linewidth=2
    )
    axes[1].set_title("Outliers vs Min Cluster Size")
    axes[1].set_xlabel("Min cluster size")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        df_sample_size["min_sample_size"],
        df_sample_size["outlier_percentage"],
        marker="x",
        color="green",
        linewidth=2
    )
    axes[2].set_title("Outliers vs Min Sample Size")
    axes[2].set_xlabel("Min sample size")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        VISUALIZATION_DIR / "outliers_hyperparameters.png",
        dpi=300
    )
    plt.show()


def create_visualizations(
    results_num_neighbors,
    results_min_topic_size,
    results_min_sample_size
):

    print("Creating thesis-quality figures...")

    # Figure 1
    plot_coherence_topics(
        x=results_num_neighbors["num_neighbors"],
        coherence=results_num_neighbors["coherence"],
        topics=results_num_neighbors["n_valid_topics"],
        xlabel="Number of neighbors",
        title="Effect of UMAP n_neighbors on topic coherence and topic count",
        filename="neighbors_coherence_topics.png"
    )

    # Figure 2
    plot_coherence_topics(
        x=results_min_topic_size["min_topic_size"],
        coherence=results_min_topic_size["coherence"],
        topics=results_min_topic_size["n_valid_topics"],
        xlabel="Minimum cluster size",
        title="Effect of HDBSCAN min_cluster_size on topic coherence and topic count",
        filename="min_cluster_size_coherence_topics.png"
    )

    # Figure 3
    plot_coherence_topics(
        x=results_min_sample_size["min_sample_size"],
        coherence=results_min_sample_size["coherence"],
        topics=results_min_sample_size["n_valid_topics"],
        xlabel="Minimum samples",
        title="Effect of HDBSCAN min_samples on topic coherence and topic count",
        filename="min_samples_coherence_topics.png"
    )

    # Figure 4
    plot_outliers_subplot(
        results_num_neighbors,
        results_min_topic_size,
        results_min_sample_size
    )


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main execution function."""

    print("=" * 70)
    print("BERTopic Hyperparameter Tuning")
    print("=" * 70)

    total_start_time = time.time()

    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    docs = load_and_preprocess(TOPIC_DOCS_CSV, GLOSSARY_PATH)

    # 2. Load embedding model
    print("\n2. Loading embedding model...")
    embedding_model = SentenceTransformer(LOCAL_MODEL_PATH)

    # 3. Compute embeddings (ONCE - major optimization)
    print("\n3. Computing document embeddings...")
    embed_start = time.time()
    embeddings = embedding_model.encode(
        docs,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    embed_time = time.time() - embed_start
    print(f"Embedding computation completed in {embed_time:.1f} seconds")
    print(f"Embedding shape: {embeddings.shape}")

    # 4. Define base parameters
    base_params = {
        "NUM_NEIGHBORS": 25,
        "MIN_TOPIC_SIZE": 130,
        "MIN_SAMPLE_SIZE": 50
    }

    print(f"\n4. Starting hyperparameter tuning w/ base params: {base_params}")

    # 5. Test NUM_NEIGHBORS
    print("\n" + "=" * 50)
    print("TESTING: UMAP n_neighbors parameter")
    print("=" * 50)
    results_num_neighbors = test_num_neighbors_optimized(
        docs, embeddings, base_params, embedding_model
    )

    # Update with best n_neighbors
    if (not results_num_neighbors.empty and
            results_num_neighbors["coherence"].max() > 0):
        best_idx = results_num_neighbors["coherence"].idxmax()
        best_neighbors = results_num_neighbors.loc[
            best_idx, "num_neighbors"
        ]
        base_params["NUM_NEIGHBORS"] = best_neighbors
        print(f"\nBest n_neighbors: {best_neighbors}")
    else:
        print("\nNo valid n_neighbors found, keeping default")

    # 6. Test MIN_TOPIC_SIZE
    print("\n" + "=" * 50)
    print("TESTING: HDBSCAN min_cluster_size parameter")
    print("=" * 50)
    results_min_topic_size = test_min_topic_size_optimized(
        docs, embeddings, base_params, embedding_model
    )

    # Update with best min_topic_size
    if (not results_min_topic_size.empty and
            results_min_topic_size["coherence"].max() > 0):
        best_idx = results_min_topic_size["coherence"].idxmax()
        best_topic_size = results_min_topic_size.loc[
            best_idx, "min_topic_size"
        ]
        base_params["MIN_TOPIC_SIZE"] = best_topic_size
        print(f"\n✓ Best min_topic_size: {best_topic_size}")
    else:
        print("\n No valid min_topic_size found, keeping default")

    # 7. Test MIN_SAMPLE_SIZE
    print("\n" + "=" * 50)
    print("TESTING: HDBSCAN min_samples parameter")
    print("=" * 50)
    results_min_sample_size = test_min_sample_size_optimized(
        docs, embeddings, base_params, embedding_model
    )

    # 8. Save results
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)

    results_num_neighbors.to_csv(
        VISUALIZATION_DIR / "num_neighbors_results.csv",
        index=False
    )
    results_min_topic_size.to_csv(
        VISUALIZATION_DIR / "min_topic_size_results.csv",
        index=False
    )
    results_min_sample_size.to_csv(
        VISUALIZATION_DIR / "min_sample_size_results.csv",
        index=False
    )

    # 9. Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(
        results_num_neighbors,
        results_min_topic_size,
        results_min_sample_size
    )

    # 10. Final summary
    total_time = time.time() - total_start_time

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 70)

    print(f"\nTotal execution time: ({total_time/60:.1f}min)")

    print("\nOPTIMAL PARAMETERS FOUND:")
    print("-" * 30)
    for param, value in base_params.items():
        print(f"  {param}: {value}")

    # Find best overall coherence
    all_results = [
        results_num_neighbors["coherence"].max(),
        results_min_topic_size["coherence"].max(),
        results_min_sample_size["coherence"].max()
    ]
    best_coherence = max(all_results)
    print(f"\nBest coherence achieved: {best_coherence:.4f}")

    print(f"\nResults saved to: {VISUALIZATION_DIR}")
    print("[DONE] Program 05 completed successfully")


if __name__ == "__main__":
    main()
