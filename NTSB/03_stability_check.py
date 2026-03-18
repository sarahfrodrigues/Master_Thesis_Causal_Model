"""
03_stability_check.py  —  BERTopic Configuration Stability Optimizer
=====================================================================
FOCUS: Identify the most stable configuration via bootstrap resampling.

Core stability metric: Mean pairwise Jaccard similarity between topic
                       keyword sets across N bootstrap iterations.

Method: For each configuration, run N bootstrap samples → fit BERTopic → 
        extract topic keywords → compute pairwise Jaccard via Hungarian 
        matching → rank by mean similarity.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

warnings.filterwarnings("ignore")

# Bootstrap NLTK
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

stopwords_additional = {"airplane", "aircraft", "acft", "flight", "flt",
                        "time", "fly", "airline", "plane", "airport",
                        "ntsb", "report", "faa"}

# Rename to avoid conflict with scikit-learn stopwords parameter
custom_stopwords = list(stopwords.words('english')) + list(stopwords_additional)

# =============================================================================
# Configuration
# =============================================================================
TOPIC_DOCS_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_ALL.csv"
)
CONFIGURATIONS_FILE = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\NTSB\gs_result.csv"
)
GLOSSARY_PATH = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)
OUTPUT_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\NTSB\stability_optimization"
)
LOCAL_MODEL_PATH = Path(r"C:\models\all-MiniLM-L6-v2")

DEFAULT_N_ITERATIONS = 10
RANDOM_STATE = 42

# =============================================================================
# Data loading
# =============================================================================


def expand_abbreviations(text: str, abbr_dict: Dict[str, str]) -> str:
    if pd.isna(text):
        return text
    text = str(text)
    for abbr, full in abbr_dict.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    return text


def load_documents(csv_path: Path, glossary_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        df = df.rename(columns={df.columns[0]: "text"})

    with open(glossary_path, "r", encoding="utf-8") as fh:
        abbreviations = json.load(fh)

    print("Expanding abbreviations...")
    df["text"] = df["text"].astype(str).apply(
        lambda x: expand_abbreviations(x, abbreviations)
    )

    documents = [doc.strip() for doc in df["text"].astype(str).tolist() if len(doc.strip()) > 10]
    print(f"Loaded {len(documents)} documents")
    return documents


def load_configurations(config_file: Path, top_n: int = 10) -> List[Tuple]:
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    df = pd.read_csv(config_file)
    required = {"coherence", "min_topic_size", "min_sample_size", "num_neighbors"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_sorted = df.sort_values("coherence", ascending=False).head(top_n).reset_index(drop=True)

    configs = []
    for rank, row in enumerate(df_sorted.itertuples(index=False), start=1):
        configs.append((
            int(row.min_topic_size),
            int(row.min_sample_size),
            int(row.num_neighbors),
            f"Config_{rank}",
        ))

    print(f"\nLoaded {len(configs)} configurations:")
    for idx, (mts, ms, nn, name) in enumerate(configs, 1):
        print(f"  {idx}. {name}: min_topic_size={mts}, min_samples={ms}, n_neighbors={nn}")

    return configs


# =============================================================================
# Stability checker
# =============================================================================


class StabilityChecker:
    """Bootstrap-based stability evaluation for BERTopic configurations."""

    def __init__(
        self,
        n_iterations: int = DEFAULT_N_ITERATIONS,
        random_state: int = RANDOM_STATE,
        local_model_path: Path | None = None,
    ):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.local_model_path = local_model_path
        self.custom_stopwords = custom_stopwords

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.local_model_path and self.local_model_path.exists():
            return SentenceTransformer(str(self.local_model_path))
        return SentenceTransformer("all-MiniLM-L6-v2")

    def bootstrap_sample(
        self, documents: List[str], seed: int, sample_size: float = 0.8
    ) -> List[str]:
        """Draw bootstrap sample with explicit seed."""
        rng = np.random.RandomState(seed)
        n = int(len(documents) * sample_size)
        indices = rng.choice(len(documents), size=n, replace=True)
        return [documents[i] for i in indices]

    def fit_topic_model(
        self,
        documents: List[str],
        seed: int,
        min_topic_size: int,
        min_samples: int,
        n_neighbors: int,
    ) -> Tuple[BERTopic, List[int]]:
        """Fit BERTopic model with given parameters."""
        embedding_model = self._get_embedding_model()

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            ngram_range=(1, 3), stop_words=self.custom_stopwords
        )

        model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            language="english",
            nr_topics=None,
            calculate_probabilities=False,
            verbose=False,
        )

        topics, _ = model.fit_transform(documents)
        return model, topics

    @staticmethod
    def extract_keywords(model: BERTopic, top_n: int = 10) -> Dict[int, Set[str]]:
        """Extract top-N keywords per topic (excluding outlier topic -1)."""
        return {
            tid: {str(w).lower().strip() for w, _ in words[:top_n]}
            for tid, words in model.get_topics().items()
            if tid != -1
        }

    @staticmethod
    def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """Jaccard index between two sets."""
        if not set_a and not set_b:
            return 1.0
        union = len(set_a | set_b)
        return len(set_a & set_b) / union if union else 0.0

    def evaluate_stability(
        self,
        documents: List[str],
        min_topic_size: int,
        min_samples: int,
        n_neighbors: int,
        config_name: str,
    ) -> Dict:
        """
        Core stability evaluation: run N bootstrap iterations and compute
        mean pairwise Jaccard similarity via Hungarian matching.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"  min_topic_size={min_topic_size}, min_samples={min_samples}, n_neighbors={n_neighbors}")
        print(f"{'='*60}")

        start = time.time()
        all_keywords = []

        # Bootstrap iterations
        for i in range(self.n_iterations):
            seed = self.random_state + i
            bootstrap_docs = self.bootstrap_sample(documents, seed)
            model, _ = self.fit_topic_model(
                bootstrap_docs, seed, min_topic_size, min_samples, n_neighbors
            )
            keywords = self.extract_keywords(model)
            all_keywords.append(keywords)

            if (i + 1) % 2 == 0 or i == 0 or i == self.n_iterations - 1:
                print(f"  Iteration {i+1}/{self.n_iterations}: {len(keywords)} topics")

        # Pairwise similarity via Hungarian matching
        similarities = []
        for i in range(self.n_iterations):
            for j in range(i + 1, self.n_iterations):
                topics_i = list(all_keywords[i].values())
                topics_j = list(all_keywords[j].values())

                if not topics_i or not topics_j:
                    continue

                # Build similarity matrix
                sim_matrix = np.array([
                    [self.jaccard_similarity(ki, kj) for kj in topics_j]
                    for ki in topics_i
                ])

                # Hungarian algorithm for optimal matching
                row_ind, col_ind = linear_sum_assignment(-sim_matrix)
                matched = sim_matrix[row_ind, col_ind]

                if len(matched) > 0:
                    similarities.append(float(matched.mean()))

        # Stability metric
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        std_similarity = float(np.std(similarities)) if similarities else 0.0

        elapsed = time.time() - start

        # Stability tier
        if mean_similarity > 0.5:
            tier = "EXCELLENT"
        elif mean_similarity > 0.3:
            tier = "GOOD"
        elif mean_similarity > 0.2:
            tier = "MODERATE"
        else:
            tier = "POOR"

        print(f"\n  Results:")
        print(f"    Mean Jaccard Similarity: {mean_similarity:.4f} (±{std_similarity:.4f})")
        print(f"    Stability Tier: {tier}")
        print(f"    Time: {elapsed:.1f}s")

        return {
            "config_name": config_name,
            "min_topic_size": min_topic_size,
            "min_samples": min_samples,
            "n_neighbors": n_neighbors,
            "mean_similarity": mean_similarity,
            "std_similarity": std_similarity,
            "stability_tier": tier,
            "elapsed_time": elapsed,
            "pairwise_similarities": similarities,
        }

    def run_all(
        self, documents: List[str], configurations: List[Tuple]
    ) -> pd.DataFrame:
        """Run stability evaluation for all configurations."""
        print("\n" + "="*80)
        print(f"Testing {len(configurations)} configurations | {self.n_iterations} iterations each")
        print("="*80)

        results = []
        for idx, (mts, ms, nn, name) in enumerate(configurations, 1):
            print(f"\n── Configuration {idx}/{len(configurations)} ──")
            try:
                res = self.evaluate_stability(documents, mts, ms, nn, name)
            except Exception as exc:
                print(f"  ⚠ {name} failed: {exc}")
                res = {
                    "config_name": name,
                    "min_topic_size": mts,
                    "min_samples": ms,
                    "n_neighbors": nn,
                    "mean_similarity": 0.0,
                    "stability_tier": "FAILED",
                }
            results.append(res)

        df = pd.DataFrame(results).sort_values("mean_similarity", ascending=False)
        return df


# =============================================================================
# Visualization (focused on stability comparison only)
# =============================================================================


def visualize_stability(results_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate 2 key visualizations:
    1. Box-plot of pairwise similarity distributions
    2. Statistical significance heatmap (Wilcoxon test)
    """
    valid_df = results_df[results_df["mean_similarity"] > 0].copy()
    valid_df = valid_df.sort_values("mean_similarity", ascending=False).reset_index(drop=True)

    if len(valid_df) < 2:
        print("Need ≥2 valid configs to visualize.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    names = valid_df["config_name"].tolist()
    n = len(names)

    # Color palette
    colors = plt.cm.tab10.colors if n <= 10 else plt.cm.tab20.colors

    # =========================================================================
    # FIGURE 1: Box-plot of pairwise similarities
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [row["pairwise_similarities"] for _, row in valid_df.iterrows()]
    bp = ax.boxplot(
        box_data,
        labels=names,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=2.5),
    )
    for patch, clr in zip(bp["boxes"], colors[:n]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)

    ax.set_ylabel("Pairwise Jaccard Similarity", fontsize=12)
    ax.set_title("Configuration Stability Comparison", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "stability_comparison.png", dpi=300)
    plt.show()
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Statistical significance heatmap
    # =========================================================================
    p_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            sim_i = valid_df.iloc[i]["pairwise_similarities"]
            sim_j = valid_df.iloc[j]["pairwise_similarities"]

            min_len = min(len(sim_i), len(sim_j))
            if min_len < 10:
                continue

            a, b = np.array(sim_i[:min_len]), np.array(sim_j[:min_len])
            if np.all(a == b):
                p_matrix[i, j] = p_matrix[j, i] = 1.0
                continue

            try:
                _, p = wilcoxon(a, b)
                p_matrix[i, j] = p_matrix[j, i] = p
            except ValueError:
                p_matrix[i, j] = p_matrix[j, i] = 1.0

    masked = np.ma.masked_invalid(p_matrix)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.1, aspect="auto")
    fig.colorbar(im, ax=ax, label="p-value (Wilcoxon)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = p_matrix[i, j]
            txt = "—" if np.isnan(val) else f"{val:.3f}"
            color = "white" if (not np.isnan(val) and val < 0.05) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    ax.set_title("Statistical Significance (α = 0.05)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "stability_significance.png", dpi=300)
    plt.show()
    plt.close(fig)


# =============================================================================
# Results summary
# =============================================================================


def print_summary(results_df: pd.DataFrame) -> None:
    """Print ranked stability results."""
    valid = results_df[results_df["mean_similarity"] > 0].copy()

    if valid.empty:
        print("\n⚠ No valid configurations found.")
        return

    print("\n" + "="*80)
    print("  STABILITY RANKING")
    print("="*80)

    for idx, row in valid.iterrows():
        tier_emoji = {
            "EXCELLENT": "🟢",
            "GOOD": "🟡",
            "MODERATE": "🟠",
            "POOR": "🔴",
        }.get(row["stability_tier"], "⚪")

        print(f"\n  Rank {idx+1}: {row['config_name']}  {tier_emoji} {row['stability_tier']}")
        print(f"    min_topic_size={row['min_topic_size']}, "
              f"min_samples={row['min_samples']}, "
              f"n_neighbors={row['n_neighbors']}")
        print(f"    Jaccard Similarity: {row['mean_similarity']:.4f} "
              f"(±{row['std_similarity']:.4f})")

    print("\n" + "="*80)
    print("  RECOMMENDED CONFIGURATION")
    print("="*80)

    best = valid.iloc[0]
    print(f"\n  Use: {best['config_name']}")
    print(f"    min_topic_size = {best['min_topic_size']}")
    print(f"    min_samples    = {best['min_samples']}")
    print(f"    n_neighbors    = {best['n_neighbors']}")
    print(f"\n  Stability: {best['stability_tier']} "
          f"(Jaccard = {best['mean_similarity']:.4f})")
    print("="*80)


def save_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results to CSV and JSON."""
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    results_df.to_csv(output_path / f"stability_results_{ts}.csv", index=False)

    # JSON (top 3)
    valid = results_df[results_df["mean_similarity"] > 0].head(3)
    top3 = valid.to_dict("records")

    with open(output_path / f"top_configs_{ts}.json", "w") as f:
        json.dump(top3, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main(n_iterations: int = DEFAULT_N_ITERATIONS) -> None:
    print("\n" + "="*80)
    print("  BERTopic Stability Optimizer")
    print("="*80)

    try:
        # Load
        configs = load_configurations(CONFIGURATIONS_FILE, top_n=10)
        documents = load_documents(TOPIC_DOCS_CSV, GLOSSARY_PATH)

        # Evaluate
        checker = StabilityChecker(
            n_iterations=n_iterations,
            random_state=RANDOM_STATE,
            local_model_path=LOCAL_MODEL_PATH,
        )
        results_df = checker.run_all(documents, configs)

        # Display
        print_summary(results_df)

        # Visualize
        print("\nGenerating visualizations...")
        visualize_stability(results_df, OUTPUT_DIR)

        # Save
        save_results(results_df, OUTPUT_DIR)

        print("\n  ✓ Stability check completed successfully")

    except Exception as exc:
        import traceback
        print(f"\n  ERROR: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main(n_iterations=10)
