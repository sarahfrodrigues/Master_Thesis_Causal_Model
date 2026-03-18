"""
7_stability_check.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Any
import warnings
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import json
import time
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')

# =============================================================================
# Download NLTK stopwords (if not already present)
# =============================================================================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =============================================================================
# Configuration
# =============================================================================

# Input produced by Program 04
TOPIC_DOCS_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_docs.csv"
)
CONFIGURATIONS_FILE = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\gs_result.csv"
)
GLOSSARY_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)

# Output directories
STABILITY_OUTPUT_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\stability_optimization"
)
VISUALIZATION_DIR = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\visualizations"
)
# Local model path (if available)
LOCAL_MODEL_PATH = Path(r"C:\models\all-MiniLM-L6-v2")

# Stability check parameters
N_ITERATIONS = 10
RANDOM_STATE = 42
SIMILARITY_THRESHOLD = 0.3

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

    if "text" not in df.columns:
        df = df.rename(columns={df.columns[0]: "text"})

    with open(glossary, "r", encoding="utf-8") as f:
        abreviations = json.load(f)

    print("Expanding abbreviations...")
    df["text"] = df["text"].astype(str).apply(
        lambda x: expand_abbreviations(str(x), abreviations)
    )

    documents = df["text"].astype(str).tolist()
    documents = [doc.strip() for doc in documents if len(doc.strip()) > 10]

    print(f"Loaded {len(documents)} documents")
    return documents


def load_best_configs(config_file: Path, top_n: int = 10) -> List[Tuple]:
    """
    Load the best configurations from grid search results file.

    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Read the CSV file
    df = pd.read_csv(config_file)

    # Sort by coherence (descending) and take top N
    df_sorted = df.sort_values('coherence', ascending=False)
    df_top = df_sorted.head(top_n)

    # Create configurations list
    configurations = []
    for idx, row in df_top.iterrows():
        min_topic_size = int(row['min_topic_size'])
        min_samples = int(row['min_sample_size'])
        n_neighbors = int(row['num_neighbors'])
        coherence = float(row['coherence'])

        # Simple configuration name
        config_name = f"GS_Top{idx+1}_coh{coherence:.3f}"

        configurations.append((min_topic_size, min_samples, n_neighbors,
                               config_name))

    print(f"\nLoaded {len(configurations)} configurations from grid search:")
    for idx, (mts, ms, nn, name) in enumerate(configurations, 1):
        print(f"{idx:2d}. {name}:")
        print(f"    min_topic_size={mts}, min_samples={ms}, "
              f"n_neighbors={nn}")

    return configurations


# =============================================================================
# BERTopic Stability Checker Class
# =============================================================================


class BERTopicStabilityChecker:
    def __init__(self,
                 n_iterations: int = 10,
                 random_state: int = 42,
                 local_model_path: Path = None):
        """
        Initialize the stability checker

        Args:
            n_iterations: Number of bootstrap iterations
            random_state: Base random seed
            local_model_path: Path to local SentenceTransformer model
        """
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.local_model_path = local_model_path
        self.custom_stopwords = list(stopwords.words('english'))
        self.all_results = {}

    def _get_embedding_model(self):
        """Get embedding model - use local if available"""
        if self.local_model_path and self.local_model_path.exists():
            return SentenceTransformer(str(self.local_model_path))
        else:
            return SentenceTransformer("all-MiniLM-L6-v2")

    def create_topic_model(self, documents: List[str], seed: int,
                           min_topic_size: int, min_samples: int,
                           n_neighbors: int) -> Tuple[BERTopic, List[int]]:
        """Create a BERTopic model with given parameters"""
        embedding_model = self._get_embedding_model()

        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=3,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )

        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # Vectorizer with stopwords
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=self.custom_stopwords
        )

        # Create and fit BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            nr_topics=None,
            calculate_probabilities=False,
            verbose=False
        )

        topics, _ = topic_model.fit_transform(documents)
        return topic_model, topics

    def extract_topic_keywords(self, topic_model: BERTopic,
                               top_n: int = 10) -> Dict[int, Set[str]]:
        """Extract topic keywords as sets for Jaccard similarity calculation"""
        topic_keywords = {}

        all_topics = topic_model.get_topics()

        for topic_num, word_scores in all_topics.items():
            if topic_num != -1:  # Skip outlier topic
                topic_words = [word for word, _ in word_scores[:top_n]]
                cleaned_words = set([str(word).lower().strip() for word in topic_words])
                topic_keywords[topic_num] = cleaned_words

        return topic_keywords

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def bootstrap_sampling(self, documents: List[str],
                           sample_size: float = 0.8) -> List[str]:
        """Create bootstrap sample from documents"""
        n_samples = int(len(documents) * sample_size)
        indices = np.random.choice(len(documents), n_samples, replace=True)
        return [documents[i] for i in indices]

    def evaluate_configuration(self, documents: List[str],
                               min_topic_size: int, min_samples: int,
                               n_neighbors: int,
                               config_name: str) -> Dict[str, Any]:
        """Evaluate stability for a single configuration"""

        print(f"\n{'='*60}")
        print(f"Evaluating Configuration: {config_name}")
        print(f"min_topic_size={min_topic_size}, min_samples={min_samples}, "
              f"n_neighbors={n_neighbors}")
        print(f"{'='*60}")

        start_time = time.time()

        all_topic_keywords = []
        topic_counts = []
        outlier_rates = []

        # Run multiple iterations
        for i in range(self.n_iterations):
            # Create bootstrap sample
            bootstrap_docs = self.bootstrap_sampling(documents)

            # Create topic model with different seed
            seed = self.random_state + i
            topic_model, topics = self.create_topic_model(
                bootstrap_docs, seed, min_topic_size, min_samples, n_neighbors
            )

            # Extract topic keywords
            topic_keywords = self.extract_topic_keywords(topic_model)
            all_topic_keywords.append(topic_keywords)
            topic_counts.append(len(topic_keywords))

            # Calculate outlier rate
            n_outliers = sum(1 for t in topics if t == -1)
            outlier_rate = n_outliers / len(topics) if len(topics) > 0 else 0
            outlier_rates.append(outlier_rate)

            # Print progress
            if (i + 1) % 2 == 0 or i == 0 or i == self.n_iterations - 1:
                print(f"  Iteration {i+1}/{self.n_iterations}: "
                      f"{len(topic_keywords)} topics, "
                      f"{outlier_rate:.1%} outliers")

        # Calculate pairwise Jaccard similarities
        pairwise_similarities = []

        for i in range(self.n_iterations):
            for j in range(i + 1, self.n_iterations):
                if all_topic_keywords[i] and all_topic_keywords[j]:
                    # Calculate max similarity for each topic in i to any in j
                    max_similarities = []
                    for topic_i, keywords_i in all_topic_keywords[i].items():
                        best_similarity = 0
                        for topic_j, keywords_j in all_topic_keywords[j].items():
                            similarity = self.jaccard_similarity(keywords_i, keywords_j)
                            if similarity > best_similarity:
                                best_similarity = similarity
                        max_similarities.append(best_similarity)

                    if max_similarities:
                        avg_max_similarity = np.mean(max_similarities)
                        pairwise_similarities.append(avg_max_similarity)

        # Calculate metrics
        if pairwise_similarities:
            mean_similarity = np.mean(pairwise_similarities)
            std_similarity = np.std(pairwise_similarities)
            similarity_cv = std_similarity / mean_similarity if mean_similarity > 0 else 0
        else:
            mean_similarity = 0
            std_similarity = 0
            similarity_cv = 0

        # Topic count metrics
        mean_topic_count = np.mean(topic_counts) if topic_counts else 0
        std_topic_count = np.std(topic_counts) if topic_counts else 0
        topic_count_cv = std_topic_count / mean_topic_count if mean_topic_count > 0 else 0

        # Outlier metrics
        mean_outlier_rate = np.mean(outlier_rates) if outlier_rates else 0

        # Calculate stability score (weighted combination)
        # Higher similarity, more topics, lower outlier rate = better
        stability_score = (
            mean_similarity * 0.5 +  # Similarity is most important
            (1 - topic_count_cv) * 0.3 +  # Consistent topic count
            (1 - mean_outlier_rate) * 0.2  # Few outliers
        )

        elapsed_time = time.time() - start_time

        # Store results
        results = {
            'config_name': config_name,
            'min_topic_size': min_topic_size,
            'min_samples': min_samples,
            'n_neighbors': n_neighbors,
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'similarity_cv': similarity_cv,
            'mean_topic_count': mean_topic_count,
            'std_topic_count': std_topic_count,
            'topic_count_cv': topic_count_cv,
            'mean_outlier_rate': mean_outlier_rate,
            'stability_score': stability_score,
            'elapsed_time': elapsed_time,
            'topic_counts': topic_counts,
            'outlier_rates': outlier_rates,
            'pairwise_similarities': pairwise_similarities
        }

        # Print summary
        print(f"\n Results for {config_name}:")
        print(f"  • Mean Jaccard Similarity: {mean_similarity:.4f} (±{std_similarity:.4f})")
        print(f"  • Mean Topic Count: {mean_topic_count:.1f} (±{std_topic_count:.1f})")
        print(f"  • Mean Outlier Rate: {mean_outlier_rate:.1%}")
        print(f"  • Stability Score: {stability_score:.4f}")
        print(f"  • Time: {elapsed_time:.1f}s")

        # Stability interpretation
        if mean_similarity > 0.5:
            stability = "EXCELLENT"
        elif mean_similarity > 0.3:
            stability = "GOOD"
        elif mean_similarity > 0.2:
            stability = "MODERATE"
        else:
            stability = "POOR"

        print(f"  • Stability: {stability}")

        return results

    def run_all_configurations(self, documents: List[str],
                               configurations: List[Tuple]) -> pd.DataFrame:
        """Run stability evaluation for all configurations"""

        print("\n" + "="*80)
        print(f"Testing {len(configurations)} configurations")
        print("="*80)

        all_results = []

        for idx, (min_topic_size, min_samples, n_neighbors, config_name) in enumerate(configurations):
            print(f"\n\nConfiguration {idx+1}/{len(configurations)}")

            try:
                results = self.evaluate_configuration(
                    documents, min_topic_size, min_samples,
                    n_neighbors, config_name
                )
                all_results.append(results)

                # Store in dictionary
                self.all_results[config_name] = results

            except Exception as e:
                print(f" Error evaluating {config_name}: {str(e)}")
                # Create a failed result entry
                failed_results = {
                    'config_name': config_name,
                    'min_topic_size': min_topic_size,
                    'min_samples': min_samples,
                    'n_neighbors': n_neighbors
                }
                all_results.append(failed_results)
                self.all_results[config_name] = failed_results

        # Create DataFrame from results
        results_df = pd.DataFrame(all_results)

        # Sort by stability score (descending)
        results_df = results_df.sort_values('stability_score', ascending=False)

        return results_df

    def identify_best_configuration(self, results_df: pd.DataFrame) -> Dict:
        """Identify the best configuration based on multiple criteria"""

        if results_df.empty:
            return {}

        # Filter out configurations with errors
        valid_results = results_df[results_df['stability_score'] > 0].copy()

        if valid_results.empty:
            print("\n No valid configurations found")
            return {}

        # Calculate rankings for different metrics
        valid_results['similarity_rank'] = valid_results['mean_similarity'].rank(ascending=False)
        valid_results['topic_count_rank'] = valid_results['mean_topic_count'].rank(ascending=False)
        valid_results['outlier_rank'] = valid_results['mean_outlier_rate'].rank(ascending=True)
        valid_results['stability_cv_rank'] = valid_results['similarity_cv'].rank(ascending=True)

        # Calculate combined rank (lower is better)
        valid_results['combined_rank'] = (
            valid_results['similarity_rank'] * 0.4 +
            valid_results['topic_count_rank'] * 0.2 +
            valid_results['outlier_rank'] * 0.3 +
            valid_results['stability_cv_rank'] * 0.1
        )

        # Find best by combined rank
        best_by_rank = valid_results.loc[valid_results['combined_rank'].idxmin()]

        # Also get best by stability score (original metric)
        best_by_score = valid_results.iloc[0]

        # Get top 3 configurations
        top_configs = valid_results.head(3)

        # Prepare best configuration info
        best_config = {
            'by_stability_score': best_by_score.to_dict(),
            'by_combined_rank': best_by_rank.to_dict(),
            'top_3': top_configs.to_dict('records'),
            'all_configs': valid_results.to_dict('records')
        }

        return best_config

    def visualize_results(self, results_df: pd.DataFrame, output_path: Path):
        """
        Thesis-quality configuration comparison with statistical analysis
        """

        valid_df = results_df[results_df['stability_score'] > 0].copy()
        valid_df = valid_df.sort_values("stability_score", ascending=False)

        if len(valid_df) < 2:
            print("Not enough configurations for comparison.")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # FIGURE 1 — Stability score with uncertainty
        # =====================================================

        plt.figure(figsize=(9, 5))

        means = valid_df["stability_score"]
        stds = valid_df["std_similarity"]

        plt.errorbar(
            valid_df["config_name"],
            means,
            yerr=stds,
            fmt="o",
            capsize=5,
            linewidth=2
        )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Stability score")
        plt.title("Configuration stability comparison (mean ± std)")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "stability_comparison_ci.png", dpi=300)
        plt.show()

        # =====================================================
        # FIGURE 2 — Statistical significance (Wilcoxon)
        # =====================================================

        config_names = valid_df["config_name"].tolist()
        n = len(config_names)

        significance = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim_i = valid_df.iloc[i]["pairwise_similarities"]
                sim_j = valid_df.iloc[j]["pairwise_similarities"]

                if len(sim_i) > 3 and len(sim_j) > 3:
                    stat, p = wilcoxon(sim_i[:min(len(sim_i), len(sim_j))],
                                       sim_j[:min(len(sim_i), len(sim_j))])
                    significance[i, j] = p
                    significance[j, i] = p

        plt.figure(figsize=(8, 6))
        plt.imshow(significance, aspect="auto")
        plt.colorbar(label="p-value")

        plt.xticks(range(n), config_names, rotation=45, ha="right")
        plt.yticks(range(n), config_names)
        plt.title("Pairwise statistical significance (Wilcoxon test)")

        plt.tight_layout()
        plt.savefig(output_path / "stability_significance_matrix.png", dpi=300)
        plt.show()

        # =====================================================
        # FIGURE 3 — Trade-off analysis
        # =====================================================

        plt.figure(figsize=(7, 5))

        sizes = valid_df["mean_topic_count"] * 10

        plt.scatter(
            valid_df["mean_similarity"],
            valid_df["mean_outlier_rate"],
            s=sizes,
            alpha=0.7
        )

        for _, row in valid_df.iterrows():
            plt.text(
                row["mean_similarity"] + 0.005,
                row["mean_outlier_rate"],
                row["config_name"],
                fontsize=8
            )

        plt.xlabel("Mean topic similarity")
        plt.ylabel("Mean outlier rate")
        plt.title("Configuration trade-off analysis")

        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "configuration_tradeoff.png", dpi=300)
        plt.show()

    def print_best_configuration_summary(self, best_config: Dict):
        """Print detailed summary of the best configuration"""

        if not best_config:
            print("\nNo valid configurations found!")
            return

        best_by_score = best_config['by_stability_score']
        best_by_rank = best_config['by_combined_rank']

        print("\n" + "="*80)
        print("BEST CONFIGURATION IDENTIFIED")
        print("="*80)

        # Check if both methods agree
        if best_by_score['config_name'] == best_by_rank['config_name']:
            print(f"\nBOTH METHODS AGREE: {best_by_score['config_name']}")
        else:
            print("\n DIFFERENT BEST CONFIGURATIONS IDENTIFIED:")
            print(f"   • By Stability Score: {best_by_score['config_name']}")
            print(f"   • By Combined Rank: {best_by_rank['config_name']}")
            print(f"\nUse {best_by_score['config_name']} for max stability")

        print("\n" + "-"*80)
        print("TOP CONFIGURATION DETAILS:")
        print("-"*80)

        # Show top 3 configurations
        for idx, config in enumerate(best_config['top_3'], 1):
            print(f"\n#{idx}: {config['config_name']}")
            print(f"Parameters: min_topic_size={config['min_topic_size']}, "
                  f"min_samples={config['min_samples']},"
                  f" n_neighbors={config['n_neighbors']}")
            print(f"Mean Jaccard Similarity: {config['mean_similarity']:.4f} "
                  f"(±{config['std_similarity']:.4f})")
            print(f"Mean Topic Count: {config['mean_topic_count']:.1f} "
                  f"(±{config['std_topic_count']:.1f})")
            print(f"Outlier Rate: {config['mean_outlier_rate']:.1%}")
            print(f"Stability Score: {config['stability_score']:.4f}")

            # Stability interpretation
            similarity = config['mean_similarity']
            if similarity > 0.4:
                stability = "GOOD"
                color = "🟢"
            elif similarity > 0.3:
                stability = "MODERATE"
                color = "🟡"
            else:
                stability = "POOR"
                color = "🔴"

            print(f"   Stability Level: {color} {stability}")

        print("\n" + "-"*80)
        print("RECOMMENDED PARAMETERS")
        print("-"*80)

        best_config_details = best_config['top_3'][0]

        print(f"  min_topic_size = {best_config_details['min_topic_size']}")
        print(f"  min_samples = {best_config_details['min_samples']}")
        print(f"  n_neighbors = {best_config_details['n_neighbors']}")

        print("\nExpected performance:")
        print(f"Topic similarity: {best_config_details['mean_similarity']:.1%}")
        print(f"N of topics: ~{best_config_details['mean_topic_count']:.0f}")
        print(f"Outlier docs: {best_config_details['mean_outlier_rate']:.1%}")

        # Final recommendation
        similarity = best_config_details['mean_similarity']
        if similarity > 0.5:
            print("\n EXCELLENT STABILITY")
        elif similarity > 0.3:
            print("\n GOOD STABILITY")
        else:
            print("\n MODERATE STABILITY")

        print("\n" + "="*80)

    def save_all_results(self, results_df: pd.DataFrame,
                         best_config: Dict, output_path: Path):
        """Save all results to files"""

        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results DataFrame
        results_df.to_csv(output_path / f'stability_results_{timestamp}.csv',
                          index=False)

        # Save best configuration
        if best_config:
            config_file = (
                output_path / f'best_configuration_{timestamp}.json'
            )
            with open(config_file, 'w') as f:
                json.dump(best_config, f, indent=2, default=str)

        # Save summary report
        self._save_summary_report(results_df, best_config, output_path,
                                  timestamp)

        print(f"\nResults saved to: {output_path}")

    def _save_summary_report(self, results_df: pd.DataFrame, best_config: Dict,
                             output_path: Path, timestamp: str):
        """Save a text summary report"""

        report_path = output_path / f'summary_report_{timestamp}.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BERTopic Configuration Stability Optimization Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-d %H:%M:%S')}\n")
            f.write(f"Dataset size: {len(self.documents):,} documents\n")
            f.write(f"Configurations tested: {len(results_df)}\n")
            f.write(f"Iterations per config: {self.n_iterations}\n\n")

            f.write("="*80 + "\n")
            f.write("TOP 3 CONFIGURATIONS\n")
            f.write("="*80 + "\n\n")

            if best_config and 'top_3' in best_config:
                for idx, config in enumerate(best_config['top_3'], 1):
                    f.write(f"{idx}. {config['config_name']}\n")
                    f.write(f" min_topic_size={config['min_topic_size']}\n")
                    f.write(f"min_samples={config['min_samples']} \n")
                    f.write(f"n_neighbors={config['n_neighbors']}\n")
                    f.write(f"Mean Similarity: {config['mean_similarity']:.4f}\n")
                    f.write(f"Mean Topics: {config['mean_topic_count']:.1f}\n")
                    f.write(f"Outlier Rate: {config['mean_outlier_rate']:.1%}")
                    f.write(f"Stability: {config['stability_score']:.4f}\n")

            f.write("="*80 + "\n")
            f.write("RECOMMENDED PARAMETERS FOR PRODUCTION\n")
            f.write("="*80 + "\n\n")

            if best_config and 'top_3' in best_config and best_config['top_3']:
                best = best_config['top_3'][0]
                f.write(f"min_topic_size = {best['min_topic_size']}\n")
                f.write(f"min_samples = {best['min_samples']}\n")
                f.write(f"n_neighbors = {best['n_neighbors']}\n\n")

                f.write("Expected Performance:\n")
                f.write(f"Topic Stability: {best['mean_similarity']:.1%}\n")
                f.write(f"Number of Topics: ~{best['mean_topic_count']:.0f}\n")
                f.write(f"Outlier Docs: {best['mean_outlier_rate']:.1%}\n")

                # Interpretation
                similarity = best['mean_similarity']
                if similarity > 0.5:
                    f.write("Stability Level: EXCELLENT\n")
                    f.write("Highly reproducible topics.\n")
                elif similarity > 0.3:
                    f.write("Stability Level: GOOD\n")
                    f.write("Reasonably stable topics.\n")
                else:
                    f.write("Stability Level: MODERATE/POOR\n")
                    f.write("Consider further parameter optimization.\n")

        print(f"Summary report saved to: {report_path}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    print("\n" + "="*80)
    print("BERTopic Configuration Stability Optimizer")
    print("="*80)

    try:
        # Load configurations from grid search results
        CONFIGURATIONS = load_best_configs(CONFIGURATIONS_FILE, top_n=10)

        if not CONFIGURATIONS:
            print("\nERROR: No configurations loaded from grid search.")
            return

        print(f"\nConfigurations to test: {len(CONFIGURATIONS)}")
        print("="*80)

        # Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        documents = load_and_preprocess(TOPIC_DOCS_CSV, GLOSSARY_PATH)

        # Initialize stability checker
        print("\n2. Initializing stability checker...")
        stability_checker = BERTopicStabilityChecker(
            n_iterations=N_ITERATIONS,
            random_state=RANDOM_STATE,
            local_model_path=LOCAL_MODEL_PATH
        )
        stability_checker.documents = documents  # Store for reporting

        # Run all configurations
        print("\n3. Running stability evaluation for all configurations...")

        results_df = stability_checker.run_all_configurations(documents,
                                                              CONFIGURATIONS)

        if results_df.empty:
            print("\n No configurations could be evaluated successfully.")
            return

        # Identify best configuration
        print("\n4. Analyzing results and identifying best configuration...")
        best_config = stability_checker.identify_best_configuration(results_df)

        # Print summary
        stability_checker.print_best_configuration_summary(best_config)

        # Visualize results
        print("\n5. Generating comparison visualizations...")
        stability_checker.visualize_results(results_df, VISUALIZATION_DIR)

        # Save all results
        print("\n6. Saving detailed results...")
        stability_checker.save_all_results(results_df, best_config,
                                           STABILITY_OUTPUT_DIR)

        print("[DONE] Program 07 completed successfully")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"Please ensure {CONFIGURATIONS_FILE} exists.")
    except KeyboardInterrupt:
        print("\n\n Optimization interrupted by user.")
    except Exception as e:
        print(f"\nERROR during optimization: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# Quick Run Function (for testing specific configurations)
# =============================================================================

def quick_test():
    """Quick test with fewer iterations for faster results"""
    print("\n" + "="*80)
    print("QUICK TEST MODE")
    print("="*80)
    print("Running with N_ITERATIONS = 3 for faster results")
    print("="*80)

    global N_ITERATIONS
    original_iterations = N_ITERATIONS
    N_ITERATIONS = 3  # Reduced for quick testing

    try:
        main()
    finally:
        N_ITERATIONS = original_iterations  # Restore original value


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\nChoose run mode:")
    print("1. Full optimization (recommended, ~30-60 minutes)")
    print("2. Quick test (faster, less accurate)")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ")

    if choice == '1':
        main()
    elif choice == '2':
        quick_test()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")
