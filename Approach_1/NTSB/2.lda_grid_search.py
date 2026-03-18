# -*- coding: utf-8 -*-
"""
LDA Model Grid Search for NTSB Aviation Safety Dataset
Purpose: Grid search over LDA hyperparameters (num_topics, alpha) to optimize
coherence (c_v) and perplexity; export results to CSV and plot coherence trends
"""
# Import core libraries
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION (EDIT THESE PATHS/VALUES AS NEEDED)
# -----------------------------------------------------------------------------
RANDOM_STATE = 100  # Reproducibility
PASSES = 10         # LDA training passes
CHUNKSIZE = 100     # LDA batch training size

# Hyperparameter grid for search
TOPIC_RANGE = [6, 7, 8, 9, 11, 12]
ALPHA_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, "symmetric", "asymmetric"]

# File paths
DATA_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_ALL.csv"
)
OUTPUT_CSV_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_1\NTSB\lda_grid_search.csv"
)
PLOT_SAVE_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_1\NTSB\lda_grid_search_coherence_plot.png"
)

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------


def load_preprocess_data(data_path: str) -> list:
    """
    Load NTSB CSV data and convert token strings to token lists
    """
    print("=== Loading and Preprocessing Data ===")
    df = pd.read_csv(data_path)
    # Convert token string column to list of token lists
    df["tokens_final"] = df["tokens_final_str"].astype(str).str.split()
    text_corpus = df["tokens_final"].tolist()
    print(f"Loaded {len(text_corpus)} documents | Preprocessing complete\n")
    return text_corpus

# -----------------------------------------------------------------------------
# Gensim Dictionary & BOW Corpus Construction
# -----------------------------------------------------------------------------


def build_gensim_corpus(texts: list) -> tuple[corpora.Dictionary, list]:
    """
    Build Gensim vocabulary dictionary and Bag-of-Words (BOW) corpus
    """
    print("=== Building Gensim Dictionary & BOW Corpus ===")
    # Create word-to-id mapping dictionary
    id2word = corpora.Dictionary(texts)
    # Convert corpus to BOW (word ID, word frequency) tuples
    bow_corpus = [id2word.doc2bow(text) for text in texts]
    print(f"Dictionary size: {len(id2word)} unique tokens | BOW corpus built")
    return id2word, bow_corpus

# -----------------------------------------------------------------------------
# LDA Model Training & Evaluation
# -----------------------------------------------------------------------------


def train_evaluate_lda(
    corpus: list,
    id2word: corpora.Dictionary,
    texts: list,
    num_topics: int,
    alpha: float | str
) -> tuple[float, float]:
    """
    Train a single LDA model and return coherence (c_v) and log perplexity
    """
    # Train LDA model with specified hyperparameters
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        alpha=alpha,
        random_state=RANDOM_STATE,
        update_every=1,
        chunksize=CHUNKSIZE,
        passes=PASSES
    )
    # Calculate c_v coherence (higher = better topic interpretability)
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=id2word,
        coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()
    # Calculate log perplexity (lower = better model fit to corpus)
    log_perplexity = lda_model.log_perplexity(corpus)
    return coherence_score, log_perplexity

# -----------------------------------------------------------------------------
# Grid Search Execution
# -----------------------------------------------------------------------------


def run_lda_grid_search(
    corpus: list,
    id2word: corpora.Dictionary,
    texts: list,
    topic_range: list,
    alpha_vals: list
) -> pd.DataFrame:
    """
    Run full grid search over num_topics and alpha; store all results
    """
    print("=== Starting LDA Hyperparameter Grid Search ===")
    # Initialize results storage
    results = {
        "num_topics": [],
        "alpha": [],
        "coherence_cv": [],
        "perplexity_log": []
    }
    # Calculate total runs for progress bar
    total_combinations = len(topic_range) * len(alpha_vals)
    print(f"Total parameter combinations to test: {total_combinations}\n")

    # Iterate over all hyperparameter combinations
    with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
        for n_topics in topic_range:
            for alpha in alpha_vals:
                # Train/evaluate model and store results
                coh, perp = train_evaluate_lda(corpus, id2word, texts,
                                               n_topics, alpha)
                results["num_topics"].append(n_topics)
                results["alpha"].append(alpha)
                results["coherence_cv"].append(coh)
                results["perplexity_log"].append(perp)
                pbar.update(1)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\nGrid search complete")
    return results_df

# -----------------------------------------------------------------------------
# Coherence Plot Generation & Saving
# -----------------------------------------------------------------------------


def plot_coherence_trends(
    results_df: pd.DataFrame,
    save_path: str,
    alpha_vals: list
) -> None:
    """
    Plot Coherence (c_v) vs Alpha (grouped by number of topics) and save plot
    """
    print("\n=== Generating Coherence Trend Plot ===")
    # Set plot style and size
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))

    # Standardize alpha labels to strings for consistent plotting
    alpha_labels = [str(a) for a in alpha_vals]
    # Plot each topic count as a separate line
    for n_topic in sorted(results_df["num_topics"].unique()):
        # Filter results for current topic count
        subset = results_df[results_df["num_topics"] == n_topic].copy()
        subset["alpha_str"] = subset["alpha"].astype(str)
        # Order subset by original alpha value order
        subset["alpha_order"] = subset["alpha_str"].apply(lambda x: alpha_labels.index(x))
        subset = subset.sort_values("alpha_order")
        # Plot line with markers
        ax.plot(
            subset["alpha_order"],
            subset["coherence_cv"],
            marker="o",
            markersize=8,
            linewidth=2,
            label=f"{n_topic} Topics"
        )

    # Customize plot labels/ticks/title
    ax.set_xticks(range(len(alpha_labels)))
    ax.set_xticklabels(alpha_labels, rotation=45, fontsize=12)
    ax.set_xlabel("Alpha Parameter", fontsize=14, fontweight="bold")
    ax.set_ylabel("Coherence Score (c_v)", fontsize=14, fontweight="bold")
    ax.set_title("LDA Coherence (c_v) vs Alpha (Grouped by Number of Topics)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(alpha=0.3)

    # Save high-resolution plot (300 DPI) and close figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------


def main():
    """Main function to run the entire LDA grid search pipeline"""
    try:
        # Step 1: Load and preprocess data
        text_corpus = load_preprocess_data(DATA_PATH)
        # Step 2: Build Gensim dictionary and BOW corpus
        id2word, bow_corpus = build_gensim_corpus(text_corpus)
        # Step 3: Run grid search
        grid_results = run_lda_grid_search(bow_corpus, id2word, text_corpus,
                                           TOPIC_RANGE, ALPHA_VALUES)
        # Step 4: Save results to CSV
        grid_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
        print(f"\nGrid search results saved to: {OUTPUT_CSV_PATH}")
        # Step 5: Generate and save coherence plot
        plot_coherence_trends(grid_results, PLOT_SAVE_PATH, ALPHA_VALUES)
        # Step 6: Print BEST model parameters (sorted by highest coherence)
        best_model = grid_results.sort_values("coherence_cv",
                                              ascending=False).iloc[0]
        print("\n=== BEST LDA MODEL PARAMETERS (HIGHEST COHERENCE) ===")
        print(f"Number of Topics: {int(best_model['num_topics'])}")
        print(f"Alpha Value: {best_model['alpha']}")
        print(f"Coherence (c_v): {best_model['coherence_cv']:.4f}")
        print(f"Log Perplexity: {best_model['perplexity_log']:.4f}")
        print("\n=== LDA Grid Search Pipeline Complete ===")

    except Exception as e:
        # Error handling with clear message
        print(f"\n PIPELINE FAILED: {str(e)}")
        raise  # Re-raise error for full traceback (debugging)


# Run main function only when script is executed directly (not imported)
if __name__ == "__main__":
    main()
