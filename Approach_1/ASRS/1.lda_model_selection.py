"""
1. lda_model_selection.py
--------------------------------------------------
Evaluates LDA models using coherence (c_v) and
perplexity across different numbers of topics.

Output:
- Saved plot: lda_coherence_perplexity.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from tqdm import tqdm


# =====================================================
# Configuration
# =====================================================

DATA_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_PREPROCESSED.csv"
)
OUTPUT_DIR = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_1\ASRS"
)

RANDOM_STATE = 100
TOPIC_RANGE = range(2, 30)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # =====================================================
    # Load and prepare data
    # =====================================================

    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    df["tokens_final"] = (
        df["tokens_final_str"]
        .astype(str)
        .str.split()
    )

    texts = df["tokens_final"].tolist()

    # =====================================================
    # Dictionary and corpus
    # =====================================================

    print("Creating dictionary and corpus...")

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    # =====================================================
    # LDA evaluation
    # =====================================================

    coherence_scores = []
    perplexity_scores = []

    print("Training LDA models...")

    for num_topics in tqdm(TOPIC_RANGE):

        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=RANDOM_STATE,
            chunksize=100,
            passes=10,
            update_every=1,
            alpha="auto",
            per_word_topics=True
        )

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v"
        )

        coherence_scores.append(coherence_model.get_coherence())
        perplexity_scores.append(lda_model.log_perplexity(corpus))

    # =====================================================
    # Plot results
    # =====================================================

    print("Saving evaluation plot...")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Coherence axis
    ax1.set_xlabel("Number of Topics")
    ax1.set_ylabel("Coherence (c_v)", color="tab:blue")
    ax1.set_xticks(range(2, 31, 2))
    ax1.plot(
        list(TOPIC_RANGE),
        coherence_scores,
        marker="o",
        color="tab:blue"
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Perplexity axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Perplexity (log)", color="tab:red")
    ax2.plot(
        list(TOPIC_RANGE),
        perplexity_scores,
        marker="x",
        linestyle="--",
        color="tab:red"
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("LDA Model Evaluation: Coherence and Perplexity")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "lda_coherence_perplexity.png"
    )

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved → {plot_path}")
    print("LDA model selection completed successfully.")


if __name__ == "__main__":
    main()
