# -*- coding: utf-8 -*-
"""
Final LDA Model Training for NTSB Aviation Safety Dataset
Purpose: Train the optimized LDA model (8 topics, alpha=0.3)
outputs: Coherence score, topic keywords, representative docs,
pyLDAvis, topic dists, heatmap, binary matrix
Outputs: .txt, .csv, .html, .png files to the specified NTSB output directory
"""
# Import core libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_ALL.csv"
)
OUTPUT_DIR = r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_1\NTSB"
NUM_TOPICS = 9                 # Optimized number of topics
ALPHA = .3                     # Optimized alpha parameter
RANDOM_STATE = 100              # Reproducibility
CHUNKSIZE = 100                 # LDA batch training size
PASSES = 10                     # LDA training passes
TOPN_KEYWORDS = 10              # Top N keywords to extract per topic
ALPHA_THRESHOLD = 0.5  # Alpha for binary matrix (mean + 0.5*std)
PYLDAVIS_MDS = "mmds"           # MDS algorithm for pyLDAvis visualization

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------


def load_preprocess_data(data_path: str) -> tuple[pd.DataFrame, list]:
    """
    Load NTSB CSV data and convert token strings to token lists
    """
    print("=== Step 1/9: Loading and Preprocessing Data ===")
    df = pd.read_csv(data_path)
    # Convert token string column to list of token lists
    df["tokens_final"] = df["tokens_final_str"].astype(str).str.split()
    text_corpus = df["tokens_final"].tolist()
    print(f"Loaded {len(text_corpus)} documents | Preprocessing complete\n")
    return df, text_corpus

# -----------------------------------------------------------------------------
# Gensim Dictionary & BOW Corpus Construction
# -----------------------------------------------------------------------------


def build_gensim_assets(texts: list) -> tuple[corpora.Dictionary, list]:
    """
    Build Gensim vocabulary dictionary and Bag-of-Words (BOW) corpus
    """
    print("=== Step 2/9: Building Gensim Dictionary & BOW Corpus ===")
    id2word = corpora.Dictionary(texts)
    bow_corpus = [id2word.doc2bow(text) for text in texts]
    print(f"Dictionary size: {len(id2word)} unique tokens | BOW corpus built")
    return id2word, bow_corpus

# -----------------------------------------------------------------------------
# Train Optimized LDA Model
# -----------------------------------------------------------------------------


def train_optimized_lda(
    corpus: list,
    id2word: corpora.Dictionary,
    num_topics: int,
    random_state: int
) -> LdaModel:
    """
    Train the final optimized LDA model with pre-determined hyperparameters
    """
    print("=== Step 3/9: Training Optimized LDA Model ===")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=CHUNKSIZE,
        passes=PASSES,
        alpha=ALPHA,
        update_every=1
    )
    print(f"LDA model trained | {num_topics} topics | alpha={ALPHA}\n")
    return lda_model

# -----------------------------------------------------------------------------
# Calculate & Save Coherence Score (c_v)
# -----------------------------------------------------------------------------


def calculate_save_coherence(
    model: LdaModel,
    texts: list,
    id2word: corpora.Dictionary,
    output_dir: str
) -> float:
    """
    Calculate c_v coherence score and save to coherence.txt
    """
    print("=== Step 4/9: Calculating Coherence Score (c_v) ===")
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=id2word,
        coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()
    # Save to text file
    with open(os.path.join(output_dir, "coherence.txt"), "w") as f:
        f.write(f"Coherence score (c_v): {coherence_score:.4f}")
    print(f"Coherence score: {coherence_score:.4f} | Saved to coherence.txt\n")
    return coherence_score

# -----------------------------------------------------------------------------
# Extract Keywords + Representative Docs and Save to Single Excel
# -----------------------------------------------------------------------------


def save_topics_summary_excel(
    model: LdaModel,
    corpus: list,
    df: pd.DataFrame,
    num_topics: int,
    topn: int,
    output_dir: str
) -> pd.DataFrame:
    """
    Create a single Excel file containing:
    - topic_id
    - top keywords
    - most representative document
    """

    print("=== Step 5/9: Creating Topic Summary Excel File ===")

    topics_summary = []

    # Map each document to its topic probabilities
    topic_docs = {topic_id: [] for topic_id in range(num_topics)}

    for doc_id, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow):
            topic_docs[topic_id].append((doc_id, prob))

    for topic_id in range(num_topics):

        # 1. Get Top Keywords
        top_words = model.show_topic(topic_id, topn=topn)
        keywords = ", ".join([word for word, _ in top_words])

        # 2. Get Representative Document
        docs = topic_docs[topic_id]
        best_doc_id = max(docs, key=lambda x: x[1])[0]
        rep_doc_text = " ".join(df.iloc[best_doc_id]["tokens_final"])

        topics_summary.append({
            "topic_id": topic_id,
            "keywords": keywords,
            "representative_document": rep_doc_text
        })

    df_summary = pd.DataFrame(topics_summary)

    # Save as Excel
    csv_path = os.path.join(output_dir, "topics_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    print("Topic summary saved to topics_summary.csv\n")

    return df_summary

# -----------------------------------------------------------------------------
# Generate & Save pyLDAvis Visualization
# -----------------------------------------------------------------------------


def generate_save_pyldavis(
    model: LdaModel,
    corpus: list,
    id2word: corpora.Dictionary,
    output_dir: str,
    mds: str
) -> None:
    """
    Generate interactive pyLDAvis visualization and save
    """
    print("=== Step 7/9: Generating pyLDAvis Visualization ===")
    # Prepare pyLDAvis data
    lda_vis = gensimvis.prepare(model, corpus, id2word, mds=mds)
    # Save interactive HTML
    pyLDAvis.save_html(lda_vis,
                       os.path.join(output_dir, "lda_visualization.html"))
    print("Interactive pyLDAvis saved | Open in a browser\n")

# -----------------------------------------------------------------------------
# Generate & Save Document-Topic Distributions
# -----------------------------------------------------------------------------


def generate_save_topic_dists(
    model: LdaModel,
    corpus: list,
    num_topics: int,
    output_dir: str
) -> pd.DataFrame:
    """
    Generate document-topic probability distributions and save to CSV
    """
    print("=== Step 8/9: Generating Document-Topic Distributions ===")
    # Calculate topic probabilities for each document
    topic_distributions = [
        [prob for _, prob in sorted(
            model.get_document_topics(bow, minimum_probability=0),
            key=lambda x: x[0]
        )]
        for bow in corpus
    ]
    # Create DataFrame with named topic columns
    df_topic_dist = pd.DataFrame(
        topic_distributions,
        columns=[f"Topic_{i}" for i in range(num_topics)]
    )
    # Save to CSV
    df_topic_dist.to_csv(
        os.path.join(output_dir, "document_topic_distributions.csv"),
        index=False
    )
    print("Document-topic distributions saved\n")
    return df_topic_dist

# -----------------------------------------------------------------------------
# Generate Heatmap + Adaptive Binary Document-Topic Matrix
# -----------------------------------------------------------------------------


def generate_heatmap_binary_matrix(
    df_topic_dist: pd.DataFrame,
    num_topics: int,
    output_dir: str,
    alpha: float
) -> pd.DataFrame:
    """
    1. Generate topic correlation heatmap (saved as PNG)
    2. Create adaptive binary document-topic matrix
    3. Save binary matrix to CSV
    """
    print("=== Step 9/9: Generating Heatmap & Adaptive Binary Matrix ===")
    # 1. Topic Correlation Heatmap
    corr_matrix = df_topic_dist.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title(f"Topic Correlation Heatmap ({num_topics} Topics)", fontsize=14,
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "topic_correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 2. Adaptive Binary Document-Topic Matrix
    thresholds = df_topic_dist.mean(axis=1) + alpha * df_topic_dist.std(axis=1)
    binary_matrix = df_topic_dist.gt(thresholds, axis=0).astype(int)
    # Save binary matrix
    binary_matrix.to_csv(
        os.path.join(output_dir, "binary_document_topics_adaptive.csv"),
        index=False
    )

    print("Topic correlation heatmap saved")
    print(f"Adaptive binary matrix saved | Threshold: mean + {alpha}*std\n")
    return binary_matrix

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------


def main():
    """Main function to run the entire final LDA model pipeline (9 steps)"""
    try:
        # Core pipeline steps
        df, text_corpus = load_preprocess_data(DATA_PATH)
        id2word, bow_corpus = build_gensim_assets(text_corpus)
        lda_model = train_optimized_lda(bow_corpus, id2word, NUM_TOPICS,
                                        RANDOM_STATE)
        calculate_save_coherence(lda_model, text_corpus, id2word, OUTPUT_DIR)
        save_topics_summary_excel(
                                  lda_model,
                                  bow_corpus,
                                  df,
                                  NUM_TOPICS,
                                  TOPN_KEYWORDS,
                                  OUTPUT_DIR)
        generate_save_pyldavis(lda_model, bow_corpus, id2word, OUTPUT_DIR,
                               PYLDAVIS_MDS)
        df_topic_dists = generate_save_topic_dists(lda_model, bow_corpus,
                                                   NUM_TOPICS, OUTPUT_DIR)
        generate_heatmap_binary_matrix(df_topic_dists, NUM_TOPICS, OUTPUT_DIR,
                                       ALPHA_THRESHOLD)

        # Final success message + binary matrix preview
        print("=== FINAL LDA MODEL PIPELINE COMPLETE ===")
        print(f"All outputs saved to: {OUTPUT_DIR}")

    except Exception as e:
        # Clear error handling with full traceback for debugging
        print(f"\nPIPELINE FAILED: {str(e)}")
        raise  # Re-raise error to show full stack trace


if __name__ == "__main__":
    main()
