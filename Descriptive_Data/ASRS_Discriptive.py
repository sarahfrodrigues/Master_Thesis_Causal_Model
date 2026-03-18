"""
ASRS_Descriptive.py

This script performs:
- Text preprocessing (abbreviation expansion, lemmatization, stopword removal)
- Sentence and word count descriptive analysis
- N-gram extraction (bi/trigrams) with Gensim
- Word frequency analysis (top 15 words)
- Temporal analysis (accidents per year/cumulative from YYYYMM Time_Date)
- Automatic saving of all plots and preprocessed tokens to CSV
"""
# =====================================================
# Core Imports
# =====================================================
import re
import json
import os
import unicodedata
import warnings
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# Suppress minor warnings (spaCy/Gensim) for clean terminal output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================
# Global Configuration
# =====================================================
DATA_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_ALL.csv"
)
GLOSSARY_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)
PLOTS_DIR = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Descriptive_Data\ASRS_plots"
)
PREPROCESSED_CSV_PATH = r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_PREPROCESSED.csv"

# Matplotlib config
plt.rcParams["figure.dpi"] = 120

# Create output directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PREPROCESSED_CSV_PATH), exist_ok=True)

STOPWORDS = pd.read_csv(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common"
    r"\stopwords.csv"
)
STOPWORDS_LIST = set(STOPWORDS["stopword"].tolist())

stopwords_additional = {"airplane", "aircraft", "acft", "flight", "flt",
                        "time", "fly", "airline", "plane", "airport",
                        "ntsb", "report"
                        }
STOPWORDS_LIST.update(stopwords_additional)

# =====================================================
# Temporal Analysis Function (YYYYMM Time_Date Format)
# =====================================================


def run_temporal_analysis(df, plots_dir):
    """Perform temporal analysis on ASRS data (Time_Date = YYYYMM format)"""
    # Convert 200601.0 → 200601
    df["Time_Date_clean"] = (
        df["Time_Date"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    df["datetime"] = pd.to_datetime(
        df["Time_Date_clean"],
        format="%Y%m",
        errors="coerce"
    )

    # Extract year and clean invalid dates
    df["year"] = df["datetime"].dt.year

    df_clean = (
        df.dropna(subset=["year"]).assign(year=lambda x: x["year"].astype(int))
    )

    # Count unique accidents per year
    accidents_per_year = (
        df_clean.groupby("year")["_ACN"]
        .nunique()
        .sort_index()
    )

    # Reusable time series plot function
    def save_yearly_plot(series, title, ylabel, filename):
        if series.empty:
            print(f"Skipping plot {filename} — no valid data")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series.values, marker="o", linewidth=2,
                 alpha=0.8, color="#2E86AB")

        plt.title(title, fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Year", fontsize=14, fontweight="medium")
        plt.ylabel(ylabel, fontsize=14, fontweight="medium")
        plt.xticks(series.index, rotation=45, ha="right")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        path = os.path.join(plots_dir, filename)
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()

    # Generate plots
    save_yearly_plot(
        accidents_per_year,
        "Number of Unique ASRS Events per Year",
        "Number of Unique Events",
        "accidents_per_year_asrs.png"
    )
    save_yearly_plot(
        accidents_per_year.cumsum(),
        "Cumulative Number of Unique ASRS Events",
        "Cumulative Unique Events",
        "cumulative_accidents_asrs.png"
    )

# =====================================================
# Abbreviation Expansion Function
# =====================================================


def expand_abbreviations(text: str, abbr_dict: dict) -> str:
    """Expand aviation abbreviations from Glossary.json (case-insensitive)"""
    if pd.isna(text):
        return text
    text = str(text)
    for abbr, full in abbr_dict.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    return text

# =====================================================
# Text Metric Functions (Sentence/Word Count)
# =====================================================


def process_text(text: Union[str, float], nlp_basic):
    """Clean text and convert to spaCy doc for sentence/word counting"""
    clean = " ".join(str(text).split())
    clean = clean.encode("ascii", "ignore").decode()
    return nlp_basic(clean)


def count_sentences(doc) -> int:
    """Count number of sentences in a spaCy doc"""
    return len(list(doc.sents)) if doc else 0


def count_words(doc) -> int:
    """Count valid words (alpha + hyphenated alpha) in a spaCy doc"""
    if not doc:
        return 0
    tokens = [
        tok for tok in doc
        if tok.text.isalpha()
        or (tok.text.count("-") == 1 and tok.text.replace("-", "").isalpha())
    ]
    return len(tokens)

# =====================================================
# Descriptive Plot Function (Histogram + Boxplot)
# =====================================================


def plot_text_distribution(df, column, title, filename, bins=50):

    data = df[column].dropna()

    mean = data.mean()
    median = data.median()
    std = data.std()
    min_val = data.min()
    max_val = data.max()

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # ======================
    # Histogram
    # ======================
    axes[0].hist(data, bins=bins, edgecolor="black", alpha=0.8)

    axes[0].axvline(mean, linestyle="--", linewidth=2,
                    label=f"Mean = {mean:.1f}", color="red")
    axes[0].axvline(median, linestyle=":", linewidth=2,
                    label=f"Median = {median:.1f}", color="#d97706")

    axes[0].text(
        0.98, 0.92,
        f"Std = {std:.1f}",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    axes[0].set_title(column.replace("_", " ").title())
    axes[0].legend()

    # ======================
    # Boxplot
    # ======================
    axes[1].boxplot(data, vert=False)

    axes[1].set_title("Boxplot")

    # annotate min and max
    axes[1].text(min_val, 1.05, f"Min = {min_val:.0f}",
                 ha="center", fontsize=10)
    axes[1].text(max_val, 1.05, f"Max = {max_val:.0f}",
                 ha="center", fontsize=10)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved plot → {path}")


def save_descriptive_and_ngrams_txt(df, output_dir):
    """
    Save descriptive statistics + all bigrams and trigrams
    in a single TXT file (thesis-ready).
    """

    total_docs = len(df)

    # Sentence stats
    sent = df["sentence_count"]
    word = df["word_count"]

    sent_stats = {
        "mean": sent.mean(),
        "median": sent.median(),
        "std": sent.std(),
        "min": sent.min(),
        "max": sent.max()
    }

    word_stats = {
        "mean": word.mean(),
        "median": word.median(),
        "std": word.std(),
        "min": word.min(),
        "max": word.max()
    }

    # Collect n-grams
    bigrams = set()
    trigrams = set()

    for tokens in df["tokens_trigrams"]:
        for token in tokens:
            if "_" in token:
                parts = token.split("_")
                if len(parts) == 2:
                    bigrams.add(token)
                elif len(parts) == 3:
                    trigrams.add(token)

    bigrams = sorted(bigrams)
    trigrams = sorted(trigrams)

    output_path = os.path.join(
        output_dir, "NTSB_descriptive_statistics.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("NTSB DESCRIPTIVE STATISTICS AND N-GRAM VOCABULARY\n")
        f.write("=" * 80 + "\n\n")

        # --------------------------------------------------
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total number of narratives: {total_docs:,}\n\n")

        # --------------------------------------------------
        f.write("2. SENTENCE COUNT STATISTICS\n")
        f.write("-" * 80 + "\n")
        for k, v in sent_stats.items():
            f.write(f"{k.capitalize():<10}: {v:.2f}\n")
        f.write("\n")

        # --------------------------------------------------
        f.write("3. WORD COUNT STATISTICS\n")
        f.write("-" * 80 + "\n")
        for k, v in word_stats.items():
            f.write(f"{k.capitalize():<10}: {v:.2f}\n")
        f.write("\n")

        # --------------------------------------------------
        f.write("4. BIGRAM VOCABULARY\n")
        f.write("-" * 80 + "\n")
        for bg in bigrams:
            f.write(bg + "\n")
        f.write("\n")

        # --------------------------------------------------
        f.write("5. TRIGRAM VOCABULARY\n")
        f.write("-" * 80 + "\n")
        for tg in trigrams:
            f.write(tg + "\n")

    print(f"Saved descriptive statistics + n-grams → {output_path}")


# =====================================================
# NLP Preprocessing Function
# =====================================================


def preprocess_text(text, nlp_full, stopwords_list):
    """Full NLP preprocessing for topic modeling:
    lemmatize, clean, remove stopwords"""
    if pd.isna(text):
        return []
    text = str(text)

    # Remove HTML, URLs, non-alphabetic characters
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)

    # Normalize unicode and remove non-ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()

    # Lowercase and process with spaCy
    text = text.lower().strip()
    doc = nlp_full(text)

    # Lemmatize and filter tokens
    tokens = [
        tok.lemma_
        for tok in doc
        if tok.lemma_
        and tok.lemma_ not in stopwords_list
        and tok.is_alpha
        and len(tok.lemma_) >= 2
    ]
    return tokens

# =====================================================
# MAIN EXECUTION FUNCTION
# =====================================================


def main():
    # Load Data & Abbreviations
    ASRS_df = pd.read_csv(DATA_PATH)
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        ABBREVIATIONS = json.load(f)

    # Temporal Analysis
    run_temporal_analysis(ASRS_df, PLOTS_DIR)
    print("1.Temporal Analysis Complete")

    # Expand Aviation Abbreviations
    ASRS_df["Report 1_Narrative"] = (
        ASRS_df["Report 1_Narrative"]
        .astype(str)
        .apply(lambda x: expand_abbreviations(x, ABBREVIATIONS))
    )

    # SpaCy Setup
    nlp_basic = spacy.blank("en")
    nlp_basic.add_pipe("sentencizer")
    nlp_full = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    # Calculate Text Metrics
    ASRS_df["nlp_doc"] = ASRS_df["Report 1_Narrative"].apply(lambda x: process_text(x, nlp_basic))
    ASRS_df["sentence_count"] = ASRS_df["nlp_doc"].apply(count_sentences)
    ASRS_df["word_count"] = ASRS_df["nlp_doc"].apply(count_words)

    # Plot Text Metric Distributions
    plot_text_distribution(
        ASRS_df,
        "sentence_count",
        "Sentence Count per ASRS Narrative",
        "sentence_distribution_asrs.png"
    )
    plot_text_distribution(
        ASRS_df,
        "word_count",
        "Word Count per ASRS Narrative",
        "word_distribution_asrs.png"
    )

    print("2.Word/Sentence Analysis Complete")

    # Full NLP Preprocessing
    ASRS_df["tokens"] = ASRS_df["Report 1_Narrative"].apply(lambda x: preprocess_text(x, nlp_full, STOPWORDS_LIST))
    ASRS_df = ASRS_df[ASRS_df["tokens"].astype(bool)]

    # Train N-gram Models
    bigram = Phrases(ASRS_df["tokens"], min_count=400, threshold=15)
    trigram = Phrases(bigram[ASRS_df["tokens"]], min_count=300, threshold=20)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    ASRS_df["tokens_bigrams"] = ASRS_df["tokens"].apply(lambda x:
                                                        bigram_mod[x])
    ASRS_df["tokens_trigrams"] = ASRS_df["tokens_bigrams"].apply(lambda x:
                                                                 trigram_mod[x])

    # Final Token Cleaning
    STOPWORDS_AFTER = {
        "around", "back", "by", "first", "no", "not", "off", "on",
        "out", "over", "take", "top", "up", "aircraft", "airplane",
        "acft", "flight", "flt", "time", "fly", "airline", "plane",
        "airport"
    }

    def clean_final_tokens(tokens):

        cleaned = []

        for token in tokens:

            # Remove stopwords
            if token in STOPWORDS_AFTER:
                continue

            # Case 1: Unigram
            if "_" not in token:
                if len(token) >= 2:
                    cleaned.append(token)

        return cleaned

    ASRS_df["tokens_final"] = ASRS_df["tokens_trigrams"].apply(
        clean_final_tokens
    )

    save_descriptive_and_ngrams_txt(ASRS_df, PLOTS_DIR)

    print("3.NLP Preprocessing Complete")

    # Word Frequency Analysis
    fd = FreqDist()
    for tokens in ASRS_df["tokens_final"]:
        fd.update(tokens)

    top_words = fd.most_common(15)
    words, freqs = zip(*top_words)
    plt.figure(figsize=(10, 10))
    plt.barh(words, freqs, edgecolor="black", color="#2E86AB", alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Frequency", fontsize=12)
    plt.title("Top 15 Most Frequent Words (After Preprocessing)", fontsize=14,
              fontweight="bold")
    plt.tight_layout()

    freq_plot_path = os.path.join(PLOTS_DIR, "top_15_words_asrs.png")
    plt.savefig(freq_plot_path, bbox_inches="tight")
    plt.close()
    print("4.Word Frequency Analysis Complete")

    # Save Preprocessed Tokens to CSV
    ASRS_df["tokens_final_str"] = ASRS_df["tokens_final"].apply(lambda x:
                                                                " ".join(x))
    ASRS_df.to_csv(PREPROCESSED_CSV_PATH, index=False, encoding="utf-8")

    print("5.ASRS Descriptive Preprocessing Complete")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Preprocessed data saved to: {PREPROCESSED_CSV_PATH}")

# =====================================================
# RUN THE PIPELINE
# =====================================================


if __name__ == "__main__":
    main()
