"""
NTSB_Descriptive.py
--------------------------------------------

This script performs:

1. Temporal analysis of NTSB accidents (per year + cumulative)
2. Narrative length analysis (sentence / word counts)
3. NLP preprocessing
4. N-gram extraction (bi- and trigrams)
5. Frequency-based descriptive analysis

"""

# =====================================================
# Imports
# =====================================================

import os
import re
import json
import unicodedata
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from gensim.models import Phrases
from gensim.models.phrases import Phraser


# =====================================================
# Configuration
# =====================================================

DATA_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_ALL.csv"
)
GLOSSARY_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)
PLOTS_DIR = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Descriptive_Data"
    r"\NTSB_plots"
)

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

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 120


# =====================================================
# NLTK setup
# =====================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS_LIST = set(STOPWORDS["stopword"].tolist())

stopwords_additional = {"airplane", "aircraft", "acft", "flight", "flt",
                        "time", "fly", "airline", "plane", "airport",
                        "ntsb", "report"
                        }
STOPWORDS_LIST.update(stopwords_additional)

# =====================================================
# Load data
# =====================================================

NTSB_df = pd.read_csv(DATA_PATH)


# =====================================================
# Temporal analysis
# =====================================================

NTSB_df["year"] = pd.to_datetime(
    NTSB_df["date"], errors="coerce"
).dt.year

NTSB_df = NTSB_df.dropna(subset=["year"])
NTSB_df["year"] = NTSB_df["year"].astype(int)

accidents_per_year = (
    NTSB_df.groupby("year")["ev_id"]
    .nunique()
    .sort_index()
)

accidents_per_year = accidents_per_year[accidents_per_year.index >= 1982]


def save_yearly_plot(series, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, marker="o", linewidth=2,
             alpha=0.8, color="#2E86AB")

    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=14, fontweight="medium")
    plt.ylabel(ylabel, fontsize=14, fontweight="medium")
    plt.xticks(series.index, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved plot → {path}")


# Accidents per year
save_yearly_plot(
    accidents_per_year,
    "Number of Accidents per Year",
    "Number of Accidents",
    "accidents_per_year_ntsb.png"
)

# Cumulative accidents
save_yearly_plot(
    accidents_per_year.cumsum(),
    "Cumulative Number of Accidents Since 1982",
    "Cumulative Accidents",
    "cumulative_accidents_ntsb.png"
)


# =====================================================
# spaCy pipelines
# =====================================================

nlp_basic = spacy.blank("en")
nlp_basic.add_pipe("sentencizer")

nlp_full = spacy.load("en_core_web_sm", disable=["ner", "parser"])


# =====================================================
# Narrative length metrics
# =====================================================

def process_text(text: Union[str, float]):
    clean = " ".join(str(text).split())
    clean = clean.encode("ascii", "ignore").decode()
    return nlp_basic(clean)


def count_sentences(doc) -> int:
    return len(list(doc.sents)) if doc else 0


def count_words(doc) -> int:
    if not doc:
        return 0

    tokens = [
        tok for tok in doc
        if tok.text.isalpha()
        or (tok.text.count("-") == 1 and tok.text.replace("-", "").isalpha())
    ]
    return len(tokens)


NTSB_df["nlp_doc"] = NTSB_df["narrative"].apply(process_text)
NTSB_df["sentence_count"] = NTSB_df["nlp_doc"].apply(count_sentences)
NTSB_df["word_count"] = NTSB_df["nlp_doc"].apply(count_words)


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


plot_text_distribution(
    NTSB_df,
    "sentence_count",
    "Sentence Count per NTSB Narrative",
    "sentence_distribution_ntsb.png"
)

plot_text_distribution(
    NTSB_df,
    "word_count",
    "Word Count per NTSB Narrative",
    "word_distribution_ntsb.png"
)


# =====================================================
# Abbreviation expansion
# =====================================================

with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
    ABBREVIATIONS = json.load(f)


def expand_abbreviations(text, abbr_dict):
    if pd.isna(text):
        return text

    text = str(text)

    for abbr, full in abbr_dict.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)

    return text


NTSB_df["narrative"] = (
    NTSB_df["narrative"]
    .astype(str)
    .apply(lambda x: expand_abbreviations(x, ABBREVIATIONS))
)


# =====================================================
# NLP preprocessing
# =====================================================

def preprocess_text(text):

    if pd.isna(text):
        return []

    text = str(text)
    text = text.lower().strip()
    text = re.sub(r"<.*?>", " ", text)             # Remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z\s]", " ", text)       # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()       # Remove extra whitespace
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    doc = nlp_full(text)

    tokens = [
        tok.lemma_
        for tok in doc
        if tok.lemma_
        and tok.lemma_ not in STOPWORDS_LIST
        and tok.is_alpha
        and len(tok.lemma_) >= 2
    ]

    return tokens


NTSB_df["tokens"] = NTSB_df["narrative"].apply(preprocess_text)
NTSB_df = NTSB_df[NTSB_df["tokens"].astype(bool)]


# =====================================================
# N-gram modeling
# =====================================================

bigram = Phrases(NTSB_df["tokens"], min_count=100, threshold=25)
trigram = Phrases(bigram[NTSB_df["tokens"]], min_count=75, threshold=20)

bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

NTSB_df["tokens_bigrams"] = NTSB_df["tokens"].apply(lambda x: bigram_mod[x])
NTSB_df["tokens_trigrams"] = (
    NTSB_df["tokens_bigrams"].apply(lambda x: trigram_mod[x])
)

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


NTSB_df["tokens_final"] = NTSB_df["tokens_trigrams"].apply(
    clean_final_tokens
)





save_descriptive_and_ngrams_txt(NTSB_df, PLOTS_DIR)

# =====================================================
# Frequency analysis
# =====================================================

fd = FreqDist()

for tokens in NTSB_df["tokens_final"]:
    fd.update(tokens)

top_words = fd.most_common(15)
words, freqs = zip(*top_words)

plt.figure(figsize=(10, 10))
plt.barh(words, freqs, edgecolor="black")
plt.gca().invert_yaxis()
plt.xlabel("Frequency")
plt.title("Top 15 Most Frequent Words (NTSB)")
plt.grid(axis="x")
plt.tight_layout()

freq_path = os.path.join(PLOTS_DIR, "top_15_words_ntsb.png")
plt.savefig(freq_path, bbox_inches="tight")
plt.close()

print(f"Saved plot → {freq_path}")

# =====================================================
# Save tokens_final to CSV
# =====================================================

NTSB_df["tokens_final_str"] = NTSB_df["tokens_final"].apply(
    lambda x: " ".join(x)
)

NTSB_df.to_csv(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\NTSB_ALL.csv",
    index=False
)

print("tokens_final saved as new column in NTSB_ALL.csv")
