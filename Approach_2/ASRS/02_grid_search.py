"""
02_grid_search.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import json

# =============================================================================
# Download NLTK stopwords
# =============================================================================
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =============================================================================
# Configuration
# =============================================================================
# Input/Output Paths
TOPIC_DOCS_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\ASRS_PREPROCESSED.csv"
)
GRID_SEARCH_RESULTS_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_2\ASRS\gs_result.csv"
)
GLOSSARY_PATH = (
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Common\Glossary.json"
)

# Local Model Path
LOCAL_MODEL_PATH = Path(r"C:\models\all-MiniLM-L6-v2")

# --------------------------
# 1. High-Impact Hyperparameters
# --------------------------
SMART_GRID_PARAMS = {
    "stage1": {
        "MIN_TOPIC_SIZE": [130, 150, 180, 200, 250],
        "MIN_SAMPLE_SIZE": [65, 85, 105, 125],
        "NUM_NEIGHBORS": [15, 25, 35, 45, 55]
    },
    "stage2": {
        "MIN_TOPIC_SIZE": [],
        "MIN_SAMPLE_SIZE": [],
        "NUM_NEIGHBORS": []
    }
}

STOPWORDS_LIST = list(stopwords.words('english'))

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


def calculate_topic_coherence_simple(topic_model, docs, topn=10):

    docs_sampled = docs
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
        for doc in docs_sampled
    ]

    dictionary = Dictionary(tokenized_docs)

    # Remove extremes - filter out too rare and too common words
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v"
        )

    return coherence_model.get_coherence()

# =============================================================================
# Grid Search Core Functions
# =============================================================================


def train_bertopic_with_params(params, docs, embedding_model):
    """
    Train a single BERTopic model with given params.
    """
    min_topic_size = params["MIN_TOPIC_SIZE"]
    min_sample_size = params["MIN_SAMPLE_SIZE"]
    num_neighbors = params["NUM_NEIGHBORS"]

    print(f"  Training with params: min_topic_size={min_topic_size}, "
          f"min_sample_size={min_sample_size}, num_neighbors={num_neighbors}")

    try:

        umap_model = UMAP(
            n_neighbors=num_neighbors,
            n_components=2,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_sample_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=STOPWORDS_LIST
        )

        # Train BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=False,
            verbose=False
        )
        topics, _ = topic_model.fit_transform(docs)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        n_valid_topics = int((topic_info["Topic"] >= 0).sum())
        n_outlier_docs = len([t for t in topics if t == -1])

        print(f"-> Created {n_valid_topics} topics, {n_outlier_docs} outliers")

        # Calculate coherence
        coherence = calculate_topic_coherence_simple(topic_model, docs)

        print(f"-> Coherence score: {coherence:.4f}")

        # Early pruning
        if coherence < 0.65:
            print(f"-> Pruned (score {coherence:.4f} < threshold 0.65)")
            return None

        # Return results
        return {
            "params": params,
            "model": topic_model,
            "topics": topics,
            "topic_info": topic_info,
            "coherence": coherence,
            "n_valid_topics": n_valid_topics,
            "n_outlier_docs": n_outlier_docs
        }

    except Exception as e:
        print(f"[ERROR] Failed to train model with params {params}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_stage_search(stage_params, docs, embedding_model, stage_name):
    """
    Run a single stage of the smarter grid search (sequential).
    """
    # Generate all valid parameter combinations for the stage
    param_combinations = []

    for min_topic_size in stage_params["MIN_TOPIC_SIZE"]:
        for min_sample_size in stage_params["MIN_SAMPLE_SIZE"]:
            for num_neighbors in stage_params["NUM_NEIGHBORS"]:
                if min_topic_size < min_sample_size:
                    continue
                param_combinations.append({
                    "MIN_TOPIC_SIZE": min_topic_size,
                    "MIN_SAMPLE_SIZE": min_sample_size,
                    "NUM_NEIGHBORS": num_neighbors
                })

    print(f"[INFO] Running {len(param_combinations)} combinations")

    if len(param_combinations) == 0:
        print(f"[WARNING] {stage_name}: No valid parameter combinations!")
        return []

    # Run training sequentially
    valid_results = []
    for i, params in enumerate(param_combinations):
        print(f"\n[INFO] {stage_name}: Run {i+1}/{len(param_combinations)}")
        result = train_bertopic_with_params(params, docs, embedding_model)
        if result:
            valid_results.append(result)

    # Sort by coherence score (descending)
    valid_results.sort(key=lambda x: x["coherence"], reverse=True)

    print(f"[INFO] {stage_name}: Completed. {len(valid_results)} valid "
          f"models out of {len(param_combinations)} attempts")

    if len(valid_results) > 0:
        best_coherence = valid_results[0]['coherence']
        print(f"[INFO] Best score in {stage_name}: {best_coherence:.4f}")
        best_params = valid_results[0]['params']
        print(f"[INFO] Best params in {stage_name}: {best_params}")

    return valid_results


def generate_fine_search_params(stage1_results, max_searches=8):
    if len(stage1_results) == 0:
        print("[WARNING] No Stage 1 results to generate fine search params!")
        return []

    top_results = stage1_results[:min(3, len(stage1_results))]

    print(f"\n[INFO] Generating fine search around top "
          f"{len(top_results)} results from Stage 1")

    # Initial step sizes (will shrink)
    step_sizes = {
        "MIN_TOPIC_SIZE": 5,
        "MIN_SAMPLE_SIZE": 5,
        "NUM_NEIGHBORS": 5
    }

    param_bounds = {
        "MIN_TOPIC_SIZE": (110, 250),
        "MIN_SAMPLE_SIZE": (45, 125),
        "NUM_NEIGHBORS": (10, 60)
    }

    candidate_params = []
    param_set = set()

    for i, result in enumerate(top_results):
        base_params = result["params"]
        base_score = result["coherence"]

        print(f"\n[INFO] Base params {i+1}: {base_params}")
        print(f"       Coherence: {base_score:.4f}")

        improved = False

        # Coordinate-wise search
        for param_name in base_params:
            step = step_sizes[param_name]

            for direction in [-1, 1]:
                new_params = base_params.copy()
                new_value = new_params[param_name] + direction * step

                # Apply bounds
                min_b, max_b = param_bounds[param_name]
                new_value = max(min_b, min(max_b, new_value))
                new_params[param_name] = new_value

                # Constraint
                if new_params["MIN_TOPIC_SIZE"] < new_params["MIN_SAMPLE_SIZE"]:
                    continue

                param_key = (
                    new_params["MIN_TOPIC_SIZE"],
                    new_params["MIN_SAMPLE_SIZE"],
                    new_params["NUM_NEIGHBORS"]
                )

                if param_key in param_set:
                    continue

                param_set.add(param_key)
                candidate_params.append(new_params)

                print(f"  Try {param_name} {'+' if direction > 0 else '-'}{step}: "
                      f"{new_params}")

                improved = True

                if len(candidate_params) >= max_searches:
                    break

            if len(candidate_params) >= max_searches:
                break

        # If no directional improvement found, shrink steps
        if not improved:
            for p in step_sizes:
                step_sizes[p] = max(1, step_sizes[p] // 2)
            print(f"[INFO] No improvement found, shrinking steps: {step_sizes}")

        if len(candidate_params) >= max_searches:
            break

    print(f"\n[INFO] Generated {len(candidate_params)} unique parameter combinations")

    # Update SMART_GRID_PARAMS (unchanged)
    if candidate_params:
        SMART_GRID_PARAMS["stage2"]["MIN_TOPIC_SIZE"] = sorted(
            {p["MIN_TOPIC_SIZE"] for p in candidate_params}
        )
        SMART_GRID_PARAMS["stage2"]["MIN_SAMPLE_SIZE"] = sorted(
            {p["MIN_SAMPLE_SIZE"] for p in candidate_params}
        )
        SMART_GRID_PARAMS["stage2"]["NUM_NEIGHBORS"] = sorted(
            {p["NUM_NEIGHBORS"] for p in candidate_params}
        )

        print("\n[INFO] Stage 2 parameter ranges:")
        for k, v in SMART_GRID_PARAMS["stage2"].items():
            print(f"  {k}: {v}")

    return candidate_params


def run_fine_stage_search(stage1_results, docs, embedding_model,
                          stage_name="Stage 2 (Fine Search)"):
    """
    Run fine search with generated parameters around top stage 1 results.
    """
    # Generate fine search parameters
    fine_params_list = generate_fine_search_params(
        stage1_results, max_searches=8)

    if not fine_params_list:
        print(f"[WARNING] {stage_name}: No fine search parameters generated!")
        return []

    print(f"\n[INFO] {stage_name}: Running "
          f"{len(fine_params_list)} combinations")

    # Run training sequentially
    valid_results = []
    for i, params in enumerate(fine_params_list):
        print(f"\n[INFO] {stage_name}: Run {i+1}/{len(fine_params_list)}")
        result = train_bertopic_with_params(params, docs, embedding_model)
        if result:
            valid_results.append(result)

    # Sort by coherence score (descending)
    valid_results.sort(key=lambda x: x["coherence"], reverse=True)

    print(f"\n[INFO] {stage_name}: Completed. {len(valid_results)} valid "
          f"models out of {len(fine_params_list)} attempts")

    if len(valid_results) > 0:
        best_coherence = valid_results[0]['coherence']
        print(f"[INFO] Best score in {stage_name}: {best_coherence:.4f}")
        best_params = valid_results[0]['params']
        print(f"[INFO] Best params in {stage_name}: {best_params}")

    return valid_results


# =============================================================================
# Main Logic
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    print("[INFO] Starting Smarter BERTopic Grid Search")

    if not TOPIC_DOCS_CSV.exists():
        print(f"[SKIP] topic_docs.csv not found at: {TOPIC_DOCS_CSV}")
        print("[SKIP] Run Program 04 first.")
        return

    df = pd.read_csv(TOPIC_DOCS_CSV)
    print(f"[INFO] Loaded CSV with {len(df)} rows")

    #filter 20000 random rows
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled down to 20,000 rows for faster processing")

    # Preprocess docs (abbreviation expansion)
    print("[INFO] Expanding abbreviations...")
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        abreviations = json.load(f)
    df["Report 1_Narrative"] = df["Report 1_Narrative"].apply(
        lambda x: expand_abbreviations(x, abreviations)
    )
    docs = df["Report 1_Narrative"].astype(str).tolist()
    n_docs = len(docs)

    print(f"[INFO] Documents available: {n_docs}")

    # -------------------------------------------------------------------------
    # Initialize Embedding Model
    # -------------------------------------------------------------------------
    print("[INFO] Initializing embedding model...")

    if LOCAL_MODEL_PATH.exists():
        embedding_model = SentenceTransformer(str(LOCAL_MODEL_PATH))
        print(f"[INFO] Using local embedding model: {LOCAL_MODEL_PATH}")
    else:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Local model not found, using remote all-MiniLM-L6-v2")

    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {embedding_dim}")

    # -------------------------------------------------------------------------
    # Stage 1: Coarse Search (Narrow Down Promising Params)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[INFO] Starting Stage 1: Coarse Search")
    print("="*60)

    stage1_results = run_stage_search(
        stage_params=SMART_GRID_PARAMS["stage1"],
        docs=docs,
        embedding_model=embedding_model,
        stage_name="Stage 1 (Coarse Search)"
    )

    if len(stage1_results) == 0:
        print("[SKIP] No valid models from Stage 1.")
        return

    # -------------------------------------------------------------------------
    # Stage 2: Fine Search (4 searches around each of top 3 from Stage 1)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[INFO] Starting Stage 2: Fine Search")
    print("[INFO] Generating 4 searches around each of top 3 results")
    print("="*60)

    stage2_results = run_fine_stage_search(
        stage1_results=stage1_results,
        docs=docs,
        embedding_model=embedding_model,
        stage_name="Stage 2 (Fine Search)"
    )

    # -------------------------------------------------------------------------
    # Select Best Model (Combine Stage 1 + Stage 2 Results)
    # -------------------------------------------------------------------------
    all_results = stage1_results + stage2_results

    if len(all_results) == 0:
        print("[SKIP] No valid models found in any stage.")
        return

    all_results.sort(key=lambda x: x["coherence"], reverse=True)
    best_result = all_results[0]

    # Extract best model details
    best_topic_info = best_result["topic_info"]
    n_valid_topics = best_result["n_valid_topics"]
    n_outlier_docs = best_result["n_outlier_docs"]

    # -------------------------------------------------------------------------
    # Validate Best Model
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("[INFO] Best Model Final Stats")
    print("="*60)
    print(f"[INFO] Coherence: {best_result['coherence']:.4f}")
    print(f"[INFO] Valid Topics: {n_valid_topics} | "
          f"Outlier Docs: {n_outlier_docs}/{n_docs}")
    print(f"[INFO] Best Params: {best_result['params']}")
    print("[INFO] Topic distribution:")
    print(best_topic_info[["Topic", "Count", "Name"]].head(10))

    if n_valid_topics < 3:
        print("[WARNING] Best model has fewer than 3 valid topics")
        print("[INFO] Continuing anyway...")

    # -------------------------------------------------------------------------
    # Save Outputs
    # -------------------------------------------------------------------------
    # Ensure output directories exist
    for out_path in [
        GRID_SEARCH_RESULTS_OUT
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save Grid Search Results
    grid_results_df = pd.DataFrame([
        {
            "min_topic_size": res["params"]["MIN_TOPIC_SIZE"],
            "min_sample_size": res["params"]["MIN_SAMPLE_SIZE"],
            "num_neighbors": res["params"]["NUM_NEIGHBORS"],
            "coherence": res["coherence"],
            "n_valid_topics": res["n_valid_topics"],
            "n_outlier_docs": res["n_outlier_docs"],
            "stage": "Stage 1" if i < len(stage1_results) else "Stage 2"
        }
        for i, res in enumerate(all_results)
    ])
    grid_results_df.to_csv(GRID_SEARCH_RESULTS_OUT, index=False)

    print(f"\n[INFO] Stage 1 results: {len(stage1_results)} valid models")
    print(f"[INFO] Stage 2 results: {len(stage2_results)} valid models")
    print(f"[INFO] Total results: {len(all_results)} models")
    print(f"[OK] Grid search results saved to {GRID_SEARCH_RESULTS_OUT}")
    print("[DONE] Program 06 completed successfully")


# =============================================================================
# Script Entry
# =============================================================================

if __name__ == "__main__":
    main()
