"""
11_llm_label_topics.py - Label topics using LLM
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import openai

# =============================================================================
# Configuration
# =============================================================================

# Inputs from Program 08
TOPIC_INFO_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_info.csv"
)
TOPIC_KEYWORDS_JSON = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_keywords.json"
)
DOC_TOPICS_CSV = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_docs.csv"
)

# Outputs
LABELED_TOPIC_INFO_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_info_labeled.csv"
)
LABEL_MAPPING_OUT = Path(
    r"C:\Users\55279\Desktop\Mestrado\0.Thesis\Codes\Approach_3\NTSB\topic_label_mapping.json"
)

# =============================================================================
# LLM Configuration
# =============================================================================

# Initialize OpenAI client for Ollama
client = openai.OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

# Prompts for labeling
system_prompt = """
You are a helpful, respectful and concise specialist for labeling events in aviation accidents.
Your task is to create a short label (1-3 words) for accident events.
Only return the label and nothing else.
"""

main_prompt = """
I have a topic that contains documents about aviation accident events.
The topic is described by these keywords: {keywords}

Create a short label (1-3 words) for accident events.

STRICT RULES:
- Only return the label, no explanations, no extra text.
- Do NOT infer causes or responsibility.
- Do NOT introduce new technical categories.
- Keep wording factual and descriptive.
- Use aviation terminology where appropriate.
"""


def get_topic_documents_sample(topic_id: int, n_samples: int = 3) -> list[str]:
    """Get sample documents for a topic."""
    if not DOC_TOPICS_CSV.exists():
        return []

    df = pd.read_csv(DOC_TOPICS_CSV)
    if "topic_id" not in df.columns or "text" not in df.columns:
        return []

    # Filter documents for this topic
    topic_docs = df[df["topic_id"] == topic_id]["text"].tolist()

    # Return sample documents
    return topic_docs[:n_samples]


def label_topic_with_llm(topic_id: int, keywords: list, client) -> str:
    """Label a single topic using LLM."""
    if topic_id == -1:
        return "outlier"

    # Prepare prompt with keywords
    prompt = system_prompt + "\n" + main_prompt.format(keywords=", ".join(keywords[:8]))

    try:
        response = client.chat.completions.create(
            model='llama3.1:latest',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.1
        )

        label = response.choices[0].message.content.strip()
        # Clean up any extra text
        label = label.split('\n')[0].strip()
        return label
    except Exception as e:
        print(f"[WARNING] Failed to label topic {topic_id}: {str(e)}")
        return f"topic_{topic_id}"


def label_topics_batch(topic_keywords: dict, client) -> dict:
    """Label all topics in a batch."""
    topic_labels = {}

    for topic_id, keywords in topic_keywords.items():
        if topic_id == -1:
            topic_labels[topic_id] = "outlier"
            continue
        label = label_topic_with_llm(topic_id, keywords, client)
        topic_labels[topic_id] = label

    return topic_labels


def create_label_mapping(generic_labels: dict, llm_labels: dict) -> dict:
    """Create mapping from generic to LLM labels."""
    mapping = {}
    for topic_id, generic_label in generic_labels.items():
        if topic_id in llm_labels:
            mapping[generic_label] = {
                "llm_label": llm_labels[topic_id],
                "topic_id": topic_id
            }
    return mapping

# =============================================================================
# Main logic
# =============================================================================


def main():
    # Load topic info and keywords
    if not TOPIC_INFO_CSV.exists() or not TOPIC_KEYWORDS_JSON.exists():
        print("[ERROR] Required files not found. Run Program 08 first.")
        return

    topic_info = pd.read_csv(TOPIC_INFO_CSV)

    with open(TOPIC_KEYWORDS_JSON, 'r') as f:
        topic_keywords = json.load(f)

    # Convert keys to integers
    topic_keywords = {int(k): v for k, v in topic_keywords.items()}

    print(f"[INFO] Found {len(topic_keywords)} topics to label")

    # Label topics using LLM
    print("[INFO] Labeling topics with LLM...")
    llm_labels = label_topics_batch(topic_keywords, client)

    # Update topic info with LLM labels
    topic_info["llm_label"] = topic_info["topic_id"].map(llm_labels)

    # Create generic labels mapping
    generic_labels = {row["topic_id"]: row["Name"]
                      for _, row in topic_info.iterrows()}

    label_mapping = create_label_mapping(generic_labels, llm_labels)

    # Save outputs
    topic_info.to_csv(LABELED_TOPIC_INFO_OUT, index=False)

    with open(LABEL_MAPPING_OUT, 'w') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    print(f"[OK] Labeled topic info → {LABELED_TOPIC_INFO_OUT}")
    print(f"[OK] Label mapping → {LABEL_MAPPING_OUT}")
    print("\n[DONE] Program 11 completed successfully")


if __name__ == "__main__":
    main()
