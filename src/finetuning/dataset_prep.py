"""
dataset_prep.py
───────────────
Prepares a QA dataset from research paper chunks for Mistral-7B fine-tuning.

Strategy:
  1. Extract chunks from indexed papers
  2. Generate (question, answer, context) triples using the base model
  3. Format in Alpaca / ChatML instruction format for SFTTrainer

Run this BEFORE train_qlora.py to create your training dataset.
"""

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from loguru import logger


# ── Instruction format for Mistral Instruct ───────────────────────────────────
INSTRUCTION_TEMPLATE = """<s>[INST] You are an expert research assistant. \
Use the following context from a research paper to answer the question accurately.

Context:
{context}

Question: {question} [/INST]
{answer}</s>"""


def format_sample(context: str, question: str, answer: str) -> str:
    """Format a single training sample in Mistral instruction format."""
    return INSTRUCTION_TEMPLATE.format(
        context=context.strip(),
        question=question.strip(),
        answer=answer.strip(),
    )


def load_raw_qa_pairs(json_path: str) -> List[Dict]:
    """
    Load hand-crafted or auto-generated QA pairs from a JSON file.

    Expected format:
    [
      {
        "context": "...",
        "question": "What method does the paper propose?",
        "answer": "The paper proposes..."
      },
      ...
    ]
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} QA pairs from {json_path}")
    return data


def prepare_dataset(
    qa_pairs: List[Dict],
    output_path: str = "data/finetune_dataset.jsonl",
    train_ratio: float = 0.9,
) -> Dict[str, str]:
    """
    Format QA pairs into instruction tuples and split into train/val.

    Returns:
        {"train": train_path, "val": val_path}
    """
    formatted = []
    for pair in tqdm(qa_pairs, desc="Formatting samples"):
        text = format_sample(
            context=pair["context"],
            question=pair["question"],
            answer=pair["answer"],
        )
        formatted.append({"text": text})

    # Train / val split
    split_idx = int(len(formatted) * train_ratio)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    out = Path(output_path)
    train_path = str(out.parent / f"train_{out.name}")
    val_path = str(out.parent / f"val_{out.name}")

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    logger.info(f"Train samples: {len(train_data)} → {train_path}")
    logger.info(f"Val   samples: {len(val_data)} → {val_path}")
    return {"train": train_path, "val": val_path}


# ── Example synthetic QA pairs (seed dataset) ────────────────────────────────
SEED_QA_PAIRS = [
    {
        "context": "Federated Learning (FL) enables multiple clients to collaboratively "
                   "train a shared model without sharing raw data. The FedAvg algorithm "
                   "aggregates client model weights by averaging them proportionally to "
                   "the number of local training samples.",
        "question": "How does FedAvg aggregate client models?",
        "answer": "FedAvg aggregates client model weights by computing a weighted average "
                  "proportional to each client's number of local training samples, "
                  "ensuring larger datasets have more influence on the global model.",
    },
    {
        "context": "Non-IID (non-independent and identically distributed) data is a key "
                   "challenge in federated learning. Label skew occurs when different "
                   "clients hold data with different class distributions, causing the "
                   "global model to be biased towards classes more represented across "
                   "clients.",
        "question": "What is label skew in federated learning?",
        "answer": "Label skew in federated learning refers to the scenario where different "
                  "clients hold training data with different class label distributions. "
                  "This Non-IID condition causes the global model to be biased and "
                  "perform poorly on underrepresented classes.",
    },
]


if __name__ == "__main__":
    # Quick test with seed data
    paths = prepare_dataset(SEED_QA_PAIRS, output_path="data/finetune_dataset.jsonl")
    print(f"Dataset prepared: {paths}")
