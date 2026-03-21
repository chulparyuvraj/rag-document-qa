"""
evaluate.py
───────────
Evaluates the RAG pipeline on a test QA set.
Computes: Exact Match, ROUGE-L, Recall@k, and avg latency.

Usage:
    python scripts/evaluate.py --test_file data/test_qa.json
"""

import json, sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rouge_score import rouge_scorer
from src.pipeline.rag_chain import RAGPipeline
from dotenv import load_dotenv
load_dotenv()


def recall_at_k(retrieved_docs, ground_truth_source, k=5):
    """Check if ground-truth source appears in top-k retrieved docs."""
    sources = [d.metadata.get("source", "") for d in retrieved_docs[:k]]
    return int(ground_truth_source in sources)


def evaluate(test_file: str, index_path: str = "indexes/faiss"):
    with open(test_file) as f:
        test_data = json.load(f)

    rag = RAGPipeline(index_path=index_path)
    rag.build_index(save_index=False)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_scores, recall_scores, latencies = [], [], []

    for sample in test_data:
        start = time.perf_counter()
        result = rag.query(sample["question"])
        latency = (time.perf_counter() - start) * 1000

        # ROUGE-L between predicted and reference answer
        score = scorer.score(sample["answer"], result["result"])
        rouge_scores.append(score["rougeL"].fmeasure)

        # Recall@5
        r = recall_at_k(result["source_documents"], sample.get("source", ""), k=5)
        recall_scores.append(r)
        latencies.append(latency)

    print("\n── Evaluation Results ──────────────────────────")
    print(f"  ROUGE-L F1:    {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"  Recall@5:      {sum(recall_scores)/len(recall_scores):.4f}")
    print(f"  Avg Latency:   {sum(latencies)/len(latencies):.1f}ms")
    print(f"  Samples:       {len(test_data)}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--index_path", default="indexes/faiss")
    args = parser.parse_args()
    evaluate(args.test_file, args.index_path)
