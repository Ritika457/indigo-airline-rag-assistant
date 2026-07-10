import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from vectorstore import get_chroma_client, get_or_create_collection, query_collection


def load_eval_questions(path='D://Airline_RAG//data//eval_questions.json'):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_sources, relevant_source, k):
    """Out of top-k retrieved chunks, what fraction came from the relevant source doc?"""
    top_k = retrieved_sources[:k]
    hits = sum(1 for src in top_k if src == relevant_source)
    return hits / k


def recall_at_k(retrieved_sources, relevant_source, k):
    """Did the relevant source doc appear anywhere in top-k? (binary hit, since we have 1 relevant doc per question)"""
    top_k = retrieved_sources[:k]
    return 1.0 if relevant_source in top_k else 0.0


def reciprocal_rank(retrieved_sources, relevant_source):
    """1 / rank of first correct hit. 0 if not found."""
    for i, src in enumerate(retrieved_sources):
        if src == relevant_source:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_strategy(collection, eval_questions, k=3):
    precisions, recalls, rr_scores = [], [], []

    for q in eval_questions:
        results = query_collection(collection, q["question"], top_k=k)
        retrieved_sources = [meta["source"] for meta in results["metadatas"][0]]

        p = precision_at_k(retrieved_sources, q["relevant_source"], k)
        r = recall_at_k(retrieved_sources, q["relevant_source"], k)
        rr = reciprocal_rank(retrieved_sources, q["relevant_source"])

        precisions.append(p)
        recalls.append(r)
        rr_scores.append(rr)

    return {
        f"Precision@{k}": sum(precisions) / len(precisions),
        f"Recall@{k}": sum(recalls) / len(recalls),
        "MRR": sum(rr_scores) / len(rr_scores),
    }


if __name__ == "__main__":
    eval_questions = load_eval_questions()
    client = get_chroma_client()

    strategies = {
        "Fixed-size": "airline_policy_fixed",
        "Recursive": "airline_policy_recursive_v2",
        "Section-aware": "airline_policy_section",
    }

    print(f"Evaluating on {len(eval_questions)} questions...\n")
    print(f"{'Strategy':<15} {'Precision@3':<15} {'Recall@3':<12} {'MRR':<8}")
    print("-" * 55)

    results_summary = {}
    for strategy_name, collection_name in strategies.items():
        collection = get_or_create_collection(client, collection_name)
        metrics = evaluate_strategy(collection, eval_questions, k=3)
        results_summary[strategy_name] = metrics
        print(f"{strategy_name:<15} {metrics['Precision@3']:<15.3f} {metrics['Recall@3']:<12.3f} {metrics['MRR']:<8.3f}")

    # Save results to file for later use (e.g. in README, resume, or UI)
    with open('D://Airline_RAG//data//eval_questions.json', "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults saved to data/eval_results.json")