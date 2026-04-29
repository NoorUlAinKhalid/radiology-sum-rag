"""
Evaluation metrics: ROUGE-L and BERTScore.
"""

import json
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def compute_rouge_l(predictions: list, references: list) -> dict:
    """
    Compute ROUGE-L scores.

    Args:
        predictions: List of generated impression strings
        references: List of reference impression strings

    Returns:
        Dict with mean precision, recall, fmeasure
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    avg_f = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    avg_p = sum(s['rougeL'].precision for s in scores) / len(scores)
    avg_r = sum(s['rougeL'].recall for s in scores) / len(scores)
    return {"rouge_l_f": round(avg_f, 4), "rouge_l_p": round(avg_p, 4), "rouge_l_r": round(avg_r, 4)}


def compute_bertscore(predictions: list, references: list, lang: str = "en") -> dict:
    """
    Compute BERTScore F1.

    Args:
        predictions: List of generated impression strings
        references: List of reference impression strings
        lang: Language code

    Returns:
        Dict with mean BERTScore F1
    """
    P, R, F = bert_score_fn(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_f1": round(F.mean().item(), 4),
        "bertscore_p": round(P.mean().item(), 4),
        "bertscore_r": round(R.mean().item(), 4)
    }


def evaluate(predictions: list, references: list, condition: str = "unknown") -> dict:
    """
    Run full evaluation suite.

    Args:
        predictions: Generated impressions
        references: Ground truth impressions
        condition: Label for this evaluation condition

    Returns:
        Combined metrics dict
    """
    rouge = compute_rouge_l(predictions, references)
    bert = compute_bertscore(predictions, references)
    results = {"condition": condition, **rouge, **bert}
    print(f"\n=== Results: {condition} ===")
    for k, v in results.items():
        if k != "condition":
            print(f"  {k}: {v}")
    return results


def save_results(results: list, output_path: str = "artifacts/logs/results.json"):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
