"""
eval.py — Run evaluation only (assumes predictions already saved).
Usage: python scripts/eval.py --predictions artifacts/logs/predictions.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import evaluate, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to JSON with predictions and references")
    args = parser.parse_args()

    with open(args.predictions, "r") as f:
        data = json.load(f)

    results = []
    for entry in data:
        condition = entry.get("condition", "unknown")
        preds = entry["predictions"]
        refs = entry["references"]
        r = evaluate(preds, refs, condition=condition)
        results.append(r)

    save_results(results, "artifacts/logs/eval_results.json")


if __name__ == "__main__":
    main()
