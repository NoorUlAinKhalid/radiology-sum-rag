"""
run_pipeline.py — Main entry point for A2 baseline experiments.

Runs three conditions:
  1. Clean baseline
  2. Noisy baseline (OCR-style noise)
  3. RAG-enhanced (noisy input + retrieval context)

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""

import argparse
import yaml
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_openi_dataset, split_data, save_split
from src.data.noise_injection import inject_noise_batch
from src.models.baseline_model import load_summarizer, summarize_batch
from src.models.rag_model import SimpleRAG, augment_records_with_rag
from src.evaluation.metrics import evaluate, save_results


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config: dict):
    print("=" * 60)
    print("CS818 A2 — Radiology Summarization Baseline Pipeline")
    print("=" * 60)

    #1. Load Data 
    print("\n[1/5] Loading dataset...")
    records = load_openi_dataset(
        max_samples=config["data"]["max_samples"],
        seed=config["seed"]
    )
    splits = split_data(records, seed=config["seed"])
    save_split(splits)
    test_records = splits["test"]
    train_records = splits["train"]
    print(f"Test set size: {len(test_records)}")

    # 2. Noise Injection 
    noise_level = config["noise"]["level"]
    print(f"\n[2/5] Injecting OCR noise at level {noise_level}...")
    noisy_test = inject_noise_batch(test_records, noise_level=noise_level, seed=config["seed"])

    # 3. Load Summarizer 
    print("\n[3/5] Loading summarization model...")
    summarizer = load_summarizer(config["model"]["name"])

    references = [r["impression"] for r in test_records]

    all_results = []

    # 4a. Clean Baseline 
    print("\n[4a/5] Running CLEAN baseline...")
    clean_preds = summarize_batch(
        summarizer, test_records,
        max_new_tokens=config["model"]["max_new_tokens"],
        min_new_tokens=config["model"]["min_new_tokens"]
    )
    clean_results = evaluate(clean_preds, references, condition="clean_baseline")
    all_results.append(clean_results)

    # 4b. Noisy Baseline
    print("\n[4b/5] Running NOISY baseline...")
    noisy_preds = summarize_batch(
        summarizer, noisy_test,
        max_new_tokens=config["model"]["max_new_tokens"],
        min_new_tokens=config["model"]["min_new_tokens"]
    )
    noisy_results = evaluate(noisy_preds, references, condition=f"noisy_baseline_{int(noise_level*100)}pct")
    all_results.append(noisy_results)

    # 4c. RAG-Enhanced 
    print("\n[4c/5] Running RAG-ENHANCED condition...")
    rag = SimpleRAG(embedding_model=config["rag"]["embedding_model"])
    rag.build_index(train_records)
    rag_augmented = augment_records_with_rag(rag, noisy_test, top_k=config["rag"]["top_k"])
    rag_preds = summarize_batch(
        summarizer, rag_augmented,
        max_new_tokens=config["model"]["max_new_tokens"],
        min_new_tokens=config["model"]["min_new_tokens"]
    )
    rag_results = evaluate(rag_preds, references, condition="rag_enhanced")
    all_results.append(rag_results)

    # 5. Save Results
    print("\n[5/5] Saving results...")
    save_results(all_results, output_path="artifacts/logs/a2_results.json")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Condition':<30} {'ROUGE-L':>8} {'BERTScore':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['condition']:<30} {r['rouge_l_f']:>8.4f} {r['bertscore_f1']:>10.4f}")
    print("=" * 60)

    # Degradation and RAG benefit
    if len(all_results) == 3:
        clean_r = all_results[0]["rouge_l_f"]
        noisy_r = all_results[1]["rouge_l_f"]
        rag_r = all_results[2]["rouge_l_f"]
        degradation = (clean_r - noisy_r) / clean_r * 100
        rag_benefit = (rag_r - noisy_r) / noisy_r * 100
        print(f"\nDegradation (clean→noisy): {degradation:.1f}%")
        print(f"RAG Benefit (noisy→RAG):   {rag_benefit:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
