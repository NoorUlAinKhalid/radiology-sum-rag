"""
Baseline LLM summarization model using facebook/bart-large-cnn.
Handles clean, noisy, and RAG-enhanced summarization conditions.
"""

from transformers import pipeline
import torch


def load_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Load a HuggingFace summarization pipeline.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HuggingFace pipeline object
    """
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model: {model_name} on {'GPU' if device == 0 else 'CPU'}")
    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=device,
        truncation=True
    )
    return summarizer


def summarize_batch(summarizer, records: list, max_input_length: int = 512,
                    max_new_tokens: int = 128, min_new_tokens: int = 20) -> list:
    """
    Generate impressions from findings using the summarizer.

    Args:
        summarizer: HuggingFace pipeline
        records: List of dicts with 'findings' key
        max_input_length: Max tokens for input truncation
        max_new_tokens: Max tokens to generate
        min_new_tokens: Min tokens to generate

    Returns:
        List of generated impression strings
    """
    predictions = []
    for rec in records:
        findings = rec["findings"]
        # Truncate input to avoid token limit issues
        findings = findings[:max_input_length * 4]  # rough char estimate
        try:
            result = summarizer(
                findings,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False
            )
            pred = result[0]["summary_text"]
        except Exception as e:
            print(f"[WARN] Summarization failed: {e}")
            pred = ""
        predictions.append(pred)
    return predictions
