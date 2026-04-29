"""
Basic tests for A2 pipeline components.
Run: python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.noise_injection import inject_noise, inject_noise_batch
from src.data.load_data import _synthetic_fallback, split_data


def test_noise_injection_changes_text():
    text = "The lungs are clear bilaterally. No focal consolidation identified."
    noisy = inject_noise(text, noise_level=0.10, seed=42)
    assert noisy != text, "Noisy text should differ from original"


def test_noise_injection_level_zero():
    text = "No acute abnormality detected."
    # Very low noise may not always change text, but shouldn't crash
    noisy = inject_noise(text, noise_level=0.0, seed=42)
    assert isinstance(noisy, str)


def test_noise_batch():
    records = [
        {"findings": "Clear lungs bilaterally.", "impression": "Normal."},
        {"findings": "Right lower lobe consolidation.", "impression": "Pneumonia."}
    ]
    noisy = inject_noise_batch(records, noise_level=0.10, seed=42)
    assert len(noisy) == 2
    assert "original_findings" in noisy[0]
    assert "noise_level" in noisy[0]


def test_data_split():
    records = _synthetic_fallback()
    splits = split_data(records, seed=42)
    total = sum(len(v) for v in splits.values())
    assert total == len(records), "Split should preserve all records"
    assert "train" in splits and "val" in splits and "test" in splits


def test_split_ratios():
    records = _synthetic_fallback()
    splits = split_data(records, train_ratio=0.7, val_ratio=0.15, seed=42)
    n = len(records)
    assert len(splits["train"]) == int(n * 0.7)


if __name__ == "__main__":
    test_noise_injection_changes_text()
    test_noise_injection_level_zero()
    test_noise_batch()
    test_data_split()
    test_split_ratios()
    print("All tests passed.")
