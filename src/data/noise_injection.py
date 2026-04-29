"""
OCR-style noise injection for radiology report findings.
Simulates realistic corruption patterns from OCR pipelines.
"""

import random
import re

# OCR character confusion matrix (visually similar characters)
OCR_SUBSTITUTIONS = {
    'a': ['@', '4', 'o'],
    'e': ['3', 'c'],
    'i': ['1', 'l', '!'],
    'o': ['0', 'Q', 'q'],
    's': ['5', '$', 'z'],
    'l': ['1', 'I', '|'],
    'g': ['9', 'q'],
    'b': ['6', 'd'],
    'n': ['m', 'h'],
    't': ['f', '+'],
    'r': ['n', 'v'],
    'u': ['v', 'n'],
    'c': ['e', 'o'],
    'm': ['n', 'rn'],
    'h': ['n', 'b'],
}


def inject_noise(text: str, noise_level: float = 0.05, seed: int = 42) -> str:
    """
    Inject OCR-style noise into text.

    Args:
        text: Input text string
        noise_level: Fraction of characters to corrupt (0.05 = 5%)
        seed: Random seed for reproducibility

    Returns:
        Corrupted text string
    """
    random.seed(seed)
    chars = list(text)
    n_corrupt = max(1, int(len(chars) * noise_level))
    indices = random.sample(range(len(chars)), min(n_corrupt, len(chars)))

    for idx in indices:
        operation = random.choice(['substitute', 'delete', 'fragment', 'space'])
        char = chars[idx]

        if operation == 'substitute':
            lower = char.lower()
            if lower in OCR_SUBSTITUTIONS:
                chars[idx] = random.choice(OCR_SUBSTITUTIONS[lower])

        elif operation == 'delete':
            chars[idx] = ''

        elif operation == 'fragment':
            # Insert a spurious hyphen or space mid-token
            chars[idx] = char + random.choice(['-', ' '])

        elif operation == 'space':
            # Random extra space
            chars[idx] = char + ' '

    return ''.join(chars)


def inject_noise_batch(records: list, noise_level: float = 0.05, seed: int = 42) -> list:
    """
    Apply noise to a list of records.

    Args:
        records: List of dicts with 'findings' and 'impression'
        noise_level: Noise corruption level
        seed: Random seed

    Returns:
        New list with noisy 'findings', original 'impression' preserved
    """
    noisy_records = []
    for i, rec in enumerate(records):
        noisy_findings = inject_noise(rec["findings"], noise_level=noise_level, seed=seed + i)
        noisy_records.append({
            "findings": noisy_findings,
            "impression": rec["impression"],
            "original_findings": rec["findings"],
            "noise_level": noise_level
        })
    return noisy_records


if __name__ == "__main__":
    sample = "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax is identified."
    print("Original:", sample)
    for level in [0.05, 0.10, 0.15, 0.20]:
        noisy = inject_noise(sample, noise_level=level)
        print(f"Noise {int(level*100)}%: {noisy}")
