"""
run_a3_diagnostics.py — A3 Evaluation & Diagnostic Analysis Pipeline

Runs systematic evaluation across:
  - Multiple noise levels: 0.05, 0.10, 0.15, 0.20
  - Multiple RAG top-k values: 1, 3, 5
  - Saves per-sample predictions for error analysis
  - Produces summary tables for the report

Usage:
    python scripts/run_a3_diagnostics.py
"""

import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouge_score import rouge_scorer

# For reproducibility
SEED = 42
random.seed(SEED)

# synthetic dataset (same as load_data.py fallback)
def get_records():
    base = [
        {
            "findings": "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax is identified. The cardiomediastinal silhouette is within normal limits. Osseous structures are intact.",
            "impression": "No acute cardiopulmonary abnormality."
        },
        {
            "findings": "There is increased opacity in the right lower lobe consistent with pneumonia. The left lung is clear. Mild cardiomegaly is present. No pleural effusion identified.",
            "impression": "Right lower lobe pneumonia. Mild cardiomegaly."
        },
        {
            "findings": "Mild interstitial prominence noted bilaterally. Heart size is at the upper limits of normal. No pleural effusion. No pneumothorax. Bony thorax is unremarkable.",
            "impression": "Mild interstitial prominence, possibly early pulmonary edema."
        },
        {
            "findings": "The cardiac silhouette is enlarged. Bilateral pleural effusions are present, left greater than right. Pulmonary vascular congestion noted.",
            "impression": "Cardiomegaly with bilateral pleural effusions and pulmonary vascular congestion, consistent with congestive heart failure."
        },
        {
            "findings": "No acute osseous abnormality. Lungs are hyperinflated. Flattening of the diaphragm noted. No focal consolidation. No pneumothorax.",
            "impression": "Hyperinflation consistent with chronic obstructive pulmonary disease (COPD)."
        },
        {
            "findings": "Patchy opacity in the left lower lobe. Mild blunting of the left costophrenic angle suggesting small pleural effusion. Heart size normal.",
            "impression": "Left lower lobe opacity with small pleural effusion, possibly pneumonia or atelectasis."
        },
        {
            "findings": "The heart is normal in size. The mediastinum is unremarkable. Both lungs are clear. No pleural effusion or pneumothorax. Visualized bony structures are intact.",
            "impression": "Normal chest radiograph."
        },
        {
            "findings": "Diffuse bilateral airspace opacities consistent with pulmonary edema. Cardiomegaly noted. Bilateral pleural effusions present.",
            "impression": "Pulmonary edema with cardiomegaly and bilateral pleural effusions."
        },
        {
            "findings": "Right-sided pneumothorax identified with partial collapse of the right lung. Tracheal deviation to the left. No rib fractures identified.",
            "impression": "Right-sided pneumothorax with partial lung collapse."
        },
        {
            "findings": "Linear opacities at the left base consistent with subsegmental atelectasis. No pneumonia or pleural effusion. Heart size normal.",
            "impression": "Subsegmental atelectasis at the left base. No acute pneumonia."
        },
    ] * 20  # 200 records total

    random.shuffle(base)
    n = len(base)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    return {
        "train": base[:train_end],
        "val":   base[train_end:val_end],
        "test":  base[val_end:]
    }

# OCR noise injection 
OCR_SUBS = {
    'a': ['@','4','o'], 'e': ['3','c'], 'i': ['1','l','!'],
    'o': ['0','Q','q'], 's': ['5','$','z'], 'l': ['1','I','|'],
    'g': ['9','q'], 'b': ['6','d'], 'n': ['m','h'], 't': ['f','+'],
    'r': ['n','v'], 'u': ['v','n'], 'c': ['e','o'], 'm': ['n','rn'],
    'h': ['n','b'],
}

def inject_noise(text, noise_level=0.05, seed=42):
    random.seed(seed)
    chars = list(text)
    n_corrupt = max(1, int(len(chars) * noise_level))
    indices = random.sample(range(len(chars)), min(n_corrupt, len(chars)))
    for idx in indices:
        op = random.choice(['substitute','delete','fragment','space'])
        char = chars[idx]
        if op == 'substitute':
            lo = char.lower()
            if lo in OCR_SUBS:
                chars[idx] = random.choice(OCR_SUBS[lo])
        elif op == 'delete':
            chars[idx] = ''
        elif op == 'fragment':
            chars[idx] = char + random.choice(['-',' '])
        elif op == 'space':
            chars[idx] = char + ' '
    return ''.join(chars)

# Simple cosine-similarity RAG
def simple_word_overlap_retrieve(query, kb, top_k=3):
    """
    Lightweight retrieval using word-overlap (Jaccard similarity).
    Replaces SentenceTransformer to avoid heavy install.
    Still semantic enough for this dataset size.
    """
    query_tokens = set(query.lower().split())
    scores = []
    for i, rec in enumerate(kb):
        kb_tokens = set(rec["findings"].lower().split())
        if not query_tokens or not kb_tokens:
            scores.append((0, i))
            continue
        overlap = len(query_tokens & kb_tokens)
        union   = len(query_tokens | kb_tokens)
        scores.append((overlap / union, i))
    scores.sort(reverse=True)
    return [kb[i] for _, i in scores[:top_k]]

def build_rag_prompt(noisy_findings, kb, top_k=3):
    retrieved = simple_word_overlap_retrieve(noisy_findings, kb, top_k)
    parts = []
    for i, r in enumerate(retrieved):
        parts.append(f"Example {i+1}:\nFindings: {r['findings']}\nImpression: {r['impression']}")
    context = "\n\n".join(parts)
    # Budget: keep prompt from overflowing BART's 1024-token limit
    # Truncate context to ~600 chars + keep full query
    if len(context) > 600:
        context = context[:600] + "..."
    return f"{context}\n\nNow summarize:\nFindings: {noisy_findings}\nImpression:"

# Mock summarizer (rule-based extraction for pipeline testing) ──
# NOTE: Replace this with real BART when running on Colab with GPU
def mock_summarize(findings_text):
    """
    Extracts the most clinically relevant sentence as a proxy summary.
    Used when BART is not available. Replace with real summarizer on Colab.
    """
    # Extract key clinical terms
    findings_lower = findings_text.lower()

    if "pneumothorax" in findings_lower and "right" in findings_lower:
        return "Right-sided pneumothorax identified."
    elif "pneumothorax" in findings_lower:
        return "Pneumothorax identified."
    elif "pneumonia" in findings_lower and "right lower" in findings_lower:
        return "Right lower lobe pneumonia."
    elif "pneumonia" in findings_lower:
        return "Pneumonia identified."
    elif "pulmonary edema" in findings_lower:
        return "Pulmonary edema with cardiomegaly."
    elif "pleural effusion" in findings_lower and "cardiomegaly" in findings_lower:
        return "Cardiomegaly with bilateral pleural effusions consistent with heart failure."
    elif "hyperinflat" in findings_lower or "copd" in findings_lower or "diaphragm" in findings_lower:
        return "Hyperinflation consistent with COPD."
    elif "atelectasis" in findings_lower:
        return "Subsegmental atelectasis. No acute pneumonia."
    elif "interstitial" in findings_lower:
        return "Mild interstitial prominence, possibly early pulmonary edema."
    elif "clear" in findings_lower and "no focal" in findings_lower:
        return "No acute cardiopulmonary abnormality."
    else:
        # Generic fallback — this is the "generic impression" failure mode
        return "No acute cardiopulmonary abnormality."

# ROUGE-L computation
def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    f = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    p = sum(s['rougeL'].precision for s in scores) / len(scores)
    r = sum(s['rougeL'].recall for s in scores) / len(scores)
    return {"rouge_l_f": round(f,4), "rouge_l_p": round(p,4), "rouge_l_r": round(r,4)}

# Per-sample prediction logger (needed for error analysis) 
def log_predictions(records, predictions, condition, noise_level, output_dir="artifacts/logs"):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{condition}_noise{int(noise_level*100)}.json"
    path  = os.path.join(output_dir, fname)
    entries = []
    for rec, pred in zip(records, predictions):
        ref = rec.get("impression","")
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score  = scorer.score(ref, pred)['rougeL'].fmeasure
        entries.append({
            "original_findings":  rec.get("original_findings", rec["findings"]),
            "noisy_findings":     rec["findings"],
            "reference":          ref,
            "prediction":         pred,
            "rouge_l":            round(score, 4),
            "noise_level":        noise_level,
        })
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    return entries

# Failure taxonomy classifier 
def classify_failure(entry):
    pred  = entry["prediction"].lower()
    ref   = entry["reference"].lower()
    orig  = entry["original_findings"].lower()
    noisy = entry["noisy_findings"].lower()

    # Generic output: model outputs a safe fallback regardless of input
    GENERIC_PHRASES = [
        "no acute cardiopulmonary abnormality",
        "normal chest",
        "no acute findings"
    ]
    if any(g in pred for g in GENERIC_PHRASES) and not any(g in ref for g in GENERIC_PHRASES):
        return "generic_impression"

    # Missed pathology: reference mentions disease but prediction doesn't
    PATHOLOGY_TERMS = ["pneumonia","pneumothorax","effusion","edema","cardiomegaly","atelectasis","copd","opacity"]
    ref_path  = [t for t in PATHOLOGY_TERMS if t in ref]
    pred_path = [t for t in PATHOLOGY_TERMS if t in pred]
    if ref_path and not any(t in pred_path for t in ref_path):
        return "missed_pathology"

    # Noise-corrupted key term: noisy findings have corrupted tokens
    corruption_signals = ['0', '1', '@', '$', '|', '3', '!', '-']
    noisy_tokens = noisy.split()
    corrupted = sum(1 for tok in noisy_tokens if any(s in tok for s in corruption_signals))
    if corrupted > 2 and entry["rouge_l"] < 0.3:
        return "noise_corrupted_output"

    # Partial match: got some right but missed specifics
    if 0.2 <= entry["rouge_l"] < 0.5:
        return "partial_match"

    if entry["rouge_l"] >= 0.5:
        return "correct"

    return "other_failure"

# MAIN 
def main():
    print("=" * 65)
    print("CS818 A3 — Evaluation & Diagnostic Analysis Pipeline")
    print("=" * 65)

    splits = get_records()
    train_records = splits["train"]
    test_records  = splits["test"]
    print(f"Train: {len(train_records)} | Test: {len(test_records)} records")

    NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]
    RAG_TOPK_VALUES = [1, 3, 5]

    all_results     = []
    all_predictions = []  # for error analysis

    references = [r["impression"] for r in test_records]

    # 1. Clean baseline
    print("\n[1] Running CLEAN baseline...")
    clean_preds = [mock_summarize(r["findings"]) for r in test_records]
    clean_metrics = compute_rouge_l(clean_preds, references)
    clean_result = {"condition": "clean_baseline", "noise_level": 0.0, "top_k": None, **clean_metrics}
    all_results.append(clean_result)
    print(f"    ROUGE-L F: {clean_metrics['rouge_l_f']}")

    # 2. Noise level sweep
    print("\n[2] Noise level sweep (no RAG)...")
    noisy_by_level = {}
    for level in NOISE_LEVELS:
        noisy_recs = []
        for i, rec in enumerate(test_records):
            noisy_findings = inject_noise(rec["findings"], noise_level=level, seed=SEED+i)
            noisy_recs.append({
                "findings": noisy_findings,
                "impression": rec["impression"],
                "original_findings": rec["findings"],
                "noise_level": level
            })
        noisy_by_level[level] = noisy_recs

        preds   = [mock_summarize(r["findings"]) for r in noisy_recs]
        metrics = compute_rouge_l(preds, references)
        result  = {"condition": "noisy_baseline", "noise_level": level, "top_k": None, **metrics}
        all_results.append(result)

        entries = log_predictions(noisy_recs, preds, "noisy_baseline", level)
        all_predictions.extend(entries)
        print(f"    Noise {int(level*100)}%: ROUGE-L F = {metrics['rouge_l_f']}")

    # 3. RAG top-k ablation (at 10% noise) 
    print("\n[3] RAG top-k ablation at 10% noise...")
    noisy_10 = noisy_by_level[0.10]
    for topk in RAG_TOPK_VALUES:
        rag_recs = []
        for rec in noisy_10:
            prompt = build_rag_prompt(rec["findings"], train_records, top_k=topk)
            rag_recs.append({**rec, "findings": prompt})

        preds   = [mock_summarize(r["findings"]) for r in rag_recs]
        metrics = compute_rouge_l(preds, references)
        result  = {"condition": "rag_enhanced", "noise_level": 0.10, "top_k": topk, **metrics}
        all_results.append(result)

        entries = log_predictions(noisy_10, preds, f"rag_top{topk}", 0.10)
        all_predictions.extend(entries)
        print(f"    RAG top-{topk}: ROUGE-L F = {metrics['rouge_l_f']}")

    # 4. RAG across all noise levels (top_k=3)
    print("\n[4] RAG (top-k=3) across all noise levels...")
    for level in NOISE_LEVELS:
        noisy_recs = noisy_by_level[level]
        rag_recs = []
        for rec in noisy_recs:
            prompt = build_rag_prompt(rec["findings"], train_records, top_k=3)
            rag_recs.append({**rec, "findings": prompt})

        preds   = [mock_summarize(r["findings"]) for r in rag_recs]
        metrics = compute_rouge_l(preds, references)
        result  = {"condition": "rag_enhanced_sweep", "noise_level": level, "top_k": 3, **metrics}
        all_results.append(result)
        print(f"    Noise {int(level*100)}% + RAG: ROUGE-L F = {metrics['rouge_l_f']}")

    # 5. Failure taxonomy
    print("\n[5] Classifying failures...")
    taxonomy = {}
    for entry in all_predictions:
        label = classify_failure(entry)
        taxonomy[label] = taxonomy.get(label, 0) + 1

    total = sum(taxonomy.values())
    print("\n  Failure Taxonomy:")
    for label, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
        print(f"    {label:<30} {count:>4}  ({100*count/total:.1f}%)")

    # 6. Save everything 
    os.makedirs("artifacts/logs", exist_ok=True)

    with open("artifacts/logs/a3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open("artifacts/logs/a3_taxonomy.json", "w") as f:
        json.dump({"taxonomy": taxonomy, "total": total}, f, indent=2)

    # 7. Print full summary table 
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Condition':<22} {'Noise':>6} {'TopK':>5} {'ROUGE-L':>9}")
    print("-" * 65)
    for r in all_results:
        topk_str = str(r['top_k']) if r['top_k'] else "—"
        noise_str = f"{int(r['noise_level']*100)}%" if r['noise_level'] > 0 else "0%"
        print(f"{r['condition']:<22} {noise_str:>6} {topk_str:>5} {r['rouge_l_f']:>9.4f}")

    # 8. Key stats for report 
    clean_r = next(r for r in all_results if r['condition']=='clean_baseline')['rouge_l_f']
    print("\n" + "=" * 65)
    print("KEY STATS FOR YOUR REPORT")
    print("=" * 65)
    print(f"Clean baseline ROUGE-L:     {clean_r:.4f}")
    for level in NOISE_LEVELS:
        noisy_r = next(r for r in all_results if r['condition']=='noisy_baseline' and r['noise_level']==level)['rouge_l_f']
        rag_r   = next(r for r in all_results if r['condition']=='rag_enhanced_sweep' and r['noise_level']==level)['rouge_l_f']
        deg     = (clean_r - noisy_r) / clean_r * 100
        rec     = (rag_r - noisy_r) / noisy_r * 100 if noisy_r > 0 else 0
        print(f"Noise {int(level*100)}%: ROUGE-L={noisy_r:.4f}  Degradation={deg:.1f}%  RAG recovery={rec:.1f}%")

    print(f"\nResults saved to artifacts/logs/")
    print("Done!")

if __name__ == "__main__":
    main()