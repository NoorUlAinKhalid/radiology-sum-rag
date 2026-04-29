"""
run_evaluation.py — Full Diagnostic Evaluation Pipeline

Runs systematic evaluation across:
  - Multiple noise levels: 0.05, 0.10, 0.15, 0.20
  - Multiple RAG top-k values: 1, 3, 5
  - Real BART summarization on GPU/CPU
  - Saves per-sample predictions for error analysis
  - Generates publication-quality plots

Usage:
    python scripts/run_evaluation.py
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_openi_dataset, split_data, save_split
from src.data.noise_injection import inject_noise_batch
from src.models.baseline_model import load_summarizer, summarize_batch
from src.models.rag_model import SimpleRAG, augment_records_with_rag
from src.evaluation.metrics import compute_rouge_l, compute_bertscore
from rouge_score import rouge_scorer as rouge_scorer_module

import numpy as np

SEED = 42
random.seed(SEED)

# ── Failure taxonomy classifier ───────────────────────────────────
def classify_failure(entry):
    pred  = entry["prediction"].lower()
    ref   = entry["reference"].lower()
    noisy = entry["noisy_findings"].lower()

    GENERIC_PHRASES = [
        "no acute cardiopulmonary abnormality",
        "normal chest",
        "no acute findings"
    ]
    if any(g in pred for g in GENERIC_PHRASES) and not any(g in ref for g in GENERIC_PHRASES):
        return "generic_impression"

    PATHOLOGY_TERMS = ["pneumonia","pneumothorax","effusion","edema",
                       "cardiomegaly","atelectasis","copd","opacity"]
    ref_path  = [t for t in PATHOLOGY_TERMS if t in ref]
    pred_path = [t for t in PATHOLOGY_TERMS if t in pred]
    if ref_path and not any(t in pred_path for t in ref_path):
        return "missed_pathology"

    corruption_signals = ['0','1','@','$','|','3','!','-']
    noisy_tokens = noisy.split()
    corrupted = sum(1 for tok in noisy_tokens
                    if any(s in tok for s in corruption_signals))
    if corrupted > 2 and entry["rouge_l"] < 0.3:
        return "noise_corrupted_output"

    if 0.2 <= entry["rouge_l"] < 0.5:
        return "partial_match"

    if entry["rouge_l"] >= 0.5:
        return "correct"

    return "other_failure"


def log_predictions(original_records, noisy_records, predictions,
                    condition, noise_level, output_dir="artifacts/logs"):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{condition}_noise{int(noise_level*100)}.json"
    path  = os.path.join(output_dir, fname)
    entries = []
    scorer = rouge_scorer_module.RougeScorer(['rougeL'], use_stemmer=True)
    for orig, noisy, pred in zip(original_records, noisy_records, predictions):
        ref   = orig["impression"]
        score = scorer.score(ref, pred)['rougeL'].fmeasure
        entries.append({
            "original_findings": orig["findings"],
            "noisy_findings":    noisy["findings"],
            "reference":         ref,
            "prediction":        pred,
            "rouge_l":           round(score, 4),
            "noise_level":       noise_level,
        })
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    return entries


def generate_plots(all_results, taxonomy):
    """Generate all evaluation plots and save to artifacts/plots/"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [WARN] matplotlib not installed. Skipping plots.")
        print("  Run: pip install matplotlib")
        return

    os.makedirs("artifacts/plots", exist_ok=True)
    NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]
    NOISE_LABELS = ["5%", "10%", "15%", "20%"]

    # ── Colors ───────────────────────────────────────────────────
    CLR_CLEAN  = "#2196F3"
    CLR_NOISY  = "#F44336"
    CLR_RAG    = "#4CAF50"
    CLR_BERT   = "#9C27B0"

    # ── Plot 1: ROUGE-L Degradation Curve ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    clean_r = next(r for r in all_results
                   if r['condition'] == 'clean_baseline')['rouge_l_f']
    noisy_r = [next(r for r in all_results
                    if r['condition'] == 'noisy_baseline'
                    and r['noise_level'] == l)['rouge_l_f']
               for l in NOISE_LEVELS]
    rag_r   = [next(r for r in all_results
                    if r['condition'] == 'rag_enhanced_sweep'
                    and r['noise_level'] == l)['rouge_l_f']
               for l in NOISE_LEVELS]

    ax.axhline(y=clean_r, color=CLR_CLEAN, linestyle='--',
               linewidth=2, label=f'Clean Baseline ({clean_r:.4f})')
    ax.plot(NOISE_LABELS, noisy_r, 'o-', color=CLR_NOISY,
            linewidth=2, markersize=8, label='Noisy Baseline')
    ax.plot(NOISE_LABELS, rag_r, 's-', color=CLR_RAG,
            linewidth=2, markersize=8, label='RAG Enhanced')

    ax.fill_between(range(len(NOISE_LEVELS)),
                    noisy_r, rag_r, alpha=0.1, color=CLR_RAG)

    ax.set_xlabel('OCR Noise Level', fontsize=12)
    ax.set_ylabel('ROUGE-L F1', fontsize=12)
    ax.set_title('ROUGE-L Performance vs OCR Noise Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(clean_r * 1.3, max(rag_r) * 1.2))
    plt.tight_layout()
    plt.savefig('artifacts/plots/rouge_degradation_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: rouge_degradation_curve.png")

    # ── Plot 2: ROUGE-L vs BERTScore Divergence ───────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    clean_b = next(r for r in all_results
                   if r['condition'] == 'clean_baseline')['bertscore_f1']
    noisy_b = [next(r for r in all_results
                    if r['condition'] == 'noisy_baseline'
                    and r['noise_level'] == l)['bertscore_f1']
               for l in NOISE_LEVELS]
    rag_b   = [next(r for r in all_results
                    if r['condition'] == 'rag_enhanced_sweep'
                    and r['noise_level'] == l)['bertscore_f1']
               for l in NOISE_LEVELS]

    # Normalize to clean baseline = 1.0
    noisy_r_norm = [v / clean_r for v in noisy_r]
    rag_r_norm   = [v / clean_r for v in rag_r]
    noisy_b_norm = [v / clean_b for v in noisy_b]
    rag_b_norm   = [v / clean_b for v in rag_b]

    ax.plot(NOISE_LABELS, noisy_r_norm, 'o-', color=CLR_NOISY,
            linewidth=2, markersize=8, label='ROUGE-L (Noisy)')
    ax.plot(NOISE_LABELS, noisy_b_norm, 'o--', color=CLR_BERT,
            linewidth=2, markersize=8, label='BERTScore (Noisy)')
    ax.plot(NOISE_LABELS, rag_r_norm, 's-', color=CLR_RAG,
            linewidth=2, markersize=8, label='ROUGE-L (RAG)')
    ax.plot(NOISE_LABELS, rag_b_norm, 's--', color="#00BCD4",
            linewidth=2, markersize=8, label='BERTScore (RAG)')

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5,
               label='Clean Baseline')
    ax.set_xlabel('OCR Noise Level', fontsize=12)
    ax.set_ylabel('Score (Normalized to Clean Baseline)', fontsize=12)
    ax.set_title('ROUGE-L vs BERTScore Divergence Under Noise', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/plots/metric_divergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: metric_divergence.png")

    # ── Plot 3: RAG Top-k Ablation ────────────────────────────────
    topk_vals = [1, 3, 5]
    topk_rouge = [next(r for r in all_results
                       if r['condition'] == 'rag_enhanced'
                       and r['top_k'] == k)['rouge_l_f']
                  for k in topk_vals]
    topk_bert  = [next(r for r in all_results
                       if r['condition'] == 'rag_enhanced'
                       and r['top_k'] == k)['bertscore_f1']
                  for k in topk_vals]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(topk_vals))
    bars = ax1.bar(x - 0.2, topk_rouge, 0.35, label='ROUGE-L',
                   color=CLR_RAG, alpha=0.85)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, topk_bert, 0.35, label='BERTScore',
                    color=CLR_BERT, alpha=0.85)

    ax1.set_xlabel('RAG Top-k Retrieved Examples', fontsize=12)
    ax1.set_ylabel('ROUGE-L F1', fontsize=12, color=CLR_RAG)
    ax2.set_ylabel('BERTScore F1', fontsize=12, color=CLR_BERT)
    ax1.set_title('Effect of RAG Top-k on Performance (10% Noise)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k={k}' for k in topk_vals])

    handles = [mpatches.Patch(color=CLR_RAG, label='ROUGE-L'),
               mpatches.Patch(color=CLR_BERT, label='BERTScore')]
    ax1.legend(handles=handles, fontsize=10)
    plt.tight_layout()
    plt.savefig('artifacts/plots/rag_topk_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: rag_topk_ablation.png")

    # ── Plot 4: Failure Taxonomy Pie Chart ────────────────────────
    labels = list(taxonomy.keys())
    sizes  = list(taxonomy.values())
    colors = ['#F44336','#4CAF50','#FF9800','#2196F3','#9C27B0','#00BCD4']

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors[:len(labels)],
        autopct='%1.1f%%', startangle=140,
        pctdistance=0.82, wedgeprops=dict(width=0.6)
    )
    for at in autotexts:
        at.set_fontsize(10)

    ax.legend(wedges, [f"{l} ({v})" for l, v in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    ax.set_title('Failure Taxonomy Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('artifacts/plots/failure_taxonomy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: failure_taxonomy.png")

    # ── Plot 5: Summary Bar Chart (Clean vs Noisy vs RAG at 10%) ──
    conditions  = ['Clean\nBaseline', 'Noisy\nBaseline\n(10%)', 'RAG\nEnhanced\n(10%)']
    rouge_vals  = [
        next(r for r in all_results if r['condition']=='clean_baseline')['rouge_l_f'],
        next(r for r in all_results if r['condition']=='noisy_baseline' and r['noise_level']==0.10)['rouge_l_f'],
        next(r for r in all_results if r['condition']=='rag_enhanced' and r['noise_level']==0.10 and r['top_k']==3)['rouge_l_f'],
    ]
    bert_vals = [
        next(r for r in all_results if r['condition']=='clean_baseline')['bertscore_f1'],
        next(r for r in all_results if r['condition']=='noisy_baseline' and r['noise_level']==0.10)['bertscore_f1'],
        next(r for r in all_results if r['condition']=='rag_enhanced' and r['noise_level']==0.10 and r['top_k']==3)['bertscore_f1'],
    ]

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - 0.2, rouge_vals, 0.35, label='ROUGE-L',
                color=[CLR_CLEAN, CLR_NOISY, CLR_RAG], alpha=0.85)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, bert_vals, 0.35, label='BERTScore',
                 color=[CLR_CLEAN, CLR_NOISY, CLR_RAG], alpha=0.45,
                 hatch='//')

    ax.set_ylabel('ROUGE-L F1', fontsize=12)
    ax2.set_ylabel('BERTScore F1', fontsize=12)
    ax.set_title('Performance Comparison Across Conditions', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)

    handles = [mpatches.Patch(color='gray', label='ROUGE-L (solid)'),
               mpatches.Patch(color='gray', alpha=0.45,
                              hatch='//', label='BERTScore (hatched)')]
    ax.legend(handles=handles, fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    for bar, val in zip(b1, rouge_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('artifacts/plots/condition_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: condition_comparison.png")

    print(f"\n  All plots saved to artifacts/plots/")


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Radiology Report Summarization — Evaluation & Diagnostic Pipeline")
    print("=" * 65)

    # 1. Load Data
    print("\n[1/6] Loading dataset...")
    records = load_openi_dataset(max_samples=200, seed=SEED)
    splits  = split_data(records, seed=SEED)
    save_split(splits)
    train_records = splits["train"]
    test_records  = splits["test"]
    print(f"  Train: {len(train_records)} | Test: {len(test_records)}")

    # 2. Load Summarizer
    print("\n[2/6] Loading summarization model...")
    summarizer = load_summarizer("facebook/bart-large-cnn")

    # 3. Load RAG
    print("\n[3/6] Building RAG index...")
    rag = SimpleRAG(embedding_model="all-MiniLM-L6-v2")
    rag.build_index(train_records)

    references   = [r["impression"] for r in test_records]
    NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]
    RAG_TOPK     = [1, 3, 5]
    all_results  = []
    all_predictions = []

    # 4. Clean Baseline
    print("\n[4/6] Running clean baseline...")
    clean_preds   = summarize_batch(summarizer, test_records)
    clean_rouge   = compute_rouge_l(clean_preds, references)
    clean_bert    = compute_bertscore(clean_preds, references)
    clean_result  = {"condition": "clean_baseline", "noise_level": 0.0,
                     "top_k": None, **clean_rouge, **clean_bert}
    all_results.append(clean_result)
    print(f"  ROUGE-L: {clean_rouge['rouge_l_f']}  BERTScore: {clean_bert['bertscore_f1']}")

    # 5. Noise sweep + RAG sweep
    print("\n[5/6] Running noise sweep and RAG evaluation...")
    noisy_by_level = {}

    for level in NOISE_LEVELS:
        print(f"\n  Noise level {int(level*100)}%...")

        # Inject noise
        noisy_recs = inject_noise_batch(test_records, noise_level=level, seed=SEED)
        noisy_by_level[level] = noisy_recs

        # Noisy baseline
        noisy_preds  = summarize_batch(summarizer, noisy_recs)
        noisy_rouge  = compute_rouge_l(noisy_preds, references)
        noisy_bert   = compute_bertscore(noisy_preds, references)
        noisy_result = {"condition": "noisy_baseline", "noise_level": level,
                        "top_k": None, **noisy_rouge, **noisy_bert}
        all_results.append(noisy_result)
        print(f"    Noisy   — ROUGE-L: {noisy_rouge['rouge_l_f']}  BERTScore: {noisy_bert['bertscore_f1']}")

        # Log predictions for error analysis
        entries = log_predictions(test_records, noisy_recs, noisy_preds,
                                  "noisy_baseline", level)
        all_predictions.extend(entries)

        # RAG enhanced (top_k=3 for sweep)
        rag_aug   = augment_records_with_rag(rag, noisy_recs, top_k=3)
        rag_preds = summarize_batch(summarizer, rag_aug)
        rag_rouge = compute_rouge_l(rag_preds, references)
        rag_bert  = compute_bertscore(rag_preds, references)
        rag_result = {"condition": "rag_enhanced_sweep", "noise_level": level,
                      "top_k": 3, **rag_rouge, **rag_bert}
        all_results.append(rag_result)
        print(f"    RAG(3)  — ROUGE-L: {rag_rouge['rouge_l_f']}  BERTScore: {rag_bert['bertscore_f1']}")

    # RAG top-k ablation at 10% noise
    print(f"\n  RAG top-k ablation at 10% noise...")
    noisy_10 = noisy_by_level[0.10]
    for topk in RAG_TOPK:
        rag_aug   = augment_records_with_rag(rag, noisy_10, top_k=topk)
        rag_preds = summarize_batch(summarizer, rag_aug)
        rag_rouge = compute_rouge_l(rag_preds, references)
        rag_bert  = compute_bertscore(rag_preds, references)
        result    = {"condition": "rag_enhanced", "noise_level": 0.10,
                     "top_k": topk, **rag_rouge, **rag_bert}
        all_results.append(result)
        entries = log_predictions(test_records, noisy_10, rag_preds,
                                  f"rag_top{topk}", 0.10)
        all_predictions.extend(entries)
        print(f"    RAG top-{topk} — ROUGE-L: {rag_rouge['rouge_l_f']}  BERTScore: {rag_bert['bertscore_f1']}")

    # 6. Failure taxonomy
    print("\n[6/6] Classifying failures and generating plots...")
    taxonomy = {}
    for entry in all_predictions:
        label = classify_failure(entry)
        taxonomy[label] = taxonomy.get(label, 0) + 1

    total = sum(taxonomy.values())
    print("\n  Failure Taxonomy:")
    for label, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
        print(f"    {label:<30} {count:>4}  ({100*count/total:.1f}%)")

    # Save results
    os.makedirs("artifacts/logs", exist_ok=True)
    with open("artifacts/logs/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("artifacts/logs/failure_taxonomy.json", "w") as f:
        json.dump({"taxonomy": taxonomy, "total": total}, f, indent=2)

    # Generate plots
    print("\n  Generating plots...")
    generate_plots(all_results, taxonomy)

    # Summary table
    clean_r = next(r for r in all_results if r['condition']=='clean_baseline')['rouge_l_f']
    clean_b = next(r for r in all_results if r['condition']=='clean_baseline')['bertscore_f1']

    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Condition':<25} {'Noise':>6} {'TopK':>5} {'ROUGE-L':>9} {'BERTScore':>10}")
    print("-" * 65)
    for r in all_results:
        topk_str  = str(r['top_k']) if r['top_k'] else "—"
        noise_str = f"{int(r['noise_level']*100)}%" if r['noise_level'] > 0 else "0%"
        print(f"{r['condition']:<25} {noise_str:>6} {topk_str:>5} "
              f"{r['rouge_l_f']:>9.4f} {r['bertscore_f1']:>10.4f}")

    print("\n" + "=" * 65)
    print("KEY FINDINGS")
    print("=" * 65)
    print(f"Clean baseline — ROUGE-L: {clean_r:.4f}  BERTScore: {clean_b:.4f}")
    for level in NOISE_LEVELS:
        nr = next(r for r in all_results if r['condition']=='noisy_baseline'
                  and r['noise_level']==level)
        rr = next(r for r in all_results if r['condition']=='rag_enhanced_sweep'
                  and r['noise_level']==level)
        deg = (clean_r - nr['rouge_l_f']) / clean_r * 100
        rec = (rr['rouge_l_f'] - nr['rouge_l_f']) / nr['rouge_l_f'] * 100
        print(f"Noise {int(level*100)}%: ROUGE-L={nr['rouge_l_f']:.4f}  "
              f"Degradation={deg:.1f}%  RAG recovery={rec:.1f}%")

    print(f"\nResults saved to artifacts/logs/")
    print(f"Plots saved to artifacts/plots/")
    print("Done!")


if __name__ == "__main__":
    main()