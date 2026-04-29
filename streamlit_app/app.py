"""
Streamlit Demo: Radiology Report Summarization under OCR Noise + RAG
CS818 Project — Maria Mahmood & Noor-ul-Ain Khalid
"""

import streamlit as st
import random
import time

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RadSum — OCR Noise & RAG Demo",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0d1117;
    --bg2:       #161b22;
    --bg3:       #21262d;
    --border:    #30363d;
    --accent:    #58a6ff;
    --green:     #3fb950;
    --orange:    #f0883e;
    --red:       #f85149;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --mono:      'IBM Plex Mono', monospace;
    --serif:     'DM Serif Display', serif;
    --sans:      'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Title area ── */
.hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: var(--serif) !important;
    font-size: 2.6rem !important;
    color: var(--text) !important;
    margin-bottom: .25rem !important;
    letter-spacing: -.5px;
}
.hero p {
    color: var(--muted);
    font-size: .95rem;
    font-family: var(--sans) !important;
}

/* ── Stage badge ── */
.stage-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: .25rem .75rem;
    border-radius: 20px;
    margin-bottom: .75rem;
}
.stage-clean   { background: #1c2a1c; color: var(--green);  border: 1px solid #2ea04326; }
.stage-noisy   { background: #2a1c1c; color: var(--red);    border: 1px solid #f8514926; }
.stage-rag     { background: #1c1f2a; color: var(--accent); border: 1px solid #58a6ff26; }
.stage-eval    { background: #2a211c; color: var(--orange); border: 1px solid #f0883e26; }

/* ── Cards ── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card h4 {
    font-family: var(--mono) !important;
    font-size: .8rem !important;
    color: var(--muted) !important;
    margin-bottom: .5rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── Monospace text blocks ── */
.code-block {
    font-family: var(--mono);
    font-size: .82rem;
    line-height: 1.6;
    color: var(--text);
    background: var(--bg3);
    border-left: 3px solid var(--border);
    padding: .8rem 1rem;
    border-radius: 0 6px 6px 0;
    white-space: pre-wrap;
    word-break: break-word;
}
.code-block.noisy { border-left-color: var(--red); }
.code-block.rag   { border-left-color: var(--accent); }
.code-block.out   { border-left-color: var(--green); }

/* ── Noise highlight ── */
.noise-char { color: var(--red); font-weight: 600; }

/* ── Metric pills ── */
.metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: .5rem; }
.metric-pill {
    display: flex; flex-direction: column; align-items: center;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 8px; padding: .6rem 1.2rem;
    min-width: 120px;
}
.metric-pill .label {
    font-family: var(--mono); font-size: .65rem;
    color: var(--muted); text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: .2rem;
}
.metric-pill .value {
    font-family: var(--mono); font-size: 1.4rem; font-weight: 600;
}
.good  { color: var(--green); }
.mid   { color: var(--orange); }
.bad   { color: var(--red); }

/* ── Arrow connector ── */
.arrow {
    text-align: center; font-size: 1.4rem;
    color: var(--muted); margin: .25rem 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: .5rem 1.5rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Selectbox / sliders ── */
.stSelectbox label, .stSlider label, .stTextArea label {
    font-family: var(--mono) !important;
    font-size: .8rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
.stTextArea textarea {
    font-family: var(--mono) !important;
    font-size: .82rem !important;
    background: var(--bg3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }

/* ── Section divider ── */
.section-header {
    font-family: var(--mono);
    font-size: .7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding: .5rem 0;
    border-bottom: 1px solid var(--border);
    margin: 1.5rem 0 1rem 0;
}

/* ── Flow diagram ── */
.flow {
    display: flex;
    align-items: center;
    gap: .5rem;
    overflow-x: auto;
    padding: .75rem 0;
    margin-bottom: 1.5rem;
}
.flow-step {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .5rem 1rem;
    font-family: var(--mono);
    font-size: .75rem;
    text-align: center;
    white-space: nowrap;
    flex-shrink: 0;
}
.flow-arrow { color: var(--muted); font-size: 1rem; flex-shrink: 0; }
.flow-step.active-clean  { border-color: var(--green);  color: var(--green);  }
.flow-step.active-noisy  { border-color: var(--red);    color: var(--red);    }
.flow-step.active-rag    { border-color: var(--accent); color: var(--accent); }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: .75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: .75rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Info / warning boxes ── */
.stInfo, .stWarning, .stSuccess, .stError {
    font-family: var(--mono) !important;
    font-size: .82rem !important;
}

/* ── Horizontal rule ── */
hr { border-color: var(--border) !important; }

/* ── Comparison table ── */
.compare-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: .8rem; }
.compare-table th { color: var(--muted); font-size: .68rem; text-transform: uppercase; letter-spacing: 1.5px;
                    border-bottom: 1px solid var(--border); padding: .5rem .75rem; text-align: left; }
.compare-table td { padding: .5rem .75rem; border-bottom: 1px solid #21262d; vertical-align: top; }
.compare-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ── Imports from your actual src/ modules ─────────────────────────
import sys, os, time

# Add the project root (one level above streamlit_app/) to sys.path
# so that `src/` is importable exactly as your pipeline scripts do it.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# noise_injection.py  — inject_noise(), inject_noise_batch(), OCR_SUBSTITUTIONS
from src.data.noise_injection import inject_noise, OCR_SUBSTITUTIONS

# rag_model.py        — SimpleRAG (SentenceTransformer cosine similarity)
from src.models.rag_model import SimpleRAG

# run_a3_diagnostics  — mock_summarize() rule-based summarizer
#   We import it directly so it is the exact same function used in your pipeline.
import importlib.util, types
_diag_path = os.path.join(PROJECT_ROOT, "scripts", "run_a3_diagnostics.py")
_spec = importlib.util.spec_from_file_location("run_a3_diagnostics", _diag_path)
_diag_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag_mod)
mock_summarize = _diag_mod.mock_summarize

# metrics.py          — compute_rouge_l() using rouge_score library
from src.evaluation.metrics import compute_rouge_l

# load_data.py        — to read your saved train split as the RAG knowledge base
from src.data.load_data import _synthetic_fallback
import json

# ── Load knowledge base (use saved train split if available) ───────
@st.cache_resource
def load_knowledge_base():
    train_path = os.path.join(PROJECT_ROOT, "artifacts", "data", "train.json")
    if os.path.exists(train_path):
        with open(train_path) as f:
            kb = json.load(f)
        return kb
    # fall back to synthetic data (same as load_data._synthetic_fallback)
    return _synthetic_fallback()

@st.cache_resource
def load_rag(kb):
    rag = SimpleRAG(embedding_model="all-MiniLM-L6-v2")
    rag.build_index(kb)
    return rag

# ── Sample findings (from your actual test.json records) ──────────
SAMPLE_FINDINGS = {
    "Normal Chest":
        "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax is identified. "
        "The cardiomediastinal silhouette is within normal limits. Osseous structures are intact.",
    "Pneumonia":
        "There is increased opacity in the right lower lobe consistent with pneumonia. The left lung is clear. "
        "Mild cardiomegaly is present. No pleural effusion identified.",
    "COPD / Emphysema":
        "No acute osseous abnormality. Lungs are hyperinflated. Flattening of the diaphragm noted. "
        "No focal consolidation. No pneumothorax.",
    "Congestive Heart Failure":
        "The cardiac silhouette is enlarged. Bilateral pleural effusions are present, left greater than right. "
        "Pulmonary vascular congestion noted.",
    "Pneumothorax":
        "Right-sided pneumothorax identified with partial collapse of the right lung. "
        "Tracheal deviation to the left. No rib fractures identified.",
    "Atelectasis":
        "Linear opacities at the left base consistent with subsegmental atelectasis. "
        "No pneumonia or pleural effusion. Heart size normal.",
    "Pulmonary Edema":
        "Diffuse bilateral airspace opacities consistent with pulmonary edema. "
        "Cardiomegaly noted. Bilateral pleural effusions present.",
}

SAMPLE_REFERENCES = {
    "Normal Chest":            "No acute cardiopulmonary abnormality.",
    "Pneumonia":               "Right lower lobe pneumonia. Mild cardiomegaly.",
    "COPD / Emphysema":        "Hyperinflation consistent with chronic obstructive pulmonary disease (COPD).",
    "Congestive Heart Failure":"Cardiomegaly with bilateral pleural effusions and pulmonary vascular congestion, consistent with congestive heart failure.",
    "Pneumothorax":            "Right-sided pneumothorax with partial lung collapse.",
    "Atelectasis":             "Subsegmental atelectasis at the left base. No acute pneumonia.",
    "Pulmonary Edema":         "Pulmonary edema with cardiomegaly and bilateral pleural effusions.",
}

# ── Helper ────────────────────────────────────────────────────────
def color_class(val: float) -> str:
    if val >= 0.35: return "good"
    if val >= 0.15: return "mid"
    return "bad"

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫁 RadSum Demo")
    st.markdown("<p style='font-family:IBM Plex Mono;font-size:.75rem;color:#8b949e;'>CS818 · LLM Project</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Select Sample**")
    sample_choice = st.selectbox("Finding type", list(SAMPLE_FINDINGS.keys()), label_visibility="collapsed")

    st.markdown("**Noise Level**")
    noise_level = st.slider("OCR corruption %", 0, 30, 10, step=5, label_visibility="collapsed") / 100

    st.markdown("**RAG top-k**")
    top_k = st.slider("Retrieved examples", 1, 5, 3, label_visibility="collapsed")

    st.markdown("**Custom Input**")
    use_custom = st.toggle("Use custom findings text", value=False)

    seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)

    st.divider()
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:.68rem;color:#8b949e;line-height:1.7;'>
    <b style='color:#e6edf3'>Pipeline stages</b><br>
    1 · Raw findings input<br>
    2 · OCR noise injection<br>
    3 · RAG context retrieval<br>
    4 · Summarization model<br>
    5 · Metric evaluation
    </div>
    """, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Radiology Summarization under OCR Noise</h1>
  <p>Visualizing how OCR-style text corruption degrades LLM summarization — and how RAG recovers it.</p>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE FLOW ─────────────────────────────────────────────────
st.markdown("""
<div class="flow">
  <div class="flow-step">📄 Findings Text</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step active-noisy">⚡ OCR Noise</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step active-rag">🔍 RAG Retrieval</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">🤖 Summarizer</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step active-clean">✓ Impression</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step active-clean">📊 Metrics</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────
if use_custom:
    findings_text = st.text_area(
        "Enter radiology findings",
        value=SAMPLE_FINDINGS[sample_choice],
        height=120,
        key="custom_input"
    )
    reference_impression = st.text_input(
        "Reference impression (ground truth)",
        value=SAMPLE_REFERENCES[sample_choice]
    )
else:
    findings_text        = SAMPLE_FINDINGS[sample_choice]
    reference_impression = SAMPLE_REFERENCES[sample_choice]

run_btn = st.button("▶  Run Pipeline", use_container_width=False)

if not run_btn and "pipeline_ran" not in st.session_state:
    st.markdown("""
    <div class="card" style="border-color:#30363d;margin-top:1rem;">
      <h4>How to use</h4>
      <p style="font-family:'IBM Plex Mono';font-size:.8rem;color:#8b949e;line-height:1.7;">
      1. Choose a sample finding type from the sidebar (or toggle custom input).<br>
      2. Adjust the noise level and RAG top-k.<br>
      3. Click <b style="color:#58a6ff">▶ Run Pipeline</b> to see each stage.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── RUN PIPELINE using your exact src/ modules ────────────────────
st.session_state["pipeline_ran"] = True

with st.spinner("Loading knowledge base & building RAG index…"):
    kb  = load_knowledge_base()
    rag = load_rag(kb) # hashable key for cache

with st.spinner("Running pipeline…"):

    # ── Stage 1: Clean baseline ───────────────────────────────────
    # inject_noise() from src/data/noise_injection.py  (returns str, no tuple)
    clean_pred  = mock_summarize(findings_text)
    # compute_rouge_l() from src/evaluation/metrics.py
    clean_metrics = compute_rouge_l([clean_pred], [reference_impression])
    clean_rouge   = clean_metrics["rouge_l_f"]

    # ── Stage 2: Noisy baseline ───────────────────────────────────
    # inject_noise() signature: (text, noise_level, seed) → str
    noisy_text  = inject_noise(findings_text, noise_level=noise_level, seed=int(seed))
    noisy_pred  = mock_summarize(noisy_text)
    noisy_metrics = compute_rouge_l([noisy_pred], [reference_impression])
    noisy_rouge   = noisy_metrics["rouge_l_f"]

    # ── Stage 3: RAG enhanced ─────────────────────────────────────
    # SimpleRAG.retrieve() uses SentenceTransformer cosine similarity
    # SimpleRAG.build_rag_prompt() prepends retrieved examples
    rag_prompt_str = rag.build_rag_prompt(noisy_text, top_k=top_k)
    retrieved_raw  = rag.retrieve(noisy_text, top_k=top_k)   # list of dicts
    # Build (similarity_placeholder, doc) pairs for display
    # (SimpleRAG doesn't expose raw scores, so we compute word-overlap for display only)
    def _jaccard(a, b):
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    retrieved_docs = [(_jaccard(noisy_text, r["findings"]), r) for r in retrieved_raw]

    rag_record  = {"findings": rag_prompt_str, "impression": reference_impression}
    rag_pred    = mock_summarize(rag_prompt_str)
    rag_metrics = compute_rouge_l([rag_pred], [reference_impression])
    rag_rouge   = rag_metrics["rouge_l_f"]

    # ── Derived stats ─────────────────────────────────────────────
    degradation = (clean_rouge - noisy_rouge) / clean_rouge * 100 if clean_rouge > 0 else 0.0
    recovery    = (rag_rouge - noisy_rouge)   / noisy_rouge * 100 if noisy_rouge > 0 else 0.0

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Stage 1 · Clean", "Stage 2 · Noisy", "Stage 3 · RAG", "Results"
])

# ─── TAB 1: CLEAN ────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="stage-badge stage-clean">Stage 1 — Clean Baseline</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Input · Findings</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block">{findings_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Reference Impression</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block out">{reference_impression}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Predicted Impression</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block out">{clean_pred}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Metrics</div>', unsafe_allow_html=True)
        cc = color_class(clean_rouge)
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-pill">
            <span class="label">ROUGE-L F1</span>
            <span class="value {cc}">{clean_rouge:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Precision</span>
            <span class="value {cc}">{clean_metrics['rouge_l_p']:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Recall</span>
            <span class="value {cc}">{clean_metrics['rouge_l_r']:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Noise Level</span>
            <span class="value good">0%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("ℹ️  Clean input = best-case performance. No corruption applied.", icon=None)

# ─── TAB 2: NOISY ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="stage-badge stage-noisy">Stage 2 — Noisy Baseline</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Original Findings</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block">{findings_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">How noise is injected — src/data/noise_injection.py</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="margin-bottom:0;">
          <table class="compare-table">
            <tr><th>Operation</th><th>Example</th></tr>
            <tr><td>Character substitution</td><td><span style='color:#f85149'>e→3, o→0, s→$</span></td></tr>
            <tr><td>Deletion</td><td><span style='color:#f85149'>remove char entirely</span></td></tr>
            <tr><td>Fragmentation</td><td><span style='color:#f85149'>insert spurious hyphen</span></td></tr>
            <tr><td>Extra space</td><td><span style='color:#f85149'>insert random space</span></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-header">Corrupted Findings @ {int(noise_level*100)}% noise</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block noisy">{noisy_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Predicted Impression</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block noisy">{noisy_pred}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Metrics</div>', unsafe_allow_html=True)
        nc = color_class(noisy_rouge)
        deg_color = "bad" if degradation > 20 else "mid" if degradation > 5 else "good"
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-pill">
            <span class="label">ROUGE-L F1</span>
            <span class="value {nc}">{noisy_rouge:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Precision</span>
            <span class="value {nc}">{noisy_metrics['rouge_l_p']:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Recall</span>
            <span class="value {nc}">{noisy_metrics['rouge_l_r']:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Degradation</span>
            <span class="value {deg_color}">−{degradation:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if degradation > 20:
            st.error(f"⚠️  Significant degradation ({degradation:.1f}%) from noise.")
        elif degradation > 5:
            st.warning(f"Moderate degradation ({degradation:.1f}%) from OCR noise.")
        else:
            st.success(f"Low degradation ({degradation:.1f}%) — model was robust at this noise level.")

# ─── TAB 3: RAG ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="stage-badge stage-rag">Stage 3 — RAG Enhanced</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">Retrieved Knowledge Base Examples (top-{top_k}) — src/models/rag_model.py · SimpleRAG</div>', unsafe_allow_html=True)

    for i, (sim, doc) in enumerate(retrieved_docs):
        sim_pct = int(sim * 100)
        with st.expander(f"Example {i+1} — word-overlap similarity: {sim_pct}%", expanded=(i == 0)):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Findings**")
                st.markdown(f'<div class="code-block rag">{doc["findings"]}</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown("**Impression**")
                st.markdown(f'<div class="code-block out">{doc["impression"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Augmented Prompt sent to model</div>', unsafe_allow_html=True)
    prompt_preview = rag_prompt_str[:600] + ("…" if len(rag_prompt_str) > 600 else "")
    st.markdown(f'<div class="code-block rag">{prompt_preview}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">RAG-Enhanced Prediction</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(f'<div class="code-block out">{rag_pred}</div>', unsafe_allow_html=True)
    with col2:
        rc = color_class(rag_rouge)
        rec_color = "good" if recovery > 0 else "bad"
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-pill">
            <span class="label">ROUGE-L F1</span>
            <span class="value {rc}">{rag_rouge:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Precision</span>
            <span class="value {rc}">{rag_metrics['rouge_l_p']:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Recall</span>
            <span class="value {rc}">{rag_metrics['rouge_l_r']:.4f}</span>
          </div>
          <div class="metric-pill">
            <span class="label">RAG top-k</span>
            <span class="value" style="color:#58a6ff;">{top_k}</span>
          </div>
          <div class="metric-pill">
            <span class="label">Recovery</span>
            <span class="value {rec_color}">{'+' if recovery >= 0 else ''}{recovery:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 4: RESULTS ──────────────────────────────────────────────
with tab4:
    st.markdown('<div class="stage-badge stage-eval">Results Summary</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Condition Comparison</div>', unsafe_allow_html=True)

    def rouge_bar(val):
        width = int(val * 400)
        color_map = {"good": "#3fb950", "mid": "#f0883e", "bad": "#f85149"}
        c = color_class(val)
        return f'<div style="height:8px;width:{width}px;background:{color_map[c]};border-radius:4px;margin-top:4px;"></div>'

    st.markdown(f"""
    <table class="compare-table">
      <tr>
        <th>Condition</th>
        <th>Input</th>
        <th>ROUGE-L F1</th>
        <th>vs Clean</th>
        <th>Impression</th>
      </tr>
      <tr>
        <td><span class="stage-badge stage-clean" style="margin:0;padding:.15rem .5rem;font-size:.65rem;">Clean</span></td>
        <td style="color:#8b949e;">No noise</td>
        <td>
          <span class="{color_class(clean_rouge)}">{clean_rouge:.4f}</span>
          {rouge_bar(clean_rouge)}
        </td>
        <td style="color:#8b949e;">baseline</td>
        <td style="font-family:'IBM Plex Mono';font-size:.75rem;">{clean_pred}</td>
      </tr>
      <tr>
        <td><span class="stage-badge stage-noisy" style="margin:0;padding:.15rem .5rem;font-size:.65rem;">Noisy</span></td>
        <td style="color:#f85149;">{int(noise_level*100)}% OCR noise</td>
        <td>
          <span class="{color_class(noisy_rouge)}">{noisy_rouge:.4f}</span>
          {rouge_bar(noisy_rouge)}
        </td>
        <td class="{'bad' if degradation > 10 else 'mid'}">−{degradation:.1f}%</td>
        <td style="font-family:'IBM Plex Mono';font-size:.75rem;">{noisy_pred}</td>
      </tr>
      <tr>
        <td><span class="stage-badge stage-rag" style="margin:0;padding:.15rem .5rem;font-size:.65rem;">RAG (k={top_k})</span></td>
        <td style="color:#58a6ff;">{int(noise_level*100)}% + retrieval</td>
        <td>
          <span class="{color_class(rag_rouge)}">{rag_rouge:.4f}</span>
          {rouge_bar(rag_rouge)}
        </td>
        <td class="{'good' if rag_rouge >= clean_rouge else 'mid'}">{'+' if rag_rouge >= clean_rouge else ''}{((rag_rouge-clean_rouge)/clean_rouge*100 if clean_rouge > 0 else 0.0):.1f}%</td>
        <td style="font-family:'IBM Plex Mono';font-size:.75rem;">{rag_pred}</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:2rem;">Key Takeaways</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card">
          <h4>Noise Degradation</h4>
          <div style="font-family:'IBM Plex Mono';font-size:2rem;color:{'#f85149' if degradation > 20 else '#f0883e' if degradation > 5 else '#3fb950'};">
            −{degradation:.1f}%
          </div>
          <p style="font-family:'IBM Plex Mono';font-size:.75rem;color:#8b949e;margin-top:.5rem;">
            ROUGE-L drop from clean → noisy at {int(noise_level*100)}% corruption
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
          <h4>RAG Recovery</h4>
          <div style="font-family:'IBM Plex Mono';font-size:2rem;color:{'#3fb950' if recovery > 0 else '#f85149'};">
            {'+' if recovery >= 0 else ''}{recovery:.1f}%
          </div>
          <p style="font-family:'IBM Plex Mono';font-size:.75rem;color:#8b949e;margin-top:.5rem;">
            ROUGE-L improvement from noisy → RAG (k={top_k})
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        chars_corrupted = max(1, int(len(findings_text) * noise_level))
        st.markdown(f"""
        <div class="card">
          <h4>Corruption Stats</h4>
          <div style="font-family:'IBM Plex Mono';font-size:2rem;color:#f0883e;">
            ~{chars_corrupted}
          </div>
          <p style="font-family:'IBM Plex Mono';font-size:.75rem;color:#8b949e;margin-top:.5rem;">
            Characters corrupted out of {len(findings_text)} total ({int(noise_level*100)}% level)
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Failure Taxonomy (from full experiment — artifacts/logs/failure_taxonomy.json)</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="compare-table">
      <tr><th>Category</th><th>Count</th><th>%</th><th>Description</th></tr>
      <tr><td style="color:#f85149;">noise_corrupted_output</td><td>90</td><td>42.9%</td><td>Noisy tokens directly appear in prediction</td></tr>
      <tr><td style="color:#f0883e;">missed_pathology</td><td>44</td><td>21.0%</td><td>Reference mentions disease; prediction doesn't</td></tr>
      <tr><td style="color:#8b949e;">other_failure</td><td>58</td><td>27.6%</td><td>Low ROUGE-L, unclassified mismatch</td></tr>
      <tr><td style="color:#58a6ff;">partial_match</td><td>15</td><td>7.1%</td><td>ROUGE-L 0.2–0.5, partial overlap</td></tr>
      <tr><td style="color:#3fb950;">generic_impression</td><td>3</td><td>1.4%</td><td>Safe fallback when specific finding expected</td></tr>
    </table>
    """, unsafe_allow_html=True)
