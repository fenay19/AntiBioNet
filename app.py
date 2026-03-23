"""
═══════════════════════════════════════════════════════════════════
  AMR-Insight  |  Antibiotic Resistance Clinical Decision Dashboard
  Streamlit + Gen AI  |  Hackathon Edition
═══════════════════════════════════════════════════════════════════
  Run:
      streamlit run app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Ensure project root is on sys.path so local packages (views, etc.) resolve
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json, io

# ── Local modules ────────────────────────────────────────────────
from controllers.data_controller   import DataController
from controllers.model_controller  import ModelController
from controllers.report_controller import ReportController
from utils.preprocessing           import ANTIBIOTIC_META
from llm                           import generate_explanation, chat_response

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "AMR-Insight",
    page_icon   = "🧬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════
#  GLOBAL CSS  – clinical dark theme with teal accents
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

:root {
  --bg0:    #0d1117;
  --bg1:    #161b22;
  --bg2:    #1c2230;
  --bg3:    #21293a;
  --teal:   #00e5c3;
  --teal2:  #00b59a;
  --amber:  #ffb830;
  --red:    #ff5c6b;
  --green:  #3ddc84;
  --muted:  #8b949e;
  --text:   #e6edf3;
  --border: #30363d;
}

html, body, [class*="css"] {
  font-family: 'Sora', sans-serif !important;
  background-color: var(--bg0) !important;
  color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Main header */
.amr-header {
  background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px 32px;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 20px;
}
.amr-header h1 { font-size: 2rem; font-weight: 700; margin: 0; color: var(--teal); letter-spacing: -0.5px; }
.amr-header p  { margin: 4px 0 0; color: var(--muted); font-size: 0.9rem; }

/* Metric cards */
.metric-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  text-align: center;
}
.metric-label  { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value  { font-size: 2rem; font-weight: 700; font-family: 'DM Mono', monospace; }
.metric-sub    { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }

/* Risk badge */
.badge-low  { background:#1a3a2a; color:var(--green);  border:1px solid #3ddc8444; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }
.badge-mod  { background:#3a2e10; color:var(--amber);  border:1px solid #ffb83044; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }
.badge-high { background:#3a1520; color:var(--red);    border:1px solid #ff5c6b44; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }

/* Section cards */
.section-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 16px;
}
.section-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--teal);
  margin-bottom: 14px;
  font-weight: 600;
}

/* AI output */
.ai-box {
  background: var(--bg2);
  border-left: 3px solid var(--teal);
  border-radius: 0 8px 8px 0;
  padding: 16px 20px;
  margin: 10px 0;
  font-size: 0.92rem;
  line-height: 1.7;
  white-space: pre-wrap;
}

/* Chat bubbles */
.chat-user {
  background: var(--bg3);
  border-radius: 12px 12px 2px 12px;
  padding: 10px 16px;
  margin: 8px 0 8px 40px;
  font-size: 0.9rem;
}
.chat-ai {
  background: #0d2e2a;
  border: 1px solid #00e5c322;
  border-radius: 12px 12px 12px 2px;
  padding: 10px 16px;
  margin: 8px 40px 8px 0;
  font-size: 0.9rem;
  line-height: 1.65;
}
.chat-label { font-size: 0.68rem; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }

/* Antibiotic pills */
.pill-avoid   { background:#3a1520; color:var(--red);   border:1px solid #ff5c6b55; border-radius:6px; padding:4px 10px; margin:3px; display:inline-block; font-size:0.8rem; }
.pill-suggest { background:#1a3a2a; color:var(--green); border:1px solid #3ddc8455; border-radius:6px; padding:4px 10px; margin:3px; display:inline-block; font-size:0.8rem; }
.pill-caution { background:#3a2e10; color:var(--amber); border:1px solid #ffb83055; border-radius:6px; padding:4px 10px; margin:3px; display:inline-block; font-size:0.8rem; }

/* Tabs */
button[data-baseweb="tab"] { color: var(--muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal) !important; border-bottom-color: var(--teal) !important; }

/* Inputs */
.stSlider > div > div > div { background: var(--teal) !important; }
.stSelectbox > div > div { background: var(--bg2) !important; border-color: var(--border) !important; }

/* Progress bar */
.stProgress > div > div { background: var(--teal) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  DATA & MODEL  (cached)
# ════════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Bacteria_dataset_Multiresictance.csv")

@st.cache_resource(show_spinner="🔬 Loading dataset & training models…")
def load_pipeline():
    data_ctrl = DataController(filepath=DATA_PATH)
    data_ctrl.load()
    data_ctrl.preprocess()
    data_ctrl.engineer_features()
    dataset = data_ctrl.get_dataset()

    model_ctrl = ModelController(dataset)
    model_ctrl.train_all()
    model_ctrl.evaluate_all()
    results = model_ctrl.get_results()

    report_ctrl = ReportController(dataset, results)
    return dataset, results, report_ctrl

try:
    dataset, results, report_ctrl = load_pipeline()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    LOAD_ERROR  = str(e)

best_model = None
if DATA_LOADED:
    best_model = max(results.values(), key=lambda r: r.auc)


# ════════════════════════════════════════════════════════════════
#  SIDEBAR  — patient inputs
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px;">
      <span style="font-size:2.2rem;">🧬</span>
      <div style="font-size:1.1rem;font-weight:700;color:#00e5c3;letter-spacing:1px;">AMR-INSIGHT</div>
      <div style="font-size:0.72rem;color:#8b949e;letter-spacing:2px;text-transform:uppercase;">Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1.5px;color:#00e5c3;font-weight:600;margin-bottom:12px;">Patient Profile</div>', unsafe_allow_html=True)

    age = st.number_input("Age (years)", min_value=0, max_value=110, value=45,
                          help="Patient age — constrained 0 to 110 years")

    gender = st.selectbox("Gender", ["Female", "Male"])

    st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1.5px;color:#00e5c3;font-weight:600;margin:14px 0 8px;">Comorbidities</div>', unsafe_allow_html=True)

    diabetes     = st.checkbox("Diabetes")
    hypertension = st.checkbox("Hypertension")
    prev_hosp    = st.checkbox("Prior Hospitalisation")

    st.markdown('<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:1.5px;color:#00e5c3;font-weight:600;margin:14px 0 8px;">Infection Details</div>', unsafe_allow_html=True)

    inf_freq = st.slider("Infection Frequency", 0, 10, 1,
                         help="Number of prior infection episodes")

    species_options = [
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Proteus mirabilis",
        "Pseudomonas aeruginosa",
        "Serratia marcescens",
        "Morganella morganii",
        "Citrobacter spp.",
        "Enterobacteria spp.",
    ]
    species = st.selectbox("Bacterial Species", species_options)

    st.markdown("---")

    # ── Gen AI settings ──────────────────────────────────────────
    
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="amr-header">
  <span style="font-size:2.8rem;">🧬</span>
  <div>
    <h1>AMR-Insight</h1>
    <p>Antibiotic Resistance Prediction & Clinical Decision Support  ·  Powered by ML + Generative AI</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  DATA-NOT-LOADED fallback
# ════════════════════════════════════════════════════════════════
if not DATA_LOADED:
    st.error(f"""
**Dataset not found.**  
Place `Bacteria_dataset_Multiresictance.csv` inside the `data/` folder and restart.

Error: `{LOAD_ERROR}`
""")
    st.stop()


# ════════════════════════════════════════════════════════════════
#  SUMMARY METRICS BAR
# ════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, f"{dataset.n_samples:,}",   "Total Isolates",     "Enterobacteriaceae"),
    (c2, f"{dataset.mdr_rate:.1%}",  "MDR Prevalence",     "≥ 3 drug classes"),
    (c3, f"{best_model.auc:.3f}",    "Best AUC",           best_model.name),
    (c4, "15",                        "Antibiotics Tested", "Drug classes"),
    (c5, f"{len(species_options)}",   "Species Covered",   "in dropdown"),
]
for col, val, lbl, sub in cards:
    with col:
        color = "#00e5c3" if lbl != "MDR Prevalence" else "#ff5c6b"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value" style="color:{color};">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SESSION STATE  — prediction store
# ════════════════════════════════════════════════════════════════
if "prediction"   not in st.session_state: st.session_state.prediction   = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "ai_text"      not in st.session_state: st.session_state.ai_text      = ""


# ════════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ════════════════════════════════════════════════════════════════
def run_prediction():
    from controllers.model_controller import ModelController
    from models.resistance_dataset    import ANTIBIOTIC_COLS

    fc   = best_model.feature_cols
    row  = {c: 0 for c in fc}
    row["age"]              = age
    row["gender_enc"]       = 1 if gender == "Male" else 0
    row["Diabetes_enc"]     = int(diabetes)
    row["Hypertension_enc"] = int(hypertension)
    row["Hospital_enc"]     = int(prev_hosp)
    row["Infection_Freq"]   = inf_freq

    sp_col = "sp_" + species.replace(" ", "_").replace(".", "_")
    if sp_col in row:
        row[sp_col] = 1

    X_new = pd.DataFrame([row])[fc]
    prob  = float(best_model.estimator.predict_proba(X_new)[0, 1])

    if   prob >= 0.75: risk = "HIGH";     badge = "badge-high"
    elif prob >= 0.50: risk = "MODERATE"; badge = "badge-mod"
    else:              risk = "LOW";      badge = "badge-low"

    # Feature importances — patient-level
    fi = best_model.feature_importances
    top_factors = fi.head(5).to_dict() if fi is not None else {}

    # Confidence = how far from 0.5
    confidence = abs(prob - 0.5) * 2

    return {
        "prob":        prob,
        "risk":        risk,
        "badge":       badge,
        "confidence":  confidence,
        "top_factors": top_factors,
        "patient": {
            "age": age, "gender": gender, "species": species,
            "diabetes": diabetes, "hypertension": hypertension,
            "prev_hosp": prev_hosp, "inf_freq": inf_freq,
        },
    }

if predict_btn:
    with st.spinner("Computing MDR probability…"):
        st.session_state.prediction = run_prediction()
    # Auto-generate AI explanation
    p = st.session_state.prediction
    with st.spinner("Generating AI clinical explanation…"):
        st.session_state.ai_text = generate_explanation(
            prediction  = p,
            features    = p["patient"],
            top_factors = p["top_factors"],
            provider    = ai_provider,
            api_key     = api_key,
        )


# ════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (must be defined before tabs call them)
# ════════════════════════════════════════════════════════════════

def get_antibiotic_recommendations(risk: str, species: str):
    """Rule-based antibiotic decision support."""
    HIGH_RES = ["AMX/AMP (Amoxicillin)", "AMC (Augmentin)", "CZ (Cefazolin)",
                "FOX (Cefoxitin)", "CTX/CRO (Ceftriaxone)", "IPM (Imipenem)"]
    MOD_RES  = ["GEN (Gentamicin)", "AN (Amikacin)"]
    LOW_RES  = ["CIP (Ciprofloxacin)", "OFX (Ofloxacin)", "Colistin", "Nitrofurantoin"]
    if risk == "HIGH":
        return HIGH_RES, LOW_RES[:2], MOD_RES
    elif risk == "MODERATE":
        return HIGH_RES[:3], LOW_RES + MOD_RES[:1], MOD_RES[1:]
    else:
        return HIGH_RES[:2], LOW_RES, MOD_RES


def render_network_graph():
    """Interactive co-resistance network using Plotly."""
    ab_cols = [
        "AMX/AMP","AMC","CZ","FOX","CTX/CRO","IPM",
        "GEN","AN","ofx","CIP","C","colistine"
    ]
    df = dataset.processed_df.copy()
    r_cols = [c for c in df.columns if c.endswith("_R") and c.replace("_R","") in ab_cols]
    if len(r_cols) < 4:
        st.info("Network graph requires processed resistance binary columns.")
        return
    corr = df[r_cols].corr()
    short = [c.replace("_R","") for c in r_cols]
    n = len(short)
    angles = [2 * np.pi * i / n for i in range(n)]
    pos_x  = [np.cos(a) for a in angles]
    pos_y  = [np.sin(a) for a in angles]
    resistance_rates_list = [dataset.resistance_rates.get(s, 0) for s in short]
    edge_x, edge_y, edge_colors = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            c = corr.iloc[i, j]
            if c > 0.3:
                edge_x += [pos_x[i], pos_x[j], None]
                edge_y += [pos_y[i], pos_y[j], None]
    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.8, color="#30363d"),
        hoverinfo="none",
    ))
    node_colors = ["#ff5c6b" if r > 0.5 else "#ffb830" if r > 0.15 else "#3ddc84"
                   for r in resistance_rates_list]
    fig_net.add_trace(go.Scatter(
        x=pos_x, y=pos_y, mode="markers+text",
        marker=dict(size=28, color=node_colors, line=dict(width=1.5, color="#0d1117")),
        text=short,
        textposition="middle center",
        textfont=dict(size=9, color="#0d1117", family="DM Mono"),
        hovertemplate=[f"{s}<br>Resistance: {r:.1%}<extra></extra>"
                       for s, r in zip(short, resistance_rates_list)],
    ))
    fig_net.update_layout(
        height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(x=0, y=-1.25, text="🔴 >50% resistant   🟡 15–50%   🟢 <15%",
                 showarrow=False, font=dict(color="#8b949e", size=11)),
        ],
    )
    st.plotly_chart(fig_net, use_container_width=True)


def render_shap_waterfall(pred: dict):
    """Approximate SHAP-style waterfall for the current prediction."""
    if best_model.feature_importances is None:
        return
    fi     = best_model.feature_importances.head(8)
    base   = dataset.mdr_rate
    labels = []
    values = []
    patient = pred["patient"]
    sp_col  = "sp_" + patient["species"].replace(" ","_").replace(".","_")
    for feat, imp in fi.items():
        val = 0
        if feat == "age":              val = (patient["age"] - 45) / 100
        elif feat == "Infection_Freq": val = (patient["inf_freq"] - 1) / 10
        elif feat == "Diabetes_enc":   val = 0.05 if patient["diabetes"] else -0.02
        elif feat == "Hypertension_enc": val = 0.04 if patient["hypertension"] else -0.01
        elif feat == "Hospital_enc":   val = 0.04 if patient["prev_hosp"] else -0.01
        elif feat == sp_col:           val = 0.06
        else:                          val = 0.01 * np.random.randn()
        contrib = imp * val * 3
        clean   = feat.replace("sp_","").replace("_"," ").replace("enc","").title().strip()
        labels.append(clean)
        values.append(float(contrib))
    colors = ["#ff5c6b" if v > 0 else "#3ddc84" for v in values]
    fig_shap = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{'+' if v>0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#8b949e", size=10),
    ))
    fig_shap.add_hline(y=0, line_color="#30363d")
    fig_shap.update_layout(
        title=dict(text="Feature Contributions to Prediction (SHAP-style)",
                   font=dict(size=13, color="#e6edf3")),
        height=300, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        margin=dict(l=10, r=10, t=50, b=60),
        xaxis=dict(tickfont=dict(color="#e6edf3", size=10), tickangle=20),
        yaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e", size=10)),
        font_color="#e6edf3",
    )
    st.plotly_chart(fig_shap, use_container_width=True)


def parse_ai_sections(text: str) -> dict:
    """Split AI text into labelled sections."""
    import re
    section_keys = ["Clinical Explanation", "Resistance Mechanism",
                    "Risk Interpretation", "Treatment Strategy", "Summary"]
    sections = {}
    for key in section_keys:
        pattern = rf"(?:{key}|{key.upper()})[:\-\n]+(.*?)(?=(?:{'|'.join(section_keys)})[:\-\n]|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            sections[key] = m.group(1).strip()
    if not sections:
        sections["Clinical Explanation"] = text.strip()
    return sections


def build_text_report(pred: dict, ai_text: str) -> str:
    """Generate a plain-text downloadable report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "═" * 60,
        "  AMR-INSIGHT  |  CLINICAL DECISION REPORT",
        f"  Generated: {ts}",
        "═" * 60,
        "",
        "PATIENT SUMMARY",
        "-" * 40,
        f"  Age            : {pred['patient']['age']}",
        f"  Gender         : {pred['patient']['gender']}",
        f"  Species        : {pred['patient']['species']}",
        f"  Diabetes       : {'Yes' if pred['patient']['diabetes'] else 'No'}",
        f"  Hypertension   : {'Yes' if pred['patient']['hypertension'] else 'No'}",
        f"  Prior Hospital : {'Yes' if pred['patient']['prev_hosp'] else 'No'}",
        f"  Infection Freq : {pred['patient']['inf_freq']}",
        "",
        "PREDICTION RESULT",
        "-" * 40,
        f"  MDR Probability : {pred['prob']:.1%}",
        f"  Risk Category   : {pred['risk']}",
        f"  Confidence      : {pred['confidence']:.1%}",
        "",
        "AI CLINICAL EXPLANATION",
        "-" * 40,
        ai_text or "(No AI explanation generated)",
        "",
        "═" * 60,
        "  DISCLAIMER: For research/support use only. Always verify",
        "  with laboratory susceptibility testing before treatment.",
        "═" * 60,
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  MAIN TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Prediction", "📊 Analytics", "🤖 AI Explanation", "💬 AI Chat"])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    pred = st.session_state.prediction

    if pred is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#8b949e;">
          <div style="font-size:3rem;margin-bottom:16px;">🩺</div>
          <div style="font-size:1.1rem;font-weight:600;">Enter patient details in the sidebar</div>
          <div style="font-size:0.9rem;margin-top:8px;">Click <strong style="color:#00e5c3;">Run Prediction</strong> to start</div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Prediction result row ──────────────────────────────
        r1, r2, r3, r4 = st.columns([2, 1.5, 1.5, 1.5])

        with r1:
            st.markdown(f"""
            <div class="section-card">
              <div class="section-title">MDR Probability</div>
              <div style="font-size:4rem;font-weight:700;font-family:'DM Mono',monospace;
                          color:{'#ff5c6b' if pred['risk']=='HIGH' else '#ffb830' if pred['risk']=='MODERATE' else '#3ddc84'};">
                {pred['prob']:.1%}
              </div>
              <div style="margin-top:10px;">
                <span class="{pred['badge']}">{pred['risk']} RISK</span>
              </div>
              <div style="margin-top:12px;font-size:0.82rem;color:#8b949e;">
                {pred['patient']['species']}  ·  Age {pred['patient']['age']}  ·  {pred['patient']['gender']}
              </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            conf_pct = pred['confidence'] * 100
            color    = "#00e5c3"
            st.markdown(f"""
            <div class="section-card" style="height:100%;">
              <div class="section-title">Confidence</div>
              <div style="font-size:2.4rem;font-weight:700;font-family:'DM Mono',monospace;color:{color};">
                {conf_pct:.1f}%
              </div>
              <div style="font-size:0.8rem;color:#8b949e;margin-top:6px;">
                Model certainty score
              </div>
            </div>""", unsafe_allow_html=True)

        with r3:
            comorbidities = sum([pred['patient']['diabetes'],
                                  pred['patient']['hypertension'],
                                  pred['patient']['prev_hosp']])
            st.markdown(f"""
            <div class="section-card" style="height:100%;">
              <div class="section-title">Comorbidities</div>
              <div style="font-size:2.4rem;font-weight:700;font-family:'DM Mono',monospace;color:#ffb830;">
                {comorbidities}/3
              </div>
              <div style="font-size:0.8rem;color:#8b949e;margin-top:6px;">
                DM · HTN · Hospitalisation
              </div>
            </div>""", unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="section-card" style="height:100%;">
              <div class="section-title">Infection Freq</div>
              <div style="font-size:2.4rem;font-weight:700;font-family:'DM Mono',monospace;color:#00e5c3;">
                {pred['patient']['inf_freq']}×
              </div>
              <div style="font-size:0.8rem;color:#8b949e;margin-top:6px;">Prior episodes</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge chart ──────────────────────────────────────────
        left_col, right_col = st.columns([1, 1])

        with left_col:
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = pred["prob"] * 100,
                number = {"suffix": "%", "font": {"size": 40, "color": "#e6edf3", "family": "DM Mono"}},
                title  = {"text": "MDR Risk Score", "font": {"size": 14, "color": "#8b949e"}},
                gauge  = {
                    "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d",
                               "tickfont": {"color": "#8b949e"}},
                    "bar":   {"color": "#ff5c6b" if pred["risk"]=="HIGH" else "#ffb830" if pred["risk"]=="MODERATE" else "#3ddc84",
                               "thickness": 0.35},
                    "bgcolor": "#1c2230",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  50], "color": "#1a3a2a"},
                        {"range": [50, 75], "color": "#3a2e10"},
                        {"range": [75,100], "color": "#3a1520"},
                    ],
                    "threshold": {"line": {"color": "#e6edf3", "width": 2},
                                  "thickness": 0.8, "value": pred["prob"]*100},
                },
            ))
            fig_gauge.update_layout(
                height=260, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=20, r=20, t=30, b=20),
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with right_col:
            # Feature importance bar (patient context)
            if best_model.feature_importances is not None:
                fi = best_model.feature_importances.head(8).reset_index()
                fi.columns = ["Feature", "Importance"]
                # Clean feature names
                fi["Feature"] = fi["Feature"].str.replace("sp_","").str.replace("_"," ").str.replace("enc","").str.title().str.strip()

                fig_fi = go.Figure(go.Bar(
                    x=fi["Importance"], y=fi["Feature"],
                    orientation="h",
                    marker_color=["#00e5c3" if i == 0 else "#00b59a" if i < 3 else "#2c5364" for i in range(len(fi))],
                    text=[f"{v:.2%}" for v in fi["Importance"]],
                    textposition="outside",
                    textfont=dict(color="#8b949e", size=11),
                ))
                fig_fi.update_layout(
                    title=dict(text="Top Feature Importances", font=dict(size=13, color="#8b949e")),
                    height=260, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    margin=dict(l=10, r=60, t=40, b=10),
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(tickfont=dict(color="#e6edf3", size=11)),
                    font_color="#e6edf3",
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Decision Support: Antibiotics ────────────────────────
        st.markdown('<div class="section-title">💊  Antibiotic Decision Support</div>', unsafe_allow_html=True)

        avoid, suggest, caution = get_antibiotic_recommendations(pred["risk"], pred["patient"]["species"])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⛔ Avoid (High Resistance)</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="pill-avoid">{ab}</span>' for ab in avoid)
            st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⚠️ Caution (Moderate)</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="pill-caution">{ab}</span>' for ab in caution)
            st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">✅ Consider (Lower Resistance)</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="pill-suggest">{ab}</span>' for ab in suggest)
            st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Download report ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        report_text = build_text_report(pred, st.session_state.ai_text)
        st.download_button(
            label     = "⬇️  Download Clinical Report (.txt)",
            data      = report_text,
            file_name = f"AMR_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime      = "text/plain",
        )


# ══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Resistance Pattern Analytics</div>', unsafe_allow_html=True)

    from models.resistance_dataset import ANTIBIOTIC_COLS

    # ── Row 1: Resistance rates + MDR distribution ─────────────
    a1, a2 = st.columns(2)

    with a1:
        rates  = dataset.resistance_rates
        labels = list(rates.keys())
        vals   = [rates[l] * 100 for l in labels]
        colors = ["#ff5c6b" if v > 50 else "#ffb830" if v > 15 else "#3ddc84" for v in vals]

        fig_ab = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color="#8b949e", size=10),
        ))
        fig_ab.update_layout(
            title=dict(text="Antibiotic Resistance Rates", font=dict(size=14, color="#e6edf3")),
            height=400, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            margin=dict(l=10, r=60, t=50, b=10),
            xaxis=dict(range=[0, 80], ticksuffix="%", gridcolor="#30363d",
                       tickfont=dict(color="#8b949e")),
            yaxis=dict(tickfont=dict(color="#e6edf3", size=11)),
            font_color="#e6edf3",
        )
        st.plotly_chart(fig_ab, use_container_width=True)

    with a2:
        # MDR distribution pie
        mdr_counts = dataset.y.value_counts().sort_index()
        fig_pie = go.Figure(go.Pie(
            labels=["Non-MDR", "MDR"],
            values=mdr_counts.values,
            hole=0.55,
            marker_colors=["#3ddc84", "#ff5c6b"],
            textinfo="label+percent",
            textfont=dict(size=13, color="#e6edf3"),
            hovertemplate="%{label}: %{value:,} isolates<extra></extra>",
        ))
        fig_pie.update_layout(
            title=dict(text="MDR Distribution", font=dict(size=14, color="#e6edf3")),
            height=400, paper_bgcolor="#161b22",
            margin=dict(l=10, r=10, t=50, b=10),
            font_color="#e6edf3",
            showlegend=True,
            legend=dict(font=dict(color="#8b949e")),
            annotations=[dict(text=f"{dataset.mdr_rate:.1%}<br>MDR",
                               font=dict(size=18, color="#ff5c6b"), showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Row 2: MDR by species + Feature importance ─────────────
    b1, b2 = st.columns(2)

    with b1:
        if dataset.mdr_by_species is not None:
            sp_df = dataset.mdr_by_species.head(10).reset_index()
            sp_df.columns = ["Species", "MDR_Rate", "Count"]
            sp_colors = ["#ff5c6b" if r > 0.80 else "#ffb830" if r > 0.65 else "#3ddc84"
                         for r in sp_df["MDR_Rate"]]

            fig_sp = go.Figure(go.Bar(
                x=sp_df["Species"], y=sp_df["MDR_Rate"] * 100,
                marker_color=sp_colors,
                text=[f"{v:.1f}%" for v in sp_df["MDR_Rate"] * 100],
                textposition="outside",
                textfont=dict(color="#8b949e", size=10),
                customdata=sp_df["Count"],
                hovertemplate="%{x}<br>MDR Rate: %{y:.1f}%<br>n=%{customdata:,}<extra></extra>",
            ))
            fig_sp.add_hline(y=75, line_dash="dot", line_color="#e6edf3",
                              annotation_text="75% threshold", annotation_font_color="#8b949e")
            fig_sp.update_layout(
                title=dict(text="MDR Rate by Bacterial Species", font=dict(size=14, color="#e6edf3")),
                height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=10, r=10, t=50, b=80),
                yaxis=dict(range=[0, 105], ticksuffix="%", gridcolor="#30363d",
                           tickfont=dict(color="#8b949e")),
                xaxis=dict(tickfont=dict(color="#e6edf3", size=10), tickangle=30),
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_sp, use_container_width=True)

    with b2:
        if best_model.feature_importances is not None:
            fi = best_model.feature_importances.head(12).reset_index()
            fi.columns = ["Feature", "Importance"]
            fi["Feature"] = fi["Feature"].str.replace("sp_","").str.replace("_"," ").str.title().str.strip()

            fig_fi2 = go.Figure(go.Bar(
                x=fi["Importance"] * 100, y=fi["Feature"],
                orientation="h",
                marker=dict(
                    color=fi["Importance"] * 100,
                    colorscale=[[0,"#2c5364"],[0.5,"#00b59a"],[1,"#00e5c3"]],
                    showscale=False,
                ),
                text=[f"{v:.1f}%" for v in fi["Importance"] * 100],
                textposition="outside",
                textfont=dict(color="#8b949e", size=10),
            ))
            fig_fi2.update_layout(
                title=dict(text=f"Feature Importance — {best_model.name}", font=dict(size=14, color="#e6edf3")),
                height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=10, r=60, t=50, b=10),
                xaxis=dict(ticksuffix="%", showgrid=False, showticklabels=False),
                yaxis=dict(tickfont=dict(color="#e6edf3", size=11)),
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_fi2, use_container_width=True)

    # ── Row 3: Antibiotic resistance network ────────────────────
    st.markdown('<div class="section-title">🕸️  Antibiotic Co-Resistance Network</div>', unsafe_allow_html=True)
    render_network_graph()


# ══════════════════════════════════════════════════════════════
#  TAB 3 — AI EXPLANATION
# ══════════════════════════════════════════════════════════════
with tab3:
    pred = st.session_state.prediction
    ai   = st.session_state.ai_text

    if pred is None:
        st.info("Run a prediction first to see the AI explanation.")
    else:
        st.markdown(f"""
        <div class="section-card">
          <div class="section-title">Patient Summary</div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <span style="color:#8b949e;">Species: <strong style="color:#e6edf3;">{pred['patient']['species']}</strong></span>
            <span style="color:#8b949e;">Age: <strong style="color:#e6edf3;">{pred['patient']['age']}</strong></span>
            <span style="color:#8b949e;">Gender: <strong style="color:#e6edf3;">{pred['patient']['gender']}</strong></span>
            <span style="color:#8b949e;">Risk: <strong style="color:{'#ff5c6b' if pred['risk']=='HIGH' else '#ffb830' if pred['risk']=='MODERATE' else '#3ddc84'};">{pred['risk']}</strong></span>
            <span style="color:#8b949e;">MDR Prob: <strong style="color:#e6edf3;">{pred['prob']:.1%}</strong></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if ai:
            sections = parse_ai_sections(ai)
            icons = {"Clinical Explanation":"🔬","Resistance Mechanism":"🧫",
                     "Risk Interpretation":"⚠️","Treatment Strategy":"💊","Summary":"📋"}
            for title, body in sections.items():
                icon = icons.get(title, "🤖")
                with st.expander(f"{icon}  {title}", expanded=True):
                    st.markdown(f'<div class="ai-box">{body}</div>', unsafe_allow_html=True)
        else:
            st.warning("AI explanation not yet generated. Click **Run Prediction** with an API key configured.")

        # Regenerate button
        if st.button("🔄  Regenerate Explanation"):
            with st.spinner("Re-generating…"):
                st.session_state.ai_text = generate_explanation(
                    prediction  = pred,
                    features    = pred["patient"],
                    top_factors = pred["top_factors"],
                    provider    = ai_provider,
                    api_key     = api_key,
                )
            st.rerun()

        # SHAP-style explanation
        if best_model.feature_importances is not None:
            st.markdown('<div class="section-title" style="margin-top:20px;">📐  SHAP-Style Feature Contribution</div>', unsafe_allow_html=True)
            render_shap_waterfall(pred)


# ══════════════════════════════════════════════════════════════
#  TAB 4 — AI CHAT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Interactive AI Assistant</div>', unsafe_allow_html=True)

    # Chat history display
    for msg in st.session_state.chat_history:
        role, text = msg["role"], msg["text"]
        if role == "user":
            st.markdown(f'<div><div class="chat-label">You</div><div class="chat-user">{text}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div><div class="chat-label">AMR-AI</div><div class="chat-ai">{text}</div></div>', unsafe_allow_html=True)

    # Quick prompts
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin:12px 0 6px;">Quick questions:</div>', unsafe_allow_html=True)
    qcols = st.columns(3)
    quick_qs = [
        "Why is this bacteria resistant?",
        "What is the best treatment?",
        "What does MDR mean?",
    ]
    for i, q in enumerate(quick_qs):
        with qcols[i]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state._chat_input = q

    # Chat input
    user_input = st.chat_input("Ask about resistance, treatment, mechanisms…")

    # Handle quick prompt injection
    if hasattr(st.session_state, "_chat_input"):
        user_input = st.session_state._chat_input
        del st.session_state._chat_input

    if user_input:
        context = ""
        if st.session_state.prediction:
            p = st.session_state.prediction
            context = (f"Patient: {p['patient']['species']}, age {p['patient']['age']}, "
                       f"{p['patient']['gender']}, MDR risk {p['risk']} ({p['prob']:.1%}). "
                       f"Prior AI explanation: {st.session_state.ai_text[:400]}")

        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.spinner("AMR-AI thinking…"):
            reply = chat_response(user_input, context, provider=ai_provider, api_key=api_key)
        st.session_state.chat_history.append({"role": "ai", "text": reply})
        st.rerun()

    if st.button("🗑️  Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# (helper functions moved above MAIN TABS section)
