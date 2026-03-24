# AMR-Insight — Antibiotic Resistance Clinical Dashboard

**Streamlit + Generative AI clinical decision support tool for predicting
Multi-Drug Resistance (MDR) in bacterial isolates.**..

---

## Project Structure

```
amr_streamlit/
│
├── app.py                     ← Main Streamlit application
├── llm.py                     ← Gen AI explanation layer
├── requirements.txt
│
├── models/                    ← Data & model containers
│   ├── resistance_dataset.py
│   └── resistance_model.py
│
├── controllers/               ← Business logic
│   ├── data_controller.py
│   ├── model_controller.py
│   └── report_controller.py
│
├── utils/
│   └── preprocessing.py       ← Pure helper functions + ANTIBIOTIC_META
│
├── data/                      ← Place your CSV here
│   └── Bacteria_dataset_Multiresictance.csv
│
└── outputs/                   ← Generated plots saved here
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
cp /path/to/Bacteria_dataset_Multiresictance.csv data/

# 3. Launch dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

---

## Features

### Tab 1 — Prediction
- Patient input form (sidebar): age 0–110, gender, comorbidities, species
- MDR probability gauge chart
- Risk category badge (Low / Moderate / High) with colour coding
- Feature importance bar chart
- Antibiotic decision support (avoid / caution / suggest pills)
- One-click downloadable clinical report (.txt)

### Tab 2 — Analytics
- Antibiotic resistance rates (colour-coded bar chart)
- MDR distribution pie/donut chart
- MDR rate by bacterial species
- Feature importance ranked chart
- Interactive co-resistance network graph

### Tab 3 — AI Explanation
- Structured clinical explanation (5 sections)
- Resistance mechanism description
- Treatment strategy recommendations
- SHAP-style feature contribution waterfall chart

### Tab 4 — AI Chat
- Conversational interface
- Quick-prompt buttons
- Full context awareness of current prediction

---

## Gen AI Configuration

Select provider in sidebar:

| Provider | Package required | Setting |
|----------|-----------------|---------|
| OpenAI GPT-4 | `pip install openai` | Enter API key |
| Google Gemini | `pip install google-generativeai` | Enter API key |
| Ollama (local) | Ollama running on port 11434 | No key needed |
| Rule-based (Offline) | None | No key needed |

The **Rule-based (Offline)** mode works without any API key and provides
clinically grounded explanations using built-in resistance knowledge.

---

## Age Constraint

Patient age input is strictly constrained to **0–110 years** via
`st.number_input(min_value=0, max_value=110)` — values outside this range
cannot be entered.

---

## Target Variable

**MDR = resistant to ≥ 3 antibiotic drug classes**  
Prevalence in dataset: ~75%

---

## Models Trained

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Gradient Boosting | ~82% | ~0.778 |
| Logistic Regression | ~81% | ~0.768 |
| Random Forest | ~80% | ~0.775 |

Best model auto-selected for predictions.

---

## Disclaimer

This dashboard is intended for **research and clinical support purposes only**.
All treatment decisions must be confirmed by laboratory susceptibility testing
and clinical judgement by a qualified healthcare professional.
