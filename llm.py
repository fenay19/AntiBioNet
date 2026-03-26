"""
llm.py
======
Gen AI integration layer for AMR-Insight.

Supports:
  • OpenAI  (GPT-4o / GPT-3.5-turbo)
  • Google Gemini  (gemini-1.5-pro)
  • Ollama  (local — any model, default: llama3)
  • Rule-based fallback  (offline, no API key needed)

Public API:
  generate_explanation(prediction, features, top_factors, provider, api_key) → str
  chat_response(user_message, context, provider, api_key) → str
"""

from __future__ import annotations
import json
import textwrap
from typing import Any


# ════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES
# ════════════════════════════════════════════════════════════════

_EXPLANATION_SYSTEM = """
You are an expert clinical microbiologist and infectious disease specialist.
Your role is to provide clear, evidence-based, and clinically meaningful
explanations of antibiotic resistance predictions.

Always structure your response with these EXACT section headers:
  Clinical Explanation:
  Resistance Mechanism:
  Risk Interpretation:
  Treatment Strategy:
  Summary:

Be concise, precise, and use clinical language appropriate for a physician.
Do NOT use markdown bold (**) or bullet stars (*). Use plain text.
""".strip()

_EXPLANATION_USER = """
A machine learning model predicted the following for a patient:

PATIENT DATA:
  Age: {age} years
  Gender: {gender}
  Bacterial Species: {species}
  Diabetes: {diabetes}
  Hypertension: {hypertension}
  Prior Hospitalisation: {prev_hosp}
  Infection Frequency: {inf_freq} prior episodes

PREDICTION:
  MDR Probability: {prob:.1%}
  Risk Category: {risk}
  Model Confidence: {confidence:.1%}

TOP PREDICTIVE FACTORS (feature importances):
{top_factors}

Please provide a complete clinical explanation covering:
1. Why this patient is predicted to have {risk} MDR risk
2. The likely resistance mechanisms of {species}
3. How to interpret this risk level clinically
4. Evidence-based treatment strategy recommendations
5. A brief summary for clinical handover
""".strip()

_CHAT_SYSTEM = """
You are AMR-AI, a clinical decision support assistant specialising in
antibiotic resistance. Answer questions concisely, accurately, and in
plain language. Do NOT use markdown formatting. Limit to 3–5 sentences
unless the question requires more detail.
""".strip()


# ════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTIONS
# ════════════════════════════════════════════════════════════════

def generate_explanation(
    prediction:  dict,
    features:    dict,
    top_factors: dict,
    provider:    str = "Rule-based (Offline)",
    api_key:     str = "",
) -> str:
    """
    Generate a clinical explanation for an MDR prediction.

    Parameters
    ----------
    prediction   : dict from run_prediction() — keys: prob, risk, confidence
    features     : patient feature dict
    top_factors  : {feature_name: importance_value}
    provider     : LLM provider string from sidebar selectbox
    api_key      : provider API key (empty for offline/Ollama)

    Returns
    -------
    Multi-section explanation string.
    """
    prompt_user = _EXPLANATION_USER.format(
        age         = features.get("age", "N/A"),
        gender      = features.get("gender", "N/A"),
        species     = features.get("species", "N/A"),
        diabetes    = "Yes" if features.get("diabetes") else "No",
        hypertension= "Yes" if features.get("hypertension") else "No",
        prev_hosp   = "Yes" if features.get("prev_hosp") else "No",
        inf_freq    = features.get("inf_freq", 0),
        prob        = prediction.get("prob", 0),
        risk        = prediction.get("risk", "UNKNOWN"),
        confidence  = prediction.get("confidence", 0),
        top_factors = "\n".join(f"  {k}: {v:.3f}" for k, v in top_factors.items()),
    )

    if "OpenAI" in provider and api_key:
        return _call_openai(prompt_user, api_key)
    elif "Gemini" in provider and api_key:
        return _call_gemini(prompt_user, api_key)
    elif "Ollama" in provider:
        return _call_ollama(prompt_user)
    else:
        return _rule_based_explanation(prediction, features)


def chat_response(
    user_message: str,
    context:      str = "",
    provider:     str = "Rule-based (Offline)",
    api_key:      str = "",
) -> str:
    """
    Generate a chat reply for the AI assistant tab.

    Parameters
    ----------
    user_message : user's question
    context      : patient/prediction context string
    provider     : LLM provider
    api_key      : API key

    Returns
    -------
    Assistant reply string.
    """
    system = _CHAT_SYSTEM
    if context:
        system += f"\n\nCurrent patient context:\n{context}"

    if "OpenAI" in provider and api_key:
        return _call_openai(user_message, api_key, system=system, max_tokens=400)
    elif "Gemini" in provider and api_key:
        return _call_gemini(user_message, api_key, system=system, max_tokens=400)
    elif "Ollama" in provider:
        return _call_ollama(user_message, system=system)
    else:
        return _rule_based_chat(user_message)


# ════════════════════════════════════════════════════════════════
#  PROVIDER IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════

def _call_openai(
    user_prompt: str,
    api_key:     str,
    system:      str = _EXPLANATION_SYSTEM,
    max_tokens:  int = 800,
) -> str:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model    = "gpt-4o",
            messages = [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens  = max_tokens,
            temperature = 0.3,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        return (
            "[OpenAI Error] The 'openai' package is not installed.\n"
            "Install it with:  pip install openai\n\n"
            + _rule_based_explanation_from_text(user_prompt)
        )
    except Exception as e:
        return f"[OpenAI Error] {e}\n\n" + _rule_based_explanation_from_text(user_prompt)


def _call_gemini(
    user_prompt: str,
    api_key:     str,
    system:      str = _EXPLANATION_SYSTEM,
    max_tokens:  int = 800,
) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name     = "gemini-2.0-flash",
            system_instruction = system,
        )
        resp = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens = max_tokens,
                temperature       = 0.3,
            ),
        )
        return resp.text.strip()
    except ImportError:
        return (
            "[Gemini Error] The 'google-generativeai' package is not installed.\n"
            "Install it with:  pip install google-generativeai\n\n"
            + _rule_based_explanation_from_text(user_prompt)
        )
    except Exception as e:
        return f"[Gemini Error] {e}\n\n" + _rule_based_explanation_from_text(user_prompt)


def _call_ollama(
    user_prompt: str,
    system:      str = _EXPLANATION_SYSTEM,
    model:       str = "llama3",
    host:        str = "http://localhost:11434",
) -> str:
    try:
        import requests
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
        }
        r = requests.post(f"{host}/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama Error] {e}\n\n" + _rule_based_explanation_from_text(user_prompt)


# ════════════════════════════════════════════════════════════════
#  RULE-BASED FALLBACK
# ════════════════════════════════════════════════════════════════

_SPECIES_MECHANISMS = {
    "Escherichia coli": (
        "E. coli commonly acquires resistance through extended-spectrum beta-lactamases (ESBLs), "
        "plasmid-mediated AmpC beta-lactamases, and efflux pump overexpression. "
        "Fluoroquinolone resistance is mediated by gyrA/parC mutations. "
        "Carbapenem resistance, when present, usually involves OXA-type or NDM-type carbapenemases."
    ),
    "Klebsiella pneumoniae": (
        "K. pneumoniae is notorious for ESBL production (CTX-M variants), "
        "KPC-type carbapenemases, and OmpK35/OmpK36 porin loss. "
        "Hypervirulent strains may co-carry resistance plasmids, making treatment extremely challenging. "
        "Colistin resistance via mgrB mutations or mcr genes is increasingly reported."
    ),
    "Pseudomonas aeruginosa": (
        "P. aeruginosa employs multiple simultaneous resistance mechanisms: MexAB-OprM efflux pump "
        "overexpression, chromosomal AmpC beta-lactamase induction, OprD porin loss (carbapenem resistance), "
        "and aminoglycoside-modifying enzymes. Its intrinsic resistance and biofilm formation "
        "make it a formidable nosocomial pathogen."
    ),
    "Proteus mirabilis": (
        "P. mirabilis typically produces chromosomal inducible cephalosporinase and can acquire "
        "plasmid-mediated ESBLs. Fluoroquinolone resistance via topoisomerase mutations is frequent "
        "in hospital-acquired strains. Fosfomycin and nitrofurantoin show activity in uncomplicated UTI."
    ),
    "Serratia marcescens": (
        "S. marcescens carries inducible chromosomal AmpC that may be stably derepressed under "
        "cephalosporin selective pressure. It frequently acquires carbapenemase genes and has "
        "innate resistance to polymyxins, severely limiting treatment options for MDR strains."
    ),
}

_DEFAULT_MECHANISM = (
    "This organism may acquire resistance through plasmid-mediated enzymes (beta-lactamases, "
    "carbapenemases), efflux pump upregulation, target site mutations, and outer membrane "
    "protein alterations. Horizontal gene transfer enables rapid spread of resistance determinants "
    "across clinical isolates."
)

_RISK_INTERPRETATION = {
    "HIGH":     "This isolate carries a high probability of Multi-Drug Resistance. "
                "Empirical broad-spectrum therapy is likely to fail. Urgent susceptibility "
                "testing (AST) is mandatory. Consider infectious disease specialist consultation "
                "and institution of isolation precautions to prevent nosocomial spread.",
    "MODERATE": "This isolate shows moderate MDR risk. Empirical therapy should be guided by "
                "local antibiogram data while awaiting AST results. Avoid third-generation "
                "cephalosporins and penicillins without beta-lactamase inhibitor coverage.",
    "LOW":      "MDR risk is low for this isolate. Standard empirical regimens guided by "
                "clinical presentation are reasonable. Confirm with AST results before "
                "de-escalating or modifying therapy.",
}

_TREATMENT = {
    "HIGH": (
        "Treatment Strategy: For confirmed or suspected MDR isolate:\n"
        "  1. Await susceptibility testing — do not initiate broad empirical therapy blindly.\n"
        "  2. Consider colistin or polymyxin B combined with a carbapenem if carbapenemase suspected.\n"
        "  3. Ceftazidime-avibactam or ceftolozane-tazobactam may be active against MDR Pseudomonas.\n"
        "  4. Fosfomycin + aminoglycoside combination for urinary tract MDR infections.\n"
        "  5. Involve pharmacy for PK/PD-optimised dosing (extended infusions).\n"
        "  6. De-escalate as soon as susceptibility data permits."
    ),
    "MODERATE": (
        "Treatment Strategy:\n"
        "  1. Aminoglycosides (amikacin) or fluoroquinolones (if susceptible) are viable options.\n"
        "  2. Piperacillin-tazobactam covers many ESBL-negative strains.\n"
        "  3. Obtain cultures and await AST before 48–72 hours.\n"
        "  4. Review and narrow therapy on day 3 based on susceptibility results.\n"
        "  5. Avoid cephalosporins as monotherapy if ESBL suspected."
    ),
    "LOW": (
        "Treatment Strategy:\n"
        "  1. Standard regimens (fluoroquinolones, cephalosporins) are likely effective.\n"
        "  2. Obtain susceptibility testing to confirm and enable de-escalation.\n"
        "  3. Duration should follow evidence-based guidelines for the infection site.\n"
        "  4. Monitor clinical response at 48–72 hours and adjust accordingly."
    ),
}


def _rule_based_explanation(prediction: dict, features: dict) -> str:
    prob    = prediction.get("prob", 0)
    risk    = prediction.get("risk", "LOW")
    species = features.get("species", "Unknown")
    age     = features.get("age", 0)
    dm      = features.get("diabetes", False)
    htn     = features.get("hypertension", False)
    hosp    = features.get("prev_hosp", False)

    mech = _SPECIES_MECHANISMS.get(species, _DEFAULT_MECHANISM)

    risk_factors = []
    if age > 60:      risk_factors.append(f"advanced age ({age} years)")
    if dm:            risk_factors.append("diabetes mellitus")
    if htn:           risk_factors.append("hypertension")
    if hosp:          risk_factors.append("prior hospitalisation")

    rf_text = (
        "Key risk factors identified: " + ", ".join(risk_factors) + "."
        if risk_factors else
        "No major comorbidity risk factors identified in this patient."
    )

    return f"""Clinical Explanation:
The machine learning model predicts a {prob:.1%} probability of Multi-Drug Resistance (MDR)
for this {species} isolate, placing this patient in the {risk} risk category. {rf_text}
Age is the strongest individual predictor in this model, as older patients have higher
cumulative antibiotic exposure and more healthcare contacts, increasing selection pressure.

Resistance Mechanism:
{mech}

Risk Interpretation:
{_RISK_INTERPRETATION[risk]}

{_TREATMENT[risk]}

Summary:
Patient presents with {risk} MDR risk ({prob:.1%} probability) due to {species} infection.
{'Comorbidities (' + ", ".join(risk_factors) + ') further elevate concern.' if risk_factors else 'No major comorbidities identified.'}
Immediate actions: {
    'urgent AST, isolate patient, consult ID specialist' if risk == 'HIGH'
    else 'obtain cultures, await AST, monitor closely' if risk == 'MODERATE'
    else 'standard therapy, confirm with AST, de-escalate promptly'
}.
"""


def _rule_based_explanation_from_text(prompt_text: str) -> str:
    """Fallback when called without structured dict (error recovery)."""
    return (
        "Clinical Explanation:\n"
        "Unable to connect to AI provider. The rule-based engine is active as fallback.\n"
        "Please verify your API key and network connectivity to use LLM-powered explanations.\n\n"
        "Resistance Mechanism:\nSee offline reference for species-specific mechanisms.\n\n"
        "Risk Interpretation:\nRefer to local antibiogram and clinical judgment.\n\n"
        "Treatment Strategy:\nObtain susceptibility testing and consult infectious disease.\n\n"
        "Summary:\nAI explanation unavailable — using offline fallback mode."
    )


_CHAT_ANSWERS = {
    "resistant": (
        "Antibiotic resistance in Enterobacteriaceae arises primarily from three mechanisms: "
        "(1) enzymatic inactivation by beta-lactamases (ESBLs, carbapenemases), "
        "(2) efflux pump overexpression actively pumping antibiotics out of the cell, and "
        "(3) target site mutations reducing antibiotic binding affinity. "
        "These can be chromosomally encoded or spread via plasmids between strains."
    ),
    "treatment": (
        "Treatment selection depends on susceptibility testing results. "
        "For MDR organisms, options include carbapenems (if susceptible), "
        "ceftazidime-avibactam, colistin combinations, or fosfomycin. "
        "Always obtain cultures first, use the narrowest effective agent, "
        "and de-escalate therapy as soon as susceptibility data is available."
    ),
    "mdr": (
        "Multi-Drug Resistance (MDR) means an organism is resistant to at least one agent in "
        "three or more antibiotic categories. Extensively Drug Resistant (XDR) implies resistance "
        "to all but two categories, and Pan-Drug Resistant (PDR) means no standard agents work. "
        "MDR is driven by antibiotic overuse, horizontal gene transfer, and poor infection control."
    ),
    "risk": (
        "High MDR risk means the ML model predicts >75% probability that this isolate will be "
        "resistant to 3+ antibiotic drug classes. This warrants urgent susceptibility testing, "
        "infectious disease consultation, and infection control measures to prevent spread."
    ),
    "shap": (
        "Feature importance shows which patient and bacterial characteristics most influenced "
        "the MDR prediction. Age is typically the strongest predictor, followed by bacterial "
        "species and prior hospitalisation, reflecting cumulative antibiotic exposure and "
        "healthcare-associated selection pressure."
    ),
}


def _rule_based_chat(message: str) -> str:
    msg_lower = message.lower()
    for key, answer in _CHAT_ANSWERS.items():
        if key in msg_lower:
            return answer
    return (
        "I am AMR-AI, operating in offline rule-based mode. "
        "For LLM-powered responses, please configure an API key for OpenAI, Gemini, or Ollama. "
        "I can answer questions about antibiotic resistance mechanisms, MDR definitions, "
        "treatment strategies, and risk interpretation using my built-in knowledge base."
    )
