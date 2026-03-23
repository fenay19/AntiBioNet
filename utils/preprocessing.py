"""
utils/preprocessing.py
-----------------------
Pure helper functions used by DataController.
Also exposes ANTIBIOTIC_META for the dashboard.
"""

import re
from typing import Optional
import numpy as np
import pandas as pd

# ── Antibiotic metadata for UI ────────────────────────────────────────────────
ANTIBIOTIC_META = {
    "AMX/AMP":           {"full": "Amoxicillin / Ampicillin",    "class": "Penicillin"},
    "AMC":               {"full": "Amoxicillin-Clavulanate",     "class": "Beta-lactam/inhibitor"},
    "CZ":                {"full": "Cefazolin",                   "class": "Cephalosporin 1G"},
    "FOX":               {"full": "Cefoxitin",                   "class": "Cephalosporin 2G"},
    "CTX/CRO":           {"full": "Cefotaxime / Ceftriaxone",    "class": "Cephalosporin 3G"},
    "IPM":               {"full": "Imipenem",                    "class": "Carbapenem"},
    "GEN":               {"full": "Gentamicin",                  "class": "Aminoglycoside"},
    "AN":                {"full": "Amikacin",                    "class": "Aminoglycoside"},
    "Acide nalidixique": {"full": "Nalidixic Acid",              "class": "Quinolone 1G"},
    "ofx":               {"full": "Ofloxacin",                   "class": "Fluoroquinolone"},
    "CIP":               {"full": "Ciprofloxacin",               "class": "Fluoroquinolone"},
    "C":                 {"full": "Chloramphenicol",             "class": "Phenicol"},
    "Co-trimoxazole":    {"full": "Trimethoprim-Sulfamethoxazole","class": "Folate inhibitor"},
    "Furanes":           {"full": "Nitrofurantoin",              "class": "Nitrofuran"},
    "colistine":         {"full": "Colistin",                    "class": "Polymyxin"},
}


def normalize_resistance(value) -> Optional[str]:
    """
    Standardise a resistance label to one of: 'R', 'S', 'I', or NaN.

    Handles mixed case, whitespace, abbreviations, and noise values
    ('missing', '?', blank strings, etc.).

    Examples
    --------
    >>> normalize_resistance('r')   → 'R'
    >>> normalize_resistance('Intermediate') → 'I'
    >>> normalize_resistance('?')   → NaN
    """
    if pd.isna(value):
        return np.nan
    val = str(value).strip().upper()
    if val == "R":
        return "R"
    if val == "S":
        return "S"
    if val in {"I", "INTERMEDIATE"}:
        return "I"
    return np.nan


def normalize_boolean(value) -> int:
    """
    Convert a yes/no/true/false column to 0 or 1.

    Treats 'missing', '?', NaN as 0 (conservative default).
    """
    if pd.isna(value):
        return 0
    return 1 if str(value).strip().upper() in {"YES", "TRUE", "1"} else 0


def extract_age(age_gender_str) -> Optional[float]:
    """
    Extract numeric age from strings like '37/F', '29/M', '77/F'.

    Returns NaN if no numeric prefix is found.
    """
    if pd.isna(age_gender_str):
        return np.nan
    match = re.match(r"(\d+)", str(age_gender_str).strip())
    return float(match.group(1)) if match else np.nan


def extract_gender(age_gender_str) -> int:
    """
    Extract gender from strings like '37/F', '29/M'.

    Returns 1 for Male ('M'), 0 otherwise (Female, unknown).
    """
    if pd.isna(age_gender_str):
        return 0
    match = re.search(r"/(\w)", str(age_gender_str).strip())
    if match:
        return 1 if match.group(1).upper() == "M" else 0
    return 0


def clean_species_name(souche_str) -> str:
    """
    Extract the species name from 'S290 Escherichia coli' → 'Escherichia coli'.

    Strips the leading isolate ID code if present.
    """
    if pd.isna(souche_str):
        return "Unknown"
    # Remove leading alphanumeric ID like 'S290 '
    cleaned = re.sub(r"^\w+\d+\s+", "", str(souche_str).strip())
    return cleaned.strip() if cleaned else "Unknown"
