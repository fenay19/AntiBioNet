"""
models/resistance_dataset.py
----------------------------
Data model: holds raw DataFrame, processed features, and split arrays.
No business logic — pure data container.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


# Antibiotic columns present in the dataset
ANTIBIOTIC_COLS: List[str] = [
    "AMX/AMP",           # Amoxicillin/Ampicillin  — Penicillin
    "AMC",               # Amoxicillin-Clavulanate — Beta-lactam/inhibitor
    "CZ",                # Cefazolin               — Cephalosporin 1G
    "FOX",               # Cefoxitin               — Cephalosporin 2G
    "CTX/CRO",           # Cefotaxime/Ceftriaxone  — Cephalosporin 3G
    "IPM",               # Imipenem                — Carbapenem
    "GEN",               # Gentamicin              — Aminoglycoside
    "AN",                # Amikacin                — Aminoglycoside
    "Acide nalidixique", # Nalidixic Acid           — Quinolone 1G
    "ofx",               # Ofloxacin               — Fluoroquinolone
    "CIP",               # Ciprofloxacin            — Fluoroquinolone
    "C",                 # Chloramphenicol          — Phenicol
    "Co-trimoxazole",    # Trimethoprim-Sulfa       — Folate inhibitor
    "Furanes",           # Nitrofurantoin           — Nitrofuran
    "colistine",         # Colistin                 — Polymyxin (last resort)
]

# Species name normalisation map
SPECIES_ALIASES: dict = {
    "E.coi":  "Escherichia coli",
    "E.cli":  "Escherichia coli",
    "E. coli": "Escherichia coli",
}

MDR_THRESHOLD: int = 3   # resistant to ≥ N classes = MDR
TOP_SPECIES: int = 8      # top N species get one-hot columns


@dataclass
class ResistanceDataset:
    """Immutable-ish data container populated by DataController."""

    raw_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    processed_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Feature / target arrays (set after engineer_features)
    X: Optional[pd.DataFrame] = field(default=None, repr=False)
    y: Optional[pd.Series]    = field(default=None, repr=False)
    X_train: Optional[pd.DataFrame] = field(default=None, repr=False)
    X_test:  Optional[pd.DataFrame] = field(default=None, repr=False)
    y_train: Optional[pd.Series]    = field(default=None, repr=False)
    y_test:  Optional[pd.Series]    = field(default=None, repr=False)

    feature_cols: List[str] = field(default_factory=list)
    top_species:  List[str] = field(default_factory=list)

    # Summary statistics (populated by DataController)
    resistance_rates: dict = field(default_factory=dict)
    mdr_by_species:   Optional[pd.DataFrame] = field(default=None, repr=False)
    mdr_rate: float = 0.0
    n_samples: int  = 0

    def __repr__(self):
        return (f"ResistanceDataset(n_samples={self.n_samples}, "
                f"mdr_rate={self.mdr_rate:.1%}, "
                f"features={len(self.feature_cols)})")
