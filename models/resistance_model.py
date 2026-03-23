"""
models/resistance_model.py
--------------------------
Model wrapper: holds a trained sklearn estimator plus its evaluation metrics
and exposes a patient-level prediction helper.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ResistanceModel:
    """Wraps a single trained estimator with its metrics and feature list."""

    name:        str
    estimator:   Any                         # sklearn-compatible
    feature_cols: List[str] = field(default_factory=list)

    # Populated after evaluation
    y_pred:   Optional[np.ndarray] = field(default=None, repr=False)
    y_proba:  Optional[np.ndarray] = field(default=None, repr=False)
    accuracy: float = 0.0
    auc:      float = 0.0
    cv_auc:   float = 0.0
    cv_std:   float = 0.0
    report:   str   = ""

    # Feature importances (RF / GB only)
    feature_importances: Optional[pd.Series] = field(default=None, repr=False)

    def predict_patient(
        self,
        age: int,
        gender_M: bool,
        diabetes: bool,
        hypertension: bool,
        prev_hospital: bool,
        infection_freq: float,
        species: str = "Escherichia coli",
    ) -> float:
        """
        Return MDR probability for a single patient and print a risk summary.

        Parameters
        ----------
        age            : patient age in years
        gender_M       : True = Male
        diabetes       : comorbidity flag
        hypertension   : comorbidity flag
        prev_hospital  : prior hospitalisation flag
        infection_freq : number of previous infections
        species        : bacterial species name

        Returns
        -------
        probability (float 0–1)
        """
        row = {c: 0 for c in self.feature_cols}
        row["age"]              = age
        row["gender_enc"]       = int(gender_M)
        row["Diabetes_enc"]     = int(diabetes)
        row["Hypertension_enc"] = int(hypertension)
        row["Hospital_enc"]     = int(prev_hospital)
        row["Infection_Freq"]   = float(infection_freq)

        sp_col = "sp_" + species.replace(" ", "_").replace(".", "_")
        if sp_col in row:
            row[sp_col] = 1

        X_new = pd.DataFrame([row])[self.feature_cols]
        prob  = self.estimator.predict_proba(X_new)[0, 1]

        risk_label = (
            "HIGH MDR RISK  ⚠"  if prob >= 0.75 else
            "MODERATE RISK  ~"  if prob >= 0.50 else
            "LOW RISK       ✓"
        )
        gender_str = "M" if gender_M else "F"
        print(
            f"\n  Patient: age={age}, {gender_str}, DM={int(diabetes)}, "
            f"HTN={int(hypertension)}, hosp={int(prev_hospital)}, "
            f"freq={infection_freq}, species={species}"
        )
        print(f"  MDR Probability : {prob:.1%}  →  {risk_label}")
        return prob

    def __repr__(self):
        return (f"ResistanceModel(name='{self.name}', "
                f"accuracy={self.accuracy:.4f}, auc={self.auc:.4f})")
