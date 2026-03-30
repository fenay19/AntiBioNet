"""
controllers/data_controller.py
-------------------------------
Orchestrates data loading, cleaning, feature engineering, and train/test split.
Populates a ResistanceDataset model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from models.resistance_dataset import (
    ResistanceDataset,
    ANTIBIOTIC_COLS,
    MDR_THRESHOLD,
    SPECIES_ALIASES,
    TOP_SPECIES,
)
from utils.preprocessing import (
    normalize_resistance,
    normalize_boolean,
    extract_age,
    extract_gender,
    clean_species_name,
)


class DataController:
    """
    Responsible for:
      1. Loading the raw CSV into ResistanceDataset.raw_df
      2. Cleaning resistance labels (R/S/I normalisation)
      3. Engineering features (age, gender, comorbidities, species one-hot)
      4. Computing MDR target
      5. Performing train/test split
    """

    def __init__(self, filepath: str, test_size: float = 0.20, random_state: int = 42):
        self.filepath     = filepath
        self.test_size    = test_size
        self.random_state = random_state
        self.dataset      = ResistanceDataset()

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "DataController":
        """Load raw CSV into dataset.raw_df."""
        print(f"\n[DataController] Loading: {self.filepath}")
        df = pd.read_csv(self.filepath)
        self.dataset.raw_df  = df
        self.dataset.n_samples = len(df)
        print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
        return self

    def preprocess(self) -> "DataController":
        """Normalise resistance labels and compute MDR target."""
        print("[DataController] Preprocessing resistance labels …")
        df = self.dataset.raw_df.copy()

        # Normalise R/S/I
        for col in ANTIBIOTIC_COLS:
            df[col] = df[col].apply(normalize_resistance)

        # Binary resistance flags (R=1, other=0)
        for col in ANTIBIOTIC_COLS:
            df[f"{col}_R"] = (df[col] == "R").astype(int)

        # MDR target
        r_cols = [f"{c}_R" for c in ANTIBIOTIC_COLS]
        df["n_resistant"] = df[r_cols].sum(axis=1)
        df["MDR"] = (df["n_resistant"] >= MDR_THRESHOLD).astype(int)

        # Resistance rates per antibiotic
        self.dataset.resistance_rates = {
            col: float((df[col] == "R").sum() / df[col].notna().sum())
            for col in ANTIBIOTIC_COLS
        }

        self.dataset.mdr_rate     = float(df["MDR"].mean())
        self.dataset.processed_df = df
        print(f"  MDR prevalence: {df['MDR'].mean():.1%} ({df['MDR'].sum():,} isolates)")
        return self

    def engineer_features(self) -> "DataController":
        """Build feature matrix and perform train/test split."""
        print("[DataController] Engineering features …")
        df = self.dataset.processed_df.copy()

        # Demographics
        df["age"]        = df["age/gender"].apply(extract_age)
        df["gender_enc"] = df["age/gender"].apply(extract_gender)

        # Comorbidities
        df["Diabetes_enc"]     = df["Diabetes"].apply(normalize_boolean)
        df["Hypertension_enc"] = df["Hypertension"].apply(normalize_boolean)
        df["Hospital_enc"]     = df["Hospital_before"].apply(normalize_boolean)
        df["Infection_Freq"]   = pd.to_numeric(df["Infection_Freq"], errors="coerce").fillna(0)

        # Species one-hot
        df["bacteria"] = df["Souches"].apply(clean_species_name)
        df["bacteria"] = df["bacteria"].replace(SPECIES_ALIASES)
        top_sp = df["bacteria"].value_counts().head(TOP_SPECIES).index.tolist()
        self.dataset.top_species = top_sp

        for sp in top_sp:
            col = "sp_" + sp.replace(" ", "_").replace(".", "_")
            df[col] = (df["bacteria"] == sp).astype(int)

        species_cols = ["sp_" + sp.replace(" ", "_").replace(".", "_") for sp in top_sp]

        # MDR by species
        mdr_by_sp = (
            df.groupby("bacteria")["MDR"]
            .agg(["mean", "count"])
            .query("count >= 50")
            .sort_values("mean", ascending=False)
        )
        self.dataset.mdr_by_species = mdr_by_sp

        # Final feature set
        feature_cols = (
            ["age", "gender_enc", "Diabetes_enc", "Hypertension_enc",
             "Hospital_enc", "Infection_Freq"]
            + species_cols
        )
        self.dataset.feature_cols = feature_cols

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df["MDR"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.dataset.X       = X
        self.dataset.y       = y
        self.dataset.X_train = X_train
        self.dataset.X_test  = X_test
        self.dataset.y_train = y_train
        self.dataset.y_test  = y_test
        self.dataset.processed_df = df

        print(f"  Features: {len(feature_cols)}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
        return self

    def get_dataset(self) -> ResistanceDataset:
        return self.dataset
