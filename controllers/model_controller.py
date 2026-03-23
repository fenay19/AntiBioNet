"""
controllers/model_controller.py
--------------------------------
Trains all classifiers, runs cross-validation, computes metrics,
and stores results as ResistanceModel objects.
"""

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

from models.resistance_dataset import ResistanceDataset
from models.resistance_model import ResistanceModel

warnings.filterwarnings("ignore")


class ModelController:
    """
    Responsible for:
      1. Instantiating all classifiers
      2. Training on the pre-split data in ResistanceDataset
      3. Running 5-fold cross-validation
      4. Computing test-set metrics
      5. Extracting feature importances
    """

    def __init__(self, dataset: ResistanceDataset):
        self.dataset  = dataset
        self._models: Dict[str, ResistanceModel] = {}

        # Classifier definitions — extend here to add more models
        self._estimator_defs = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
            ),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def train_all(self) -> "ModelController":
        """Train every estimator on the training split."""
        print("\n[ModelController] Training models …")
        ds = self.dataset

        for name, estimator in self._estimator_defs.items():
            print(f"  Fitting {name} …", end="", flush=True)
            estimator.fit(ds.X_train, ds.y_train)
            self._models[name] = ResistanceModel(
                name=name,
                estimator=estimator,
                feature_cols=ds.feature_cols,
            )
            print(" done")
        return self

    def evaluate_all(self) -> "ModelController":
        """Evaluate all trained models on test split + 5-fold CV."""
        print("\n[ModelController] Evaluating models …")
        ds = self.dataset

        for name, rm in self._models.items():
            est   = rm.estimator
            y_pred  = est.predict(ds.X_test)
            y_proba = est.predict_proba(ds.X_test)[:, 1]

            cv_scores = cross_val_score(
                est, ds.X, ds.y, cv=5, scoring="roc_auc", n_jobs=-1
            )

            rm.y_pred   = y_pred
            rm.y_proba  = y_proba
            rm.accuracy = float((y_pred == ds.y_test).mean())
            rm.auc      = float(roc_auc_score(ds.y_test, y_proba))
            rm.cv_auc   = float(cv_scores.mean())
            rm.cv_std   = float(cv_scores.std())
            rm.report   = classification_report(
                ds.y_test, y_pred, target_names=["Non-MDR", "MDR"]
            )

            # Feature importances (tree-based models)
            if hasattr(est, "feature_importances_"):
                rm.feature_importances = (
                    pd.Series(est.feature_importances_, index=ds.feature_cols)
                    .sort_values(ascending=False)
                )

        return self

    def get_results(self) -> Dict[str, ResistanceModel]:
        return self._models

    def get_best_model(self, metric: str = "auc") -> ResistanceModel:
        """Return the ResistanceModel with the highest test AUC (or accuracy)."""
        key = "auc" if metric == "auc" else "accuracy"
        return max(self._models.values(), key=lambda rm: getattr(rm, key))
