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
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier

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
                n_jobs=None,
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
            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                eval_metric="logloss",
            ),
        }

        self._estimator_defs["Ensemble"] = __import__('sklearn.ensemble').ensemble.VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss"))
            ],
            voting='soft'
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def train_all(self) -> "ModelController":
        """Train every estimator on the training split."""
        print("\n[ModelController] Training models …")
        ds = self.dataset

        param_grids = {
            "Random Forest": {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15]
            },
            "Gradient Boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2]
            },
            "XGBoost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            }
        }

        for name, estimator in self._estimator_defs.items():
            if name in param_grids:
                print(f"  Tuning and Fitting {name} …", end="", flush=True)
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_grids[name],
                    n_iter=5, # Keep it fast
                    scoring='roc_auc',
                    cv=3,
                    n_jobs=None,
                    random_state=42
                )
                search.fit(ds.X_train, ds.y_train)
                best_est = search.best_estimator_
            else:
                print(f"  Fitting {name} …", end="", flush=True)
                estimator.fit(ds.X_train, ds.y_train)
                best_est = estimator

            self._models[name] = ResistanceModel(
                name=name,
                estimator=best_est,
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
                est, ds.X, ds.y, cv=5, scoring="roc_auc", n_jobs=None
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
