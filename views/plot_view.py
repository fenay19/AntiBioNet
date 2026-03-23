"""
views/plot_view.py
------------------
Matplotlib-based plotting helpers used by ReportController.save_plots().

NOTE: The Streamlit app renders everything with Plotly, so these methods
      are only invoked if you call  report_ctrl.save_plots()  from a
      CLI/notebook context.  They exist so that
      `controllers.report_controller` can import without error.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotView:
    """Matplotlib chart helpers."""

    # ── helpers ────────────────────────────────────────────────────
    @staticmethod
    def _save(fig, path: str) -> None:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#161b22")
        plt.close(fig)
        print(f"  → saved {path}")

    # ── public charts ─────────────────────────────────────────────
    def resistance_rates_chart(
        self, resistance_rates: Dict[str, float], save_path: str
    ) -> None:
        labels = list(resistance_rates.keys())
        vals = [resistance_rates[l] * 100 for l in labels]
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        ax.barh(labels, vals, color="#00e5c3")
        ax.set_xlabel("Resistance (%)", color="#e6edf3")
        ax.set_title("Antibiotic Resistance Rates", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        self._save(fig, save_path)

    def model_comparison_chart(
        self, results: dict, save_path: str
    ) -> None:
        names = list(results.keys())
        aucs = [results[n].auc for n in names]
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        ax.bar(names, aucs, color="#00b59a")
        ax.set_ylabel("AUC", color="#e6edf3")
        ax.set_title("Model Comparison", color="#e6edf3")
        ax.tick_params(colors="#8b949e", axis="x", rotation=30)
        ax.tick_params(colors="#8b949e", axis="y")
        self._save(fig, save_path)

    def feature_importance_chart(
        self,
        feature_importances: pd.Series,
        model_name: str,
        save_path: str,
    ) -> None:
        fi = feature_importances.head(12)
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        ax.barh(fi.index, fi.values, color="#00e5c3")
        ax.set_title(f"Feature Importance — {model_name}", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        self._save(fig, save_path)

    def mdr_by_species_chart(
        self, mdr_by_species: pd.DataFrame, save_path: str
    ) -> None:
        df = mdr_by_species.head(10).reset_index()
        df.columns = ["Species", "MDR_Rate", "Count"]
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        ax.bar(df["Species"], df["MDR_Rate"] * 100, color="#ffb830")
        ax.set_ylabel("MDR Rate (%)", color="#e6edf3")
        ax.set_title("MDR Rate by Species", color="#e6edf3")
        ax.tick_params(colors="#8b949e", axis="x", rotation=30)
        ax.tick_params(colors="#8b949e", axis="y")
        self._save(fig, save_path)

    def dashboard(
        self,
        resistance_rates: Dict[str, float],
        results: dict,
        mdr_by_species: Optional[pd.DataFrame],
        best_model,
        save_path: str,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#161b22")
        for ax in axes.flat:
            ax.set_facecolor("#161b22")

        # Top-left: resistance rates
        labels = list(resistance_rates.keys())
        vals = [resistance_rates[l] * 100 for l in labels]
        axes[0, 0].barh(labels, vals, color="#00e5c3")
        axes[0, 0].set_title("Resistance Rates", color="#e6edf3")

        # Top-right: model comparison
        names = list(results.keys())
        axes[0, 1].bar(names, [results[n].auc for n in names], color="#00b59a")
        axes[0, 1].set_title("Model AUC", color="#e6edf3")
        axes[0, 1].tick_params(axis="x", rotation=30)

        # Bottom-left: MDR by species
        if mdr_by_species is not None:
            df = mdr_by_species.head(8).reset_index()
            df.columns = ["Species", "MDR_Rate", "Count"]
            axes[1, 0].bar(df["Species"], df["MDR_Rate"] * 100, color="#ffb830")
            axes[1, 0].set_title("MDR by Species", color="#e6edf3")
            axes[1, 0].tick_params(axis="x", rotation=30)

        # Bottom-right: feature importance
        if best_model and best_model.feature_importances is not None:
            fi = best_model.feature_importances.head(10)
            axes[1, 1].barh(fi.index, fi.values, color="#00e5c3")
            axes[1, 1].set_title("Feature Importance", color="#e6edf3")

        for ax in axes.flat:
            ax.tick_params(colors="#8b949e")

        fig.tight_layout(pad=3)
        self._save(fig, save_path)
