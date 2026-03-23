"""
controllers/report_controller.py
---------------------------------
Handles all output: console summaries, matplotlib charts, and
treatment strategy recommendations.
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from models.resistance_dataset import ANTIBIOTIC_COLS, ResistanceDataset
from models.resistance_model import ResistanceModel
from views.console_view import ConsoleView
from views.plot_view import PlotView


class ReportController:
    """
    Responsible for:
      1. Printing resistance & model summaries to stdout
      2. Printing evidence-based treatment recommendations
      3. Saving all matplotlib figures to disk
    """

    def __init__(
        self,
        dataset: ResistanceDataset,
        results: Dict[str, ResistanceModel],
    ):
        self.dataset = dataset
        self.results = results
        self._console = ConsoleView()
        self._plot    = PlotView()

    # ── Public API ────────────────────────────────────────────────────────────

    def print_resistance_summary(self) -> None:
        self._console.resistance_summary(
            resistance_rates=self.dataset.resistance_rates,
            mdr_by_species=self.dataset.mdr_by_species,
            mdr_rate=self.dataset.mdr_rate,
            n_samples=self.dataset.n_samples,
        )

    def print_model_summary(self) -> None:
        self._console.model_summary(self.results)

    def print_treatment_recommendations(self) -> None:
        self._console.treatment_recommendations(self.dataset.resistance_rates)

    def save_plots(self, output_dir: str = "outputs") -> None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[ReportController] Saving plots to '{output_dir}/' …")

        # 1 — Resistance rates bar chart
        self._plot.resistance_rates_chart(
            resistance_rates=self.dataset.resistance_rates,
            save_path=os.path.join(output_dir, "01_resistance_rates.png"),
        )

        # 2 — Model comparison bar chart
        self._plot.model_comparison_chart(
            results=self.results,
            save_path=os.path.join(output_dir, "02_model_comparison.png"),
        )

        # 3 — Feature importance (best tree model)
        best = max(
            (rm for rm in self.results.values() if rm.feature_importances is not None),
            key=lambda rm: rm.auc,
            default=None,
        )
        if best:
            self._plot.feature_importance_chart(
                feature_importances=best.feature_importances,
                model_name=best.name,
                save_path=os.path.join(output_dir, "03_feature_importance.png"),
            )

        # 4 — MDR rate by species
        if self.dataset.mdr_by_species is not None:
            self._plot.mdr_by_species_chart(
                mdr_by_species=self.dataset.mdr_by_species,
                save_path=os.path.join(output_dir, "04_mdr_by_species.png"),
            )

        # 5 — Combined dashboard
        self._plot.dashboard(
            resistance_rates=self.dataset.resistance_rates,
            results=self.results,
            mdr_by_species=self.dataset.mdr_by_species,
            best_model=best,
            save_path=os.path.join(output_dir, "05_dashboard.png"),
        )

        print("  All plots saved.")
