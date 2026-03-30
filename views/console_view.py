 

from typing import Dict, Optional

import pandas as pd


class ConsoleView:
    """Plain-text helpers (stdout)."""

    def resistance_summary(
        self,
        resistance_rates: Dict[str, float],
        mdr_by_species: Optional[pd.DataFrame],
        mdr_rate: float,
        n_samples: int,
    ) -> None:
        print("\n══════  Resistance Summary  ══════")
        print(f"Total isolates : {n_samples:,}")
        print(f"MDR prevalence : {mdr_rate:.1%}\n")
        for ab, rate in resistance_rates.items():
            bar = "█" * int(rate * 40)
            print(f"  {ab:30s} {rate:6.1%}  {bar}")
        if mdr_by_species is not None:
            print("\nMDR Rate by Species:")
            print(mdr_by_species.to_string())

    def model_summary(self, results: dict) -> None:
        print("\n══════  Model Performance  ══════")
        for name, rm in results.items():
            print(f"  {name:25s}  AUC={rm.auc:.4f}  Acc={rm.accuracy:.4f}")

    def treatment_recommendations(self, resistance_rates: Dict[str, float]) -> None:
        print("\n══════  Treatment Recommendations  ══════")
        avoid   = [ab for ab, r in resistance_rates.items() if r >= 0.50]
        caution = [ab for ab, r in resistance_rates.items() if 0.15 <= r < 0.50]
        suggest = [ab for ab, r in resistance_rates.items() if r < 0.15]
        print("  ⛔ Avoid   :", ", ".join(avoid) or "—")
        print("  ⚠️  Caution :", ", ".join(caution) or "—")
        print("  ✅ Consider:", ", ".join(suggest) or "—")
