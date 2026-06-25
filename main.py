"""
main.py
-------
End-to-end pipeline for the Marketing Mix Model (MMM) analysis.

Usage
-----
    python main.py          (run from the marketing_mix_model/ folder)

Steps
-----
1. Generate synthetic weekly marketing and revenue data.
2. Apply adstock transformations and scale features.
3. Fit OLS regression model and evaluate diagnostics.
4. Extract channel coefficients and ROI estimates.
5. Decompose revenue into channel contributions.
6. Optimize budget allocation to maximize predicted revenue.
7. Generate and save all visualizations to /outputs/.
"""

import os
import sys

from src.data_prep      import run_prep_pipeline
from src.model          import run_model_pipeline
from src.optimizer      import run_optimization_pipeline
from src.visualizations import generate_all_plots

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MARKETING MIX MODEL (MMM)")
    print("  OLS Regression + Budget Optimization")
    print("=" * 65)

    # ── Step 1: Data Preparation ───────────────────────────────────────────
    print("\n[1/4] Data Preparation")
    df, df_scaled, scalers = run_prep_pipeline()

    # ── Step 2: Model Fitting ──────────────────────────────────────────────
    print("\n[2/4] Model Fitting & Evaluation")
    result, diagnostics, coef_df, decomp, X, y = run_model_pipeline(
        df_scaled, df, scalers
    )

    # ── Step 3: Budget Optimization ────────────────────────────────────────
    print("\n[3/4] Budget Optimization")
    opt_df, scenario_df, optimal_revenue, total_budget, rois, baseline = (
        run_optimization_pipeline(coef_df, df)
    )

    # ── Step 4: Visualizations ─────────────────────────────────────────────
    print("\n[4/4] Generating Visualizations")
    generate_all_plots(
        df, decomp, coef_df, opt_df, scenario_df, diagnostics, total_budget
    )

    # ── Done ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
