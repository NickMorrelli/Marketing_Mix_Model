"""
optimizer.py
------------
Uses the fitted MMM coefficients to recommend an optimal budget allocation
across channels to maximize predicted revenue.

Why Optimization?
-----------------
Once we know the ROI of each channel (from the regression), we can answer:
"Given a total budget of $X per week, how should we split it across
TV, Digital, Email, and Social to maximize revenue?"

This transforms MMM from a descriptive tool ("here's what drove revenue")
into a prescriptive tool ("here's what you should do next").

Approach
--------
We use scipy.optimize.minimize with a 'SLSQP' method (Sequential Least
Squares Programming) — a gradient-based optimizer that handles:
  - Equality constraints : total spend must equal the budget
  - Inequality constraints: each channel spend must be >= 0
  - Bounds              : optional min/max spend per channel

Diminishing Returns
-------------------
In reality, doubling your TV spend doesn't double your revenue — there are
diminishing returns (you run out of new audiences to reach). We model this
with a square-root transformation:

    contribution = coefficient * sqrt(spend)

This is a simplification of the full Hill/saturation curve used in
production MMMs, but captures the key diminishing returns concept.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Channel Configuration ─────────────────────────────────────────────────────

CHANNELS = ["tv_spend", "digital_spend", "email_spend", "social_spend"]

CHANNEL_LABELS = {
    "tv_spend"     : "TV",
    "digital_spend": "Digital",
    "email_spend"  : "Email",
    "social_spend" : "Social",
}

# Minimum spend per channel per week ($) — business constraints
# (e.g., you can't run a TV campaign for less than $2,000)
MIN_SPEND = {
    "tv_spend"     : 0,        # TV can be turned off
    "digital_spend": 2_000,    # Always-on digital minimum
    "email_spend"  : 500,      # Email is cheap, keep it on
    "social_spend" : 500,      # Social minimum presence
}

# Maximum spend per channel per week ($) — capacity constraints
MAX_SPEND = {
    "tv_spend"     : 50_000,
    "digital_spend": 30_000,
    "email_spend"  : 5_000,
    "social_spend" : 20_000,
}


# ── Revenue Prediction Function ───────────────────────────────────────────────

def predict_revenue(spends: np.ndarray, rois: dict, adstock_decay: dict,
                    baseline: float) -> float:
    """
    Predict weekly revenue from a given spend allocation.

    Uses square-root diminishing returns:
        contribution = roi * sqrt(spend * (1 + decay))

    The (1 + decay) factor approximates the adstock multiplier effect —
    spend this week generates some effect next week too.

    Parameters
    ----------
    spends       : np.ndarray  Spend per channel [tv, digital, email, social].
    rois         : dict        ROI per $1 per channel (from model coefficients).
    adstock_decay: dict        Decay rates per channel.
    baseline     : float       Baseline revenue (model intercept).

    Returns
    -------
    float  Predicted weekly revenue.
    """
    channels = CHANNELS
    revenue  = baseline

    for i, channel in enumerate(channels):
        spend = spends[i]
        roi   = rois.get(channel, 0)
        decay = adstock_decay.get(channel, 0)

        # Diminishing returns via square-root + adstock multiplier
        adstock_multiplier = 1 / (1 - decay + 1e-9)  # geometric series sum
        revenue += roi * np.sqrt(spend * adstock_multiplier + 1e-9)

    return revenue


# ── Optimization ──────────────────────────────────────────────────────────────

def optimize_budget(total_budget: float, rois: dict, adstock_decay: dict,
                    baseline: float) -> pd.DataFrame:
    """
    Find the spend allocation that maximizes predicted revenue subject to:
      1. Total spend == total_budget
      2. Each channel spend >= MIN_SPEND[channel]
      3. Each channel spend <= MAX_SPEND[channel]

    Parameters
    ----------
    total_budget  : float  Total weekly budget to allocate ($).
    rois          : dict   ROI per $1 per channel.
    adstock_decay : dict   Decay rates per channel.
    baseline      : float  Model intercept (baseline revenue).

    Returns
    -------
    pd.DataFrame  Optimal spend allocation with predicted contributions.
    """
    n = len(CHANNELS)

    # ── Objective: minimize NEGATIVE revenue (scipy minimizes, not maximizes)
    def objective(spends):
        return -predict_revenue(spends, rois, adstock_decay, baseline)

    # ── Constraints ────────────────────────────────────────────────────────
    constraints = [{
        "type": "eq",
        "fun" : lambda spends: np.sum(spends) - total_budget
    }]

    # ── Bounds ─────────────────────────────────────────────────────────────
    bounds = [
        (MIN_SPEND[ch], min(MAX_SPEND[ch], total_budget))
        for ch in CHANNELS
    ]

    # ── Initial guess: equal split (subject to min spend constraints) ──────
    x0 = np.array([total_budget / n] * n)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    # Renormalize to budget
    x0 = x0 / x0.sum() * total_budget

    # ── Run optimizer ──────────────────────────────────────────────────────
    result = minimize(
        objective, x0,
        method     = "SLSQP",
        bounds     = bounds,
        constraints= constraints,
        options    = {"ftol": 1e-9, "maxiter": 1000}
    )

    if not result.success:
        print(f"  ⚠️  Optimizer warning: {result.message}")

    optimal_spends = result.x
    optimal_revenue = predict_revenue(optimal_spends, rois, adstock_decay, baseline)

    # ── Build results DataFrame ────────────────────────────────────────────
    rows = []
    for i, channel in enumerate(CHANNELS):
        spend = optimal_spends[i]
        decay = adstock_decay.get(channel, 0)
        roi   = rois.get(channel, 0)
        adstock_mult = 1 / (1 - decay + 1e-9)
        contribution = roi * np.sqrt(spend * adstock_mult + 1e-9)

        rows.append({
            "channel"         : CHANNEL_LABELS[channel],
            "optimal_spend"   : spend,
            "spend_share"     : spend / total_budget,
            "contribution"    : contribution,
            "revenue_share"   : contribution / (optimal_revenue - baseline + 1e-9),
        })

    opt_df = pd.DataFrame(rows)
    return opt_df, optimal_revenue


# ── Scenario Comparison ───────────────────────────────────────────────────────

def compare_scenarios(rois: dict, adstock_decay: dict, baseline: float,
                      budgets: list[float]) -> pd.DataFrame:
    """
    Run optimization across multiple budget levels to show the revenue
    curve as budget increases (useful for budget planning conversations).

    Parameters
    ----------
    rois          : dict
    adstock_decay : dict
    baseline      : float
    budgets       : list of float  Budget levels to test.

    Returns
    -------
    pd.DataFrame  One row per budget scenario with optimal revenue.
    """
    rows = []
    for budget in budgets:
        _, opt_revenue = optimize_budget(budget, rois, adstock_decay, baseline)
        rows.append({
            "weekly_budget"   : budget,
            "annual_budget"   : budget * 52,
            "predicted_revenue": opt_revenue,
            "predicted_annual" : opt_revenue * 52,
            "revenue_roi"     : (opt_revenue - baseline) / (budget + 1e-9),
        })
    return pd.DataFrame(rows)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_optimization_pipeline(coef_df: pd.DataFrame, df: pd.DataFrame) -> tuple:
    """
    Run the full budget optimization pipeline.

    Extracts ROIs from model coefficients, runs optimization at the
    observed average budget, and compares scenarios at multiple budget levels.

    Parameters
    ----------
    coef_df : pd.DataFrame  Coefficient table from model.extract_coefficients().
    df      : pd.DataFrame  Original data (to compute average spend).

    Returns
    -------
    opt_df       : optimal allocation at current budget
    scenario_df  : revenue curve across budget levels
    optimal_revenue : predicted revenue at optimal allocation
    total_budget : budget used for optimization
    """
    from src.data_prep import ADSTOCK_DECAY

    print("\nRunning budget optimization...")

    # ── Extract ROIs from model ────────────────────────────────────────────
    rois = {}
    baseline = 0
    for _, row in coef_df.iterrows():
        if row["feature"] == "const":
            baseline = row["coefficient"]
        elif row["roi_per_dollar"] is not None:
            # Map back to original channel name
            channel = row["feature"].replace("_adstock_scaled", "_spend")
            rois[channel] = row["roi_per_dollar"]

    # ── Current average weekly budget ─────────────────────────────────────
    spend_cols    = ["tv_spend", "digital_spend", "email_spend", "social_spend"]
    total_budget  = df[spend_cols].sum(axis=1).mean()
    print(f"  Current avg weekly budget: ${total_budget:,.0f}")

    # ── Optimize at current budget ─────────────────────────────────────────
    opt_df, optimal_revenue = optimize_budget(total_budget, rois, ADSTOCK_DECAY, baseline)

    print(f"\n  Optimal Allocation (Budget: ${total_budget:,.0f}/week):")
    print(f"  {'Channel':<12} {'Spend':>10} {'Share':>8} {'Contribution':>14}")
    print("  " + "-" * 48)
    for _, row in opt_df.iterrows():
        print(f"  {row['channel']:<12} ${row['optimal_spend']:>9,.0f} "
              f"{row['spend_share']:>7.1%} ${row['contribution']:>13,.0f}")
    print(f"\n  Predicted weekly revenue: ${optimal_revenue:,.0f}")

    # ── Scenario comparison ────────────────────────────────────────────────
    budgets     = [10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000]
    scenario_df = compare_scenarios(rois, ADSTOCK_DECAY, baseline, budgets)

    return opt_df, scenario_df, optimal_revenue, total_budget, rois, baseline
