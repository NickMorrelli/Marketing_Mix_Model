"""
model.py
--------
Fits the Marketing Mix Model using Ordinary Least Squares (OLS) regression.

Why OLS / Linear Regression?
-----------------------------
MMM is fundamentally a regression problem: we want to decompose revenue
into contributions from each marketing channel plus a baseline.

    revenue = baseline
            + β_tv       * tv_adstock
            + β_digital  * digital_adstock
            + β_email    * email_adstock
            + β_social   * social_adstock
            + β_promo    * promo
            + ε  (error term)

Each coefficient (β) tells us: "for every $1 of adstocked spend on this
channel, how much revenue is generated?" That's the channel ROI.

We use statsmodels (not just sklearn) because it provides:
  - R-squared and Adjusted R-squared (model fit quality)
  - p-values for each coefficient (statistical significance)
  - Confidence intervals (uncertainty around estimates)
  - Durbin-Watson statistic (checks for autocorrelation in residuals)

Model Evaluation
----------------
Key metrics we look at:
  - R²          : % of revenue variance explained by the model (higher = better)
  - Adj. R²     : R² penalized for adding more variables (guards against overfitting)
  - p-values    : Is each channel's contribution statistically significant? (< 0.05)
  - Durbin-Watson: Should be close to 2.0 — values far from 2 suggest the model
                   is missing a time-based pattern (seasonality, trend)
  - MAPE        : Mean Absolute Percentage Error — average % error in predictions
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error


# ── Feature Configuration ─────────────────────────────────────────────────────

# These are the columns we use as predictors in the regression
FEATURE_COLS = [
    "tv_adstock_scaled",
    "digital_adstock_scaled",
    "email_adstock_scaled",
    "social_adstock_scaled",
    "promo",
]

# Human-readable labels for reporting
FEATURE_LABELS = {
    "tv_adstock_scaled"     : "TV",
    "digital_adstock_scaled": "Digital",
    "email_adstock_scaled"  : "Email",
    "social_adstock_scaled" : "Social",
    "promo"                 : "Promotions",
    "const"                 : "Baseline",
}


# ── Model Fitting ─────────────────────────────────────────────────────────────

def fit_model(df_scaled: pd.DataFrame):
    """
    Fit an OLS regression model to explain weekly revenue.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Scaled data from data_prep.scale_features().

    Returns
    -------
    result : statsmodels RegressionResults object
        Contains all model statistics, coefficients, and diagnostics.
    X      : pd.DataFrame
        Feature matrix used for fitting (with constant column added).
    y      : pd.Series
        Target variable (revenue).
    """
    print("\nFitting OLS regression model...")

    # ── Build feature matrix ───────────────────────────────────────────────
    X = df_scaled[FEATURE_COLS].copy()

    # sm.add_constant adds an intercept column (the 'baseline' revenue)
    # This is the revenue we'd expect even with zero marketing spend
    X = sm.add_constant(X)

    y = df_scaled["revenue"]

    # ── Fit the model ──────────────────────────────────────────────────────
    model  = sm.OLS(y, X)
    result = model.fit()

    return result, X, y


# ── Diagnostics ───────────────────────────────────────────────────────────────

def evaluate_model(result, y: pd.Series) -> dict:
    """
    Extract and print key model diagnostics.

    Parameters
    ----------
    result : statsmodels RegressionResults
    y      : pd.Series  Actual revenue values.

    Returns
    -------
    dict of key diagnostic metrics.
    """
    y_pred = result.fittedvalues
    mape   = mean_absolute_percentage_error(y, y_pred) * 100

    diagnostics = {
        "r_squared"     : result.rsquared,
        "adj_r_squared" : result.rsquared_adj,
        "mape"          : mape,
        "durbin_watson" : sm.stats.stattools.durbin_watson(result.resid),
        "aic"           : result.aic,
        "bic"           : result.bic,
        "n_obs"         : int(result.nobs),
        "f_statistic"   : result.fvalue,
        "f_pvalue"      : result.f_pvalue,
    }

    _print_diagnostics(diagnostics)
    return diagnostics


def _print_diagnostics(d: dict):
    print("\n  Model Diagnostics:")
    print(f"    R²              : {d['r_squared']:.4f}  ({d['r_squared']*100:.1f}% of revenue variance explained)")
    print(f"    Adjusted R²     : {d['adj_r_squared']:.4f}")
    print(f"    MAPE            : {d['mape']:.2f}%  (avg prediction error)")
    print(f"    Durbin-Watson   : {d['durbin_watson']:.4f}  (2.0 = ideal, no autocorrelation)")
    print(f"    F-statistic     : {d['f_statistic']:.2f}  (p = {d['f_pvalue']:.4f})")
    print(f"    Observations    : {d['n_obs']}")


# ── Coefficient Extraction ────────────────────────────────────────────────────

def extract_coefficients(result, scalers: dict) -> pd.DataFrame:
    """
    Extract model coefficients and convert back to raw spend units for
    interpretability.

    The model was fit on scaled features (divided by mean spend), so
    the raw coefficient tells us revenue per 'average week of spend'.
    We convert back to revenue per $1 of actual spend for ROI reporting.

    Parameters
    ----------
    result  : statsmodels RegressionResults
    scalers : dict  Mean values used for scaling (from data_prep).

    Returns
    -------
    pd.DataFrame with columns:
        channel, coefficient, std_error, p_value, conf_low, conf_high,
        significant, roi_per_dollar
    """
    params    = result.params
    pvalues   = result.pvalues
    conf      = result.conf_int()
    bse       = result.bse   # standard errors

    rows = []
    for feature in params.index:
        coef     = params[feature]
        p_val    = pvalues[feature]
        ci_low   = conf.loc[feature, 0]
        ci_high  = conf.loc[feature, 1]
        std_err  = bse[feature]
        label    = FEATURE_LABELS.get(feature, feature)

        # Convert coefficient back to ROI per $1
        # Scaled coef = revenue per (mean_spend units)
        # ROI per $1  = scaled_coef / mean_spend
        adstock_key = feature.replace("_scaled", "")
        if adstock_key in scalers:
            mean_spend  = scalers[adstock_key]
            roi_per_dollar = coef / (mean_spend + 1e-9)
        else:
            roi_per_dollar = None   # baseline and promo don't have a per-$ ROI

        rows.append({
            "channel"      : label,
            "feature"      : feature,
            "coefficient"  : coef,
            "std_error"    : std_err,
            "p_value"      : p_val,
            "conf_low"     : ci_low,
            "conf_high"    : ci_high,
            "significant"  : p_val < 0.05,
            "roi_per_dollar": roi_per_dollar,
        })

    coef_df = pd.DataFrame(rows)

    print("\n  Channel Coefficients & ROI:")
    print(f"  {'Channel':<15} {'Coef':>10} {'p-value':>10} {'Sig':>5} {'ROI/$':>10}")
    print("  " + "-" * 55)
    for _, row in coef_df.iterrows():
        roi_str = f"${row['roi_per_dollar']:.3f}" if row["roi_per_dollar"] is not None else "  N/A"
        sig_str = "✅" if row["significant"] else "❌"
        print(f"  {row['channel']:<15} {row['coefficient']:>10.2f} {row['p_value']:>10.4f} {sig_str:>5} {roi_str:>10}")

    return coef_df


# ── Revenue Decomposition ─────────────────────────────────────────────────────

def decompose_revenue(result, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose total revenue into contributions from each channel.

    Contribution = coefficient × feature_value  for each week and channel.
    This tells us how much revenue each channel drove in each week.

    Parameters
    ----------
    result : statsmodels RegressionResults
    X      : pd.DataFrame  Feature matrix (with constant).
    df     : pd.DataFrame  Original (unscaled) data for date reference.

    Returns
    -------
    pd.DataFrame  Weekly revenue decomposition by channel.
    """
    params = result.params
    decomp = pd.DataFrame(index=df.index)
    decomp["date"]     = df["date"].values
    decomp["actual"]   = df["revenue"].values
    decomp["predicted"] = result.fittedvalues.values

    for feature in X.columns:
        label = FEATURE_LABELS.get(feature, feature)
        decomp[label] = (params[feature] * X[feature]).values

    return decomp


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_model_pipeline(df_scaled: pd.DataFrame, df: pd.DataFrame, scalers: dict):
    """
    Full modeling pipeline: fit → evaluate → extract coefficients → decompose.

    Returns
    -------
    result       : fitted OLS model
    diagnostics  : dict of model metrics
    coef_df      : coefficient table with ROI
    decomp       : weekly revenue decomposition
    X            : feature matrix
    y            : target variable
    """
    result, X, y    = fit_model(df_scaled)
    diagnostics     = evaluate_model(result, y)
    coef_df         = extract_coefficients(result, scalers)
    decomp          = decompose_revenue(result, X, df)

    return result, diagnostics, coef_df, decomp, X, y
