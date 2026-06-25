"""
data_prep.py
------------
Loads and prepares synthetic marketing mix data for the MMM analysis.

Why Synthetic Data?
-------------------
Real MMM datasets are proprietary (companies guard their spend and revenue
data closely). Using a well-constructed synthetic dataset is standard
practice for portfolio projects and allows us to control the ground truth
so we can verify our model is working correctly.

What is a Marketing Mix Model?
-------------------------------
A Marketing Mix Model (MMM) is a statistical technique that measures the
contribution of each marketing channel (TV, digital, email, etc.) to
overall sales/revenue. It helps answer:
  - Which channels are driving the most revenue?
  - What is the ROI of each channel?
  - How should we reallocate budget to maximize revenue?

Data Structure
--------------
We simulate weekly data over 3 years with the following features:
  - date          : Week start date
  - tv_spend      : Weekly TV advertising spend ($)
  - digital_spend : Weekly digital/paid search spend ($)
  - email_spend   : Weekly email marketing spend ($)
  - social_spend  : Weekly social media spend ($)
  - promo         : Binary flag — was there a promotional event that week?
  - revenue       : Weekly revenue ($) — our target variable

Adstock Transformation
----------------------
Advertising doesn't just affect sales in the week it runs — it has a
'carryover' effect in subsequent weeks (think of a TV ad you saw last week
still influencing your purchase this week). This is called 'adstock'.

We apply adstock to the spend data before modeling:
    adstock[t] = spend[t] + decay_rate * adstock[t-1]

where decay_rate controls how quickly the effect fades (0 = no carryover,
1 = effect never fades). Typical TV decay rates are 0.3–0.7.
"""

import pandas as pd
import numpy as np
import os


# ── Simulation Parameters ─────────────────────────────────────────────────────

RANDOM_SEED  = 42
N_WEEKS      = 156       # 3 years of weekly data

# True channel contributions (ground truth we're trying to recover)
TRUE_COEFFICIENTS = {
    "tv_spend"     : 0.35,   # TV drives 35p per $1 spend (after adstock)
    "digital_spend": 0.50,   # Digital drives 50p per $1 spend
    "email_spend"  : 0.20,   # Email drives 20p per $1 spend
    "social_spend" : 0.15,   # Social drives 15p per $1 spend
    "promo"        : 5000,   # Promos add $5,000 in revenue on average
}
BASE_REVENUE = 20_000        # Weekly baseline revenue with no marketing ($)
NOISE_STD    = 2_000         # Random noise in revenue ($)

# Adstock decay rates per channel
ADSTOCK_DECAY = {
    "tv_spend"     : 0.6,    # TV has high carryover (people remember ads)
    "digital_spend": 0.3,    # Digital fades faster (intent-based)
    "email_spend"  : 0.1,    # Email effect is mostly immediate
    "social_spend" : 0.4,    # Social has moderate carryover
}


# ── Adstock Transformation ────────────────────────────────────────────────────

def apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to a spend array.

    Formula
    -------
    adstock[0] = spend[0]
    adstock[t] = spend[t] + decay * adstock[t-1]

    Parameters
    ----------
    spend : np.ndarray  Weekly spend values.
    decay : float       Decay rate between 0 (no carryover) and 1 (full).

    Returns
    -------
    np.ndarray  Adstocked spend values.
    """
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


# ── Generate Synthetic Data ───────────────────────────────────────────────────

def generate_data(n_weeks: int = N_WEEKS) -> pd.DataFrame:
    """
    Generate synthetic weekly marketing and revenue data.

    The data is designed to reflect realistic patterns:
    - TV spend is high and bursty (flight-based campaigns)
    - Digital spend is consistent with moderate variance
    - Email spend is low and steady
    - Social spend grows over time (reflecting industry trends)
    - Promotions occur ~8 times per year

    Parameters
    ----------
    n_weeks : int  Number of weeks to simulate.

    Returns
    -------
    pd.DataFrame  Weekly data with spend, promo, and revenue columns.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Date index ──────────────────────────────────────────────────────────
    dates = pd.date_range(start="2021-01-04", periods=n_weeks, freq="W-MON")

    # ── Spend variables ────────────────────────────────────────────────────
    # TV: bursty — high spend during campaign flights, zero otherwise
    tv_spend = np.where(
        rng.random(n_weeks) < 0.4,                       # 40% of weeks have TV
        rng.normal(15_000, 3_000, n_weeks).clip(0),
        0
    )

    # Digital: consistent weekly spend with moderate noise
    digital_spend = rng.normal(8_000, 1_500, n_weeks).clip(0)

    # Email: low and steady (cost per send is minimal)
    email_spend = rng.normal(1_500, 300, n_weeks).clip(0)

    # Social: growing over time (increasing investment trend)
    social_base  = np.linspace(2_000, 6_000, n_weeks)   # trend from $2k to $6k
    social_spend = (social_base + rng.normal(0, 500, n_weeks)).clip(0)

    # ── Promotions ─────────────────────────────────────────────────────────
    # ~8 promotional weeks per year = 8/52 ≈ 15% probability per week
    promo = (rng.random(n_weeks) < 0.15).astype(int)

    # ── Apply adstock ──────────────────────────────────────────────────────
    tv_adstock      = apply_adstock(tv_spend,      ADSTOCK_DECAY["tv_spend"])
    digital_adstock = apply_adstock(digital_spend, ADSTOCK_DECAY["digital_spend"])
    email_adstock   = apply_adstock(email_spend,   ADSTOCK_DECAY["email_spend"])
    social_adstock  = apply_adstock(social_spend,  ADSTOCK_DECAY["social_spend"])

    # ── Generate revenue using true coefficients ───────────────────────────
    revenue = (
        BASE_REVENUE
        + TRUE_COEFFICIENTS["tv_spend"]      * tv_adstock
        + TRUE_COEFFICIENTS["digital_spend"] * digital_adstock
        + TRUE_COEFFICIENTS["email_spend"]   * email_adstock
        + TRUE_COEFFICIENTS["social_spend"]  * social_adstock
        + TRUE_COEFFICIENTS["promo"]         * promo
        + rng.normal(0, NOISE_STD, n_weeks)  # random noise
    )

    df = pd.DataFrame({
        "date"           : dates,
        "tv_spend"       : tv_spend.round(2),
        "digital_spend"  : digital_spend.round(2),
        "email_spend"    : email_spend.round(2),
        "social_spend"   : social_spend.round(2),
        "promo"          : promo,
        # Adstocked versions (used in modeling)
        "tv_adstock"     : tv_adstock.round(2),
        "digital_adstock": digital_adstock.round(2),
        "email_adstock"  : email_adstock.round(2),
        "social_adstock" : social_adstock.round(2),
        "revenue"        : revenue.round(2),
    })

    return df


# ── Feature Scaling ───────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Scale adstock features by dividing by their mean.

    Why scale?
    ----------
    TV spend might average $10,000/week while email averages $1,500/week.
    Without scaling, the model coefficients aren't comparable across channels.
    Dividing by the mean puts each channel on a 'per-unit-of-typical-spend'
    basis, making coefficients directly interpretable as ROI.

    Parameters
    ----------
    df : pd.DataFrame  Output of generate_data().

    Returns
    -------
    df_scaled : pd.DataFrame  DataFrame with scaled adstock columns added.
    scalers   : dict          Mean values used for scaling (needed to
                              reverse-transform coefficients later).
    """
    adstock_cols = ["tv_adstock", "digital_adstock", "email_adstock", "social_adstock"]
    scalers = {}
    df_scaled = df.copy()

    for col in adstock_cols:
        mean = df[col].mean()
        scalers[col] = mean
        df_scaled[f"{col}_scaled"] = df[col] / (mean + 1e-9)

    return df_scaled, scalers


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_prep_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full data preparation pipeline.

    Returns
    -------
    df        : raw generated data
    df_scaled : data with scaled adstock features
    scalers   : mean values used for scaling
    """
    print("Generating synthetic marketing mix data...")
    df = generate_data()
    df_scaled, scalers = scale_features(df)

    print(f"  Weeks of data    : {len(df)}")
    print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total revenue    : ${df['revenue'].sum():,.0f}")
    print(f"  Total TV spend   : ${df['tv_spend'].sum():,.0f}")
    print(f"  Total digital    : ${df['digital_spend'].sum():,.0f}")
    print(f"  Total email      : ${df['email_spend'].sum():,.0f}")
    print(f"  Total social     : ${df['social_spend'].sum():,.0f}")
    print(f"  Promo weeks      : {df['promo'].sum()} ({df['promo'].mean():.1%})")

    # Save to CSV for reference
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "mmm_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Data saved to   : {out_path}")

    return df, df_scaled, scalers


if __name__ == "__main__":
    df, df_scaled, scalers = run_prep_pipeline()
    print("\nSample data:")
    print(df[["date", "tv_spend", "digital_spend", "revenue"]].head(10).to_string(index=False))
