"""
visualizations.py
-----------------
Generates all plots for the Marketing Mix Model analysis.

Charts
------
1. Spend Over Time       – weekly spend by channel (area chart)
2. Model Fit             – actual vs predicted revenue
3. Revenue Decomposition – stacked bar of channel contributions
4. Channel ROI           – bar chart of ROI per $1 spent
5. Budget Optimization   – optimal vs current allocation
6. Budget Scenario Curve – revenue vs total budget
7. Summary Dashboard     – key metrics in one view
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────────

CHANNEL_COLORS = {
    "TV"         : "#E74C3C",
    "Digital"    : "#3498DB",
    "Email"      : "#2ECC71",
    "Social"     : "#F39C12",
    "Promotions" : "#9B59B6",
    "Baseline"   : "#BDC3C7",
}

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "figure.dpi"       : 120,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, filename):
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def _gbp(x, pos):
    """Formatter for $ values on axes."""
    return f"${x:,.0f}"


# ── Plot 1: Spend Over Time ───────────────────────────────────────────────────

def plot_spend_over_time(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Weekly Marketing Spend by Channel", fontsize=14, fontweight="bold")

    channels = [
        ("tv_spend",      "TV Spend",      CHANNEL_COLORS["TV"]),
        ("digital_spend", "Digital Spend", CHANNEL_COLORS["Digital"]),
        ("email_spend",   "Email Spend",   CHANNEL_COLORS["Email"]),
        ("social_spend",  "Social Spend",  CHANNEL_COLORS["Social"]),
    ]

    for ax, (col, label, color) in zip(axes, channels):
        ax.fill_between(df["date"], df[col], color=color, alpha=0.7)
        ax.plot(df["date"], df[col], color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _save(fig, "01_spend_over_time.png")


# ── Plot 2: Actual vs Predicted Revenue ───────────────────────────────────────

def plot_model_fit(df: pd.DataFrame, decomp: pd.DataFrame, diagnostics: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    # Top: actual vs predicted
    ax1.plot(decomp["date"], decomp["actual"],    color="#2C3E50", linewidth=1.5,
             label="Actual Revenue", alpha=0.9)
    ax1.plot(decomp["date"], decomp["predicted"], color="#E74C3C", linewidth=1.5,
             linestyle="--", label="Predicted Revenue", alpha=0.9)
    ax1.set_title(
        f"Actual vs Predicted Revenue  |  R² = {diagnostics['r_squared']:.3f}  |  MAPE = {diagnostics['mape']:.1f}%"
    )
    ax1.set_ylabel("Weekly Revenue ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))
    ax1.legend()

    # Bottom: residuals
    residuals = decomp["actual"] - decomp["predicted"]
    ax2.bar(decomp["date"], residuals, color=np.where(residuals >= 0, "#2ECC71", "#E74C3C"),
            alpha=0.7, width=5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Residuals (Actual − Predicted)")
    ax2.set_ylabel("Residual ($)")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))

    plt.tight_layout()
    _save(fig, "02_model_fit.png")


# ── Plot 3: Revenue Decomposition ─────────────────────────────────────────────

def plot_revenue_decomposition(decomp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))

    stack_cols = ["Baseline", "TV", "Digital", "Email", "Social", "Promotions"]
    stack_cols = [c for c in stack_cols if c in decomp.columns]
    colors     = [CHANNEL_COLORS.get(c, "#999") for c in stack_cols]

    ax.stackplot(decomp["date"],
                 [decomp[c] for c in stack_cols],
                 labels=stack_cols, colors=colors, alpha=0.85)

    ax.plot(decomp["date"], decomp["actual"], color="black", linewidth=1.2,
            linestyle="--", label="Actual Revenue", alpha=0.8)

    ax.set_title("Revenue Decomposition by Channel", fontweight="bold")
    ax.set_ylabel("Weekly Revenue ($)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    _save(fig, "03_revenue_decomposition.png")


# ── Plot 4: Channel ROI ───────────────────────────────────────────────────────

def plot_channel_roi(coef_df: pd.DataFrame):
    # Only channels with a ROI per dollar (exclude baseline and promo)
    roi_df = coef_df[coef_df["roi_per_dollar"].notna()].copy()
    roi_df = roi_df.sort_values("roi_per_dollar", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CHANNEL_COLORS.get(ch, "#999") for ch in roi_df["channel"]]
    bars   = ax.barh(roi_df["channel"], roi_df["roi_per_dollar"], color=colors, alpha=0.85)

    for bar, val in zip(bars, roi_df["roi_per_dollar"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"${val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Revenue Generated per $1 Spent")
    ax.set_title("Channel ROI — Revenue per $1 of Adstocked Spend", fontweight="bold")
    ax.set_xlim(0, roi_df["roi_per_dollar"].max() * 1.3)

    plt.tight_layout()
    _save(fig, "04_channel_roi.png")


# ── Plot 5: Budget Optimization ───────────────────────────────────────────────

def plot_budget_optimization(opt_df: pd.DataFrame, df: pd.DataFrame, total_budget: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Budget Optimization  |  Weekly Budget: ${total_budget:,.0f}",
                 fontsize=13, fontweight="bold")

    channels = opt_df["channel"].tolist()
    colors   = [CHANNEL_COLORS.get(ch, "#999") for ch in channels]

    # Current allocation
    spend_cols = {"TV": "tv_spend", "Digital": "digital_spend",
                  "Email": "email_spend", "Social": "social_spend"}
    current_spends = [df[spend_cols[ch]].mean() for ch in channels]
    current_total  = sum(current_spends)
    current_shares = [s / current_total for s in current_spends]

    # Optimal allocation
    optimal_shares = opt_df["spend_share"].tolist()

    x = np.arange(len(channels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, current_shares, width, label="Current",  color=colors, alpha=0.5)
    bars2 = ax1.bar(x + width/2, optimal_shares, width, label="Optimal", color=colors, alpha=0.9)

    for bar, val in zip(bars1, current_shares):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.0%}", ha="center", fontsize=8)
    for bar, val in zip(bars2, optimal_shares):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.0%}", ha="center", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(channels)
    ax1.set_ylabel("Budget Share")
    ax1.set_title("Current vs Optimal Budget Share")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend()

    # Optimal spend amounts
    optimal_spends = opt_df["optimal_spend"].tolist()
    bars3 = ax2.bar(channels, optimal_spends, color=colors, alpha=0.85)
    for bar, val in zip(bars3, optimal_spends):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"${val:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Weekly Spend ($)")
    ax2.set_title("Optimal Weekly Spend by Channel")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))

    plt.tight_layout()
    _save(fig, "05_budget_optimization.png")


# ── Plot 6: Budget Scenario Curve ─────────────────────────────────────────────

def plot_budget_scenarios(scenario_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(scenario_df["weekly_budget"], scenario_df["predicted_revenue"],
            "o-", color="#3498DB", linewidth=2, markersize=7)

    for _, row in scenario_df.iterrows():
        ax.annotate(f"${row['predicted_revenue']:,.0f}",
                    (row["weekly_budget"], row["predicted_revenue"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)

    ax.set_xlabel("Weekly Marketing Budget ($)")
    ax.set_ylabel("Predicted Weekly Revenue ($)")
    ax.set_title("Revenue vs Marketing Budget\n(Optimal Allocation at Each Level)",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_gbp))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))

    plt.tight_layout()
    _save(fig, "06_budget_scenarios.png")


# ── Plot 7: Summary Dashboard ─────────────────────────────────────────────────

def plot_summary_dashboard(decomp, coef_df, opt_df, diagnostics, total_budget):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Marketing Mix Model — Summary Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top-left: Actual vs Predicted ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(decomp["date"], decomp["actual"],    color="#2C3E50", linewidth=1.2, label="Actual")
    ax1.plot(decomp["date"], decomp["predicted"], color="#E74C3C", linewidth=1.2,
             linestyle="--", label="Predicted")
    ax1.set_title(f"Model Fit  (R²={diagnostics['r_squared']:.3f}, MAPE={diagnostics['mape']:.1f}%)")
    ax1.set_ylabel("Revenue ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))
    ax1.legend(fontsize=8)

    # ── Top-right: Channel ROI ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    roi_df = coef_df[coef_df["roi_per_dollar"].notna()].sort_values("roi_per_dollar")
    colors = [CHANNEL_COLORS.get(ch, "#999") for ch in roi_df["channel"]]
    ax2.barh(roi_df["channel"], roi_df["roi_per_dollar"], color=colors, alpha=0.85)
    ax2.set_xlabel("ROI per $1")
    ax2.set_title("Channel ROI")

    # ── Bottom-left: Revenue decomposition (avg) ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    decomp_cols = ["Baseline", "TV", "Digital", "Email", "Social", "Promotions"]
    decomp_cols = [c for c in decomp_cols if c in decomp.columns]
    avg_contribs = [decomp[c].mean() for c in decomp_cols]
    colors3      = [CHANNEL_COLORS.get(c, "#999") for c in decomp_cols]
    ax3.pie(avg_contribs, labels=decomp_cols, colors=colors3,
            autopct="%1.0f%%", startangle=140,
            textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white"})
    ax3.set_title("Avg Revenue Attribution")

    # ── Bottom-center: Optimal budget allocation ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    opt_colors = [CHANNEL_COLORS.get(ch, "#999") for ch in opt_df["channel"]]
    bars = ax4.bar(opt_df["channel"], opt_df["optimal_spend"], color=opt_colors, alpha=0.85)
    for bar, val in zip(bars, opt_df["optimal_spend"]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"${val:,.0f}", ha="center", fontsize=7)
    ax4.set_title(f"Optimal Allocation (${total_budget:,.0f}/wk)")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(_gbp))
    ax4.tick_params(axis="x", labelsize=8)

    # ── Bottom-right: Key metrics table ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["R²",              f"{diagnostics['r_squared']:.3f}"],
        ["Adj. R²",         f"{diagnostics['adj_r_squared']:.3f}"],
        ["MAPE",            f"{diagnostics['mape']:.1f}%"],
        ["Durbin-Watson",   f"{diagnostics['durbin_watson']:.3f}"],
        ["Weekly Budget",   f"${total_budget:,.0f}"],
        ["Observations",    str(diagnostics["n_obs"])],
    ]
    tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax5.set_title("Model Statistics", pad=12)

    plt.tight_layout()
    _save(fig, "07_summary_dashboard.png")


# ── Run All ───────────────────────────────────────────────────────────────────

def generate_all_plots(df, decomp, coef_df, opt_df, scenario_df,
                       diagnostics, total_budget):
    print("\nGenerating visualizations...")
    plot_spend_over_time(df)
    plot_model_fit(df, decomp, diagnostics)
    plot_revenue_decomposition(decomp)
    plot_channel_roi(coef_df)
    plot_budget_optimization(opt_df, df, total_budget)
    plot_budget_scenarios(scenario_df)
    plot_summary_dashboard(decomp, coef_df, opt_df, diagnostics, total_budget)
    print("All plots saved to /outputs/\n")
