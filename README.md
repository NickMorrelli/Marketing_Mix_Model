# 📊 Marketing Mix Model (MMM)

An end-to-end Marketing Mix Model built in Python using **OLS regression** and **budget optimization**. The model measures the revenue contribution of each marketing channel (TV, Digital, Email, Social), estimates ROI per $1 spent, and recommends an optimal budget allocation to maximize revenue.

---

## 🎯 Business Questions

> - *Which marketing channels are driving the most revenue?*
> - *What is the ROI of each channel?*
> - *How should we reallocate our budget to maximize revenue?*

---

## 📁 Project Structure

```
marketing_mix_model/
├── outputs/                     # Generated charts saved here
├── src/
│   ├── data_prep.py             # Synthetic data generation + adstock transformation
│   ├── model.py                 # OLS regression, diagnostics, revenue decomposition
│   ├── optimizer.py             # Budget optimization via scipy
│   └── visualizations.py       # All charts and summary dashboard
├── main.py                      # Run the full pipeline
├── requirements.txt
└── README.md
```

---

## 🧪 Methodology

### 1. Data
Synthetic weekly marketing data (3 years / 156 weeks) with realistic spend patterns:
- **TV**: bursty campaign flights (~40% of weeks)
- **Digital**: consistent always-on spend
- **Email**: low cost, steady spend
- **Social**: growing investment trend over time

### 2. Adstock Transformation
Advertising has a carryover effect — a TV ad seen this week still influences purchases next week. We model this with **geometric adstock**:

```
adstock[t] = spend[t] + decay_rate × adstock[t-1]
```

| Channel | Decay Rate | Interpretation |
|---|---|---|
| TV | 0.6 | High carryover — brand awareness lingers |
| Digital | 0.3 | Moderate — intent fades quickly |
| Email | 0.1 | Low — effect is mostly immediate |
| Social | 0.4 | Moderate carryover |

### 3. OLS Regression Model

```
revenue = baseline
        + β_tv       × tv_adstock
        + β_digital  × digital_adstock
        + β_email    × email_adstock
        + β_social   × social_adstock
        + β_promo    × promo
        + ε
```

Each coefficient β represents revenue generated per unit of adstocked spend — this is the channel **ROI**.

### 4. Model Diagnostics

| Metric | Description |
|---|---|
| R² | % of revenue variance explained |
| Adjusted R² | R² penalized for model complexity |
| MAPE | Mean Absolute Percentage Error |
| Durbin-Watson | Checks for autocorrelation in residuals |
| p-values | Statistical significance of each channel |

### 5. Budget Optimization
Uses **scipy.optimize.minimize** (SLSQP) with:
- Equality constraint: total spend = budget
- Diminishing returns modeled via square-root transformation
- Minimum/maximum spend bounds per channel

---

## 📈 Output Charts

| File | Description |
|---|---|
| `01_spend_over_time.png` | Weekly spend by channel |
| `02_model_fit.png` | Actual vs predicted revenue + residuals |
| `03_revenue_decomposition.png` | Stacked channel contributions over time |
| `04_channel_roi.png` | ROI per $1 spent by channel |
| `05_budget_optimization.png` | Current vs optimal budget allocation |
| `06_budget_scenarios.png` | Revenue curve across budget levels |
| `07_summary_dashboard.png` | Full summary dashboard |

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/marketing_mix_model.git
cd marketing_mix_model
pip install -r requirements.txt
python main.py
```

No data download required — data is generated synthetically by the pipeline.

---

## 🛠 Tech Stack

- **Python 3.14+**
- `statsmodels` — OLS regression with full statistical diagnostics
- `scipy` — budget optimization (SLSQP)
- `pandas` / `numpy` — data wrangling
- `matplotlib` — visualizations

---

## 💡 Key Concepts Demonstrated

- Adstock / carryover effect modeling
- OLS regression with statistical inference (p-values, confidence intervals)
- Revenue decomposition by marketing channel
- Constrained budget optimization with diminishing returns
- Marketing ROI measurement and reporting

---

## 👤 Author

Built as part of a data science portfolio project.  
Background: 15+ years in Marketing Analytics | SQL | Python | Statistical Modeling
