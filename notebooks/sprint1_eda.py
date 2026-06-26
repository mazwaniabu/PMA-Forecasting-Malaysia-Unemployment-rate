"""
Sprint 1 - Exploratory Data Analysis (EDA)
===========================================
Dataset : Malaysia Monthly Labour Force Statistics
Source  : Department of Statistics Malaysia (DOSM) via OpenDOSM
Target  : u_rate (monthly unemployment rate %)
"""

import os
import sys
import warnings
import numpy as np

# Force UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# ── Matplotlib / Seaborn setup ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import seaborn as sns

# Try the modern style name first; fall back gracefully
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("default")
        print("[INFO] Using default matplotlib style (seaborn style not found).")

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH     = os.path.join(BASE_DIR, "data", "raw", "lfs_month.csv")
CLEANED_DIR  = os.path.join(BASE_DIR, "data", "cleaned")
CLEANED_PATH = os.path.join(CLEANED_DIR, "lfs_month_cleaned.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)

# ============================================================================
# Q1(a) — DATASET DESCRIPTION
# ============================================================================
print("=" * 80)
print("Q1(a) — DATASET DESCRIPTION")
print("=" * 80)

print(f"""
Dataset Source URL : https://open.dosm.gov.my/data-catalogue/lfs_month

Organizational Problem:
  Forecasting Malaysia's monthly unemployment rate to support workforce
  planning by the Ministry of Human Resources. Accurate short-term forecasts
  enable proactive labour-market interventions, budget allocation for
  reskilling programmes, and evidence-based policy adjustments.

Key Stakeholders:
  1. Ministry of Human Resources (MOHR)
     — primary consumer for workforce planning decisions
  2. Department of Statistics Malaysia (DOSM)
     — data custodian and publisher of the Labour Force Survey
  3. Economic Planning Unit (EPU), Prime Minister's Department
     — macroeconomic policy coordination
  4. Employers' associations (e.g., MEF)
     — labour-demand forecasting and hiring strategies

Dataset Shape : {df.shape[0]} records  ×  {df.shape[1]} variables
Target Variable : u_rate (unemployment rate, %)
Prediction Type : Time-series regression forecasting
  — we predict a continuous numeric value (u_rate) that evolves over time.
""")

print("-" * 80)
print("First 10 Records (df.head(10)):")
print("-" * 80)
print(df.head(10).to_string(index=False))

print("\n" + "-" * 80)
print("Dataset Info (df.info()):")
print("-" * 80)
import io
buf = io.StringIO()
df.info(buf=buf)
print(buf.getvalue())

print("-" * 80)
print("Descriptive Statistics (df.describe()):")
print("-" * 80)
print(df.describe().round(2).to_string())

# ============================================================================
# Q1(b) — EXPLORATORY DATA ANALYSIS  (3 visualisations)
# ============================================================================
print("\n" + "=" * 80)
print("Q1(b) — EXPLORATORY DATA ANALYSIS  (3 Visualisations)")
print("=" * 80)

# Colour palette
PAL = sns.color_palette("muted")

# ── Vis 1: Distribution of u_rate (Histogram + Boxplot) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram with KDE
sns.histplot(df["u_rate"], bins=20, kde=True, color=PAL[0], edgecolor="white",
             ax=axes[0])
axes[0].axvline(df["u_rate"].mean(), color="red", ls="--", lw=1.2,
                label=f'Mean = {df["u_rate"].mean():.2f}%')
axes[0].axvline(df["u_rate"].median(), color="green", ls="--", lw=1.2,
                label=f'Median = {df["u_rate"].median():.2f}%')
axes[0].set_title("Distribution of Unemployment Rate", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Unemployment Rate (%)")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=9)

# Boxplot
sns.boxplot(y=df["u_rate"], color=PAL[1], width=0.4, ax=axes[1])
axes[1].set_title("Boxplot of Unemployment Rate", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Unemployment Rate (%)")

fig.suptitle("Q1(b) — Visualisation 1: Distribution Analysis of u_rate",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "q1b_distribution.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\n[SAVED] {os.path.join(OUTPUT_DIR, 'q1b_distribution.png')}")

# Observation
u_mean   = df["u_rate"].mean()
u_median = df["u_rate"].median()
u_std    = df["u_rate"].std()
u_min    = df["u_rate"].min()
u_max    = df["u_rate"].max()
skew_val = df["u_rate"].skew()

print(f"""
Observation (Vis 1 — Distribution):
  • The unemployment rate ranges from {u_min:.1f}% to {u_max:.1f}%,
    with a mean of {u_mean:.2f}% and median of {u_median:.2f}%.
  • Standard deviation is {u_std:.2f}%, indicating moderate dispersion.
  • Skewness = {skew_val:.2f} → the distribution is {'positively (right-)' if skew_val > 0 else 'negatively (left-)'}skewed,
    {'largely driven by the COVID-19 unemployment spike in 2020.' if u_max > 4.5 else 'suggesting occasional high-unemployment periods.'}
  • The boxplot shows {'outliers above the upper whisker, corresponding to pandemic months.' if u_max > 4.5 else 'no extreme outliers.'}
""")

# ── Vis 2: Relationship Analysis (Scatter + Correlation Heatmap) ───────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr().round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Scatter plot: u_rate vs p_rate
sns.regplot(x="p_rate", y="u_rate", data=df, ax=axes[0],
            scatter_kws={"alpha": 0.55, "s": 30, "color": PAL[0]},
            line_kws={"color": "red", "lw": 1.3})
r_val = df["p_rate"].corr(df["u_rate"])
axes[0].set_title(f"u_rate vs p_rate  (r = {r_val:.2f})", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Participation Rate (%)")
axes[0].set_ylabel("Unemployment Rate (%)")

# Correlation heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            mask=mask, linewidths=0.5, ax=axes[1],
            annot_kws={"fontsize": 8})
axes[1].set_title("Correlation Heatmap (numeric columns)", fontsize=12, fontweight="bold")

fig.suptitle("Q1(b) — Visualisation 2: Relationship Analysis",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "q1b_relationship.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {os.path.join(OUTPUT_DIR, 'q1b_relationship.png')}")

print(f"""
Observation (Vis 2 — Relationships):
  • u_rate vs p_rate: Pearson r = {r_val:.2f}
    {'→ weak/negligible linear relationship.' if abs(r_val) < 0.3 else '→ moderate inverse relationship: higher participation tends to coincide with lower unemployment.' if r_val < -0.3 else '→ moderate positive relationship.'}
  • Correlation heatmap highlights:
    - lf_employed and lf have the strongest positive correlation (near 1.0)
      because employed persons dominate the labour force.
    - u_rate correlates positively with lf_unemployed (by construction).
    - ep_ratio (employment-population ratio) correlates negatively with u_rate,
      confirming that higher employment shares reduce unemployment.
""")

# Print full correlation matrix
print("Full Correlation Matrix:")
print(corr_matrix.to_string())
print()

# ── Vis 3: Categorical / Yearly Trend Analysis ────────────────────────────
df["_year"] = pd.to_datetime(df["date"]).dt.year
yearly = df.groupby("_year")["u_rate"].mean().reset_index()
yearly.columns = ["Year", "Mean_u_rate"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(yearly["Year"].astype(str), yearly["Mean_u_rate"],
              color=PAL[2], edgecolor="white", width=0.7)
# Highlight COVID year
for bar, yr in zip(bars, yearly["Year"]):
    if yr == 2020:
        bar.set_color(PAL[3])
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

ax.set_title("Mean Unemployment Rate by Year", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Unemployment Rate (%)")
ax.set_ylim(0, yearly["Mean_u_rate"].max() * 1.25)
plt.xticks(rotation=45, ha="right")

# Annotate bars
for bar, val in zip(bars, yearly["Mean_u_rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.suptitle("Q1(b) — Visualisation 3: Categorical (Yearly) Analysis",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "q1b_categorical.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {os.path.join(OUTPUT_DIR, 'q1b_categorical.png')}")

# Identify peak and trough years
peak_year = yearly.loc[yearly["Mean_u_rate"].idxmax()]
low_year  = yearly.loc[yearly["Mean_u_rate"].idxmin()]

print(f"""
Observation (Vis 3 — Yearly Trends):
  • The highest mean unemployment rate was in {int(peak_year['Year'])}
    at {peak_year['Mean_u_rate']:.2f}%, coinciding with the COVID-19 pandemic
    and associated Movement Control Orders (MCO).
  • The lowest mean unemployment rate was in {int(low_year['Year'])}
    at {low_year['Mean_u_rate']:.2f}%, reflecting a robust labour market.
  • Pre-pandemic (2010–2019), unemployment was relatively stable around
    3.0–3.4%, indicating structural unemployment rather than cyclical swings.
  • Post-2020, a clear recovery trend is visible as the economy reopened
    and labour-market conditions normalised.
""")

# Yearly table
print("Mean Unemployment Rate by Year:")
print(yearly.to_string(index=False))

# Drop temporary column
df.drop(columns=["_year"], inplace=True)

# ============================================================================
# Q1(c) — DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("Q1(c) — DATA QUALITY CHECKS")
print("=" * 80)

issues_log = []  # collect (Issue, Column, Details, Impact)

# ── Check 1: Missing Values ────────────────────────────────────────────────
print("\n" + "-" * 60)
print("Check 1 — Missing Values")
print("-" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Missing_Count": missing, "Missing_%": missing_pct})
print(miss_df.to_string())

total_missing = missing.sum()
if total_missing == 0:
    print("\n✅ No missing values detected in any column.")
else:
    cols_with_missing = missing[missing > 0].index.tolist()
    print(f"\n⚠️  Missing values found in: {cols_with_missing}")
    for col in cols_with_missing:
        issues_log.append((
            "Missing Values", col,
            f"{missing[col]} missing ({missing_pct[col]}%)",
            "Biased model training; loss of temporal continuity if rows dropped."
        ))

# ── Check 2: Duplicate Records ─────────────────────────────────────────────
print("\n" + "-" * 60)
print("Check 2 — Duplicate Records")
print("-" * 60)
n_dupes = df.duplicated().sum()
print(f"Total duplicate rows: {n_dupes}")
if n_dupes == 0:
    print("✅ No duplicate records found.")
else:
    print("⚠️  Duplicate rows detected:")
    print(df[df.duplicated(keep=False)].to_string())
    issues_log.append((
        "Duplicate Records", "All",
        f"{n_dupes} exact duplicate row(s)",
        "Inflated training set; over-representation of duplicated months."
    ))

# ── Check 3: Data Type Validation ──────────────────────────────────────────
print("\n" + "-" * 60)
print("Check 3 — Data Type Validation")
print("-" * 60)
print(df.dtypes.to_string())

if df["date"].dtype == "object":
    print("\n⚠️  'date' column is of type 'object' — should be datetime64.")
    issues_log.append((
        "Incorrect Data Type", "date",
        f"Current dtype = {df['date'].dtype}; expected datetime64[ns]",
        "Cannot perform time-based indexing, resampling, or feature "
        "engineering (lags, rolling windows) without datetime conversion."
    ))
else:
    print("✅ 'date' column is already datetime64.")

# Check numeric columns are truly numeric
for col in ["lf", "lf_employed", "lf_unemployed", "lf_outside",
             "p_rate", "ep_ratio", "u_rate"]:
    if not np.issubdtype(df[col].dtype, np.number):
        print(f"⚠️  '{col}' is not numeric (dtype = {df[col].dtype})")
        issues_log.append((
            "Incorrect Data Type", col,
            f"Expected numeric, got {df[col].dtype}",
            "Non-numeric types will cause errors in modelling."
        ))
    else:
        print(f"✅ '{col}' is numeric ({df[col].dtype})")

# ── Check 4: Outlier Detection (IQR method on u_rate) ──────────────────────
print("\n" + "-" * 60)
print("Check 4 — Outlier Detection (IQR Method on u_rate)")
print("-" * 60)
Q1 = df["u_rate"].quantile(0.25)
Q3 = df["u_rate"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"  Q1 = {Q1:.2f}%    Q3 = {Q3:.2f}%    IQR = {IQR:.2f}%")
print(f"  Lower Bound = {lower_bound:.2f}%    Upper Bound = {upper_bound:.2f}%")

outliers = df[(df["u_rate"] < lower_bound) | (df["u_rate"] > upper_bound)]
n_outliers = len(outliers)
print(f"\n  Outliers detected: {n_outliers}")

if n_outliers > 0:
    print("\n  Outlier Records:")
    print(outliers[["date", "u_rate"]].to_string(index=False))
    issues_log.append((
        "Outliers (IQR)", "u_rate",
        f"{n_outliers} values outside [{lower_bound:.2f}, {upper_bound:.2f}]",
        "Outliers (likely COVID-19 months) can skew model training. "
        "They are real events, so capping rather than removal is recommended."
    ))
else:
    print("  ✅ No outliers detected using IQR method.")

# ── Summary Table of Issues ────────────────────────────────────────────────
print("\n" + "-" * 60)
print("DATA QUALITY — SUMMARY OF ISSUES")
print("-" * 60)
if issues_log:
    issues_df = pd.DataFrame(issues_log,
                             columns=["Issue", "Column", "Details",
                                      "Potential Impact on Model"])
    issues_df.index = range(1, len(issues_df) + 1)
    issues_df.index.name = "#"
    # Print each issue clearly
    for idx, row in issues_df.iterrows():
        print(f"\n  Issue #{idx}")
        print(f"    Type    : {row['Issue']}")
        print(f"    Column  : {row['Column']}")
        print(f"    Details : {row['Details']}")
        print(f"    Impact  : {row['Potential Impact on Model']}")
else:
    print("  ✅ No data quality issues detected.")

# ============================================================================
# Q1(c) — DATA CLEANING
# ============================================================================
print("\n" + "=" * 80)
print("Q1(c) — DATA CLEANING")
print("=" * 80)

df_clean = df.copy()

# ── Fix 1: Convert 'date' to datetime ──────────────────────────────────────
print("\n[1] Converting 'date' column to datetime64 …")
df_clean["date"] = pd.to_datetime(df_clean["date"])
print(f"    dtype after conversion: {df_clean['date'].dtype}")
print(f"    Date range: {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")

# ── Fix 2: Sort by date (ensure chronological order) ──────────────────────
df_clean.sort_values("date", inplace=True)
df_clean.reset_index(drop=True, inplace=True)
print("[2] Sorted dataframe chronologically by date.")

# ── Fix 3: Remove duplicates if any ───────────────────────────────────────
n_before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
n_after = len(df_clean)
print(f"[3] Duplicate removal: {n_before} → {n_after} rows "
      f"({n_before - n_after} dropped).")

# ── Fix 4: Handle missing values (forward-fill if any) ────────────────────
n_missing_before = df_clean.isnull().sum().sum()
if n_missing_before > 0:
    df_clean.fillna(method="ffill", inplace=True)
    df_clean.fillna(method="bfill", inplace=True)  # cover leading NaNs
    print(f"[4] Filled {n_missing_before} missing value(s) via forward/backward fill.")
else:
    print("[4] No missing values to fill.")

# ── Fix 5: Flag outliers (add column, do NOT remove — real events) ────────
df_clean["u_rate_outlier"] = (
    (df_clean["u_rate"] < lower_bound) | (df_clean["u_rate"] > upper_bound)
)
n_flagged = df_clean["u_rate_outlier"].sum()
print(f"[5] Flagged {n_flagged} u_rate outlier(s) in new column "
      f"'u_rate_outlier' (retained, not removed — real events).")

# ── Save cleaned data ─────────────────────────────────────────────────────
df_clean.to_csv(CLEANED_PATH, index=False)
print(f"\n✅ Cleaned data saved to: {CLEANED_PATH}")
print(f"   Shape: {df_clean.shape}")
print(f"   Columns: {list(df_clean.columns)}")

# Quick sanity check
print("\n--- Cleaned Data Preview ---")
print(df_clean.head(5).to_string(index=False))
print(f"\nData types after cleaning:")
print(df_clean.dtypes.to_string())

# ============================================================================
# Q1(d) — SPRINT 1 BACKLOG
# ============================================================================
print("\n" + "=" * 80)
print("Q1(d) — SPRINT 1 PRODUCT BACKLOG")
print("=" * 80)

backlog = [
    {
        "Priority": "P1 (Must)",
        "Backlog Item": "Convert 'date' column from object to datetime64 and set as index",
        "Reason": "Identified in Q1(c) — incorrect dtype prevents time-based feature "
                  "engineering (lags, rolling stats, seasonal decomposition).",
        "Expected Deliverable": "Cleaned DataFrame with DatetimeIndex; "
                                "validated via df.index.dtype == datetime64[ns]"
    },
    {
        "Priority": "P1 (Must)",
        "Backlog Item": "Flag / treat u_rate outliers (COVID-19 spike months)",
        "Reason": "IQR check in Q1(c) detected outlier months. Must decide strategy "
                  "(winsorisation, indicator variable, or separate regime model).",
        "Expected Deliverable": "Boolean outlier flag column; documented decision "
                                "on treatment approach"
    },
    {
        "Priority": "P1 (Must)",
        "Backlog Item": "Engineer time-series features (lag, rolling mean, month dummies)",
        "Reason": "Raw columns lack temporal context; lag-1 and rolling-3/6/12 month "
                  "features capture autocorrelation & trend essential for forecasting.",
        "Expected Deliverable": "Feature matrix CSV with ≥ 5 new engineered columns"
    },
    {
        "Priority": "P2 (Should)",
        "Backlog Item": "Perform stationarity test (ADF) and differencing if needed",
        "Reason": "Time-series models (ARIMA, LSTM) assume or benefit from stationarity; "
                  "ADF test result determines if differencing is required.",
        "Expected Deliverable": "ADF test p-value; differenced series if non-stationary"
    },
    {
        "Priority": "P2 (Should)",
        "Backlog Item": "Train/validation/test split (chronological 70/15/15)",
        "Reason": "Time-series data must not be randomly shuffled; a temporal split avoids "
                  "data leakage and mirrors real forecasting conditions.",
        "Expected Deliverable": "Three disjoint DataFrames; date boundaries documented"
    },
    {
        "Priority": "P3 (Could)",
        "Backlog Item": "Baseline model — Naïve / Seasonal Naïve forecast",
        "Reason": "A simple baseline is needed to benchmark more complex models. "
                  "Reports RMSE and MAE on the test set.",
        "Expected Deliverable": "Baseline RMSE & MAE; comparison notebook"
    },
]

# Print as formatted table
header = f"{'Priority':<14} {'Backlog Item':<58} {'Reason':<65} {'Expected Deliverable'}"
print(f"\n{header}")
print("-" * len(header))
for item in backlog:
    # Wrap long text for console readability
    print(f"{item['Priority']:<14} {item['Backlog Item']:<58} "
          f"{item['Reason'][:63]:<65} {item['Expected Deliverable'][:60]}")

# Also print detailed version
print("\n\n--- Detailed Sprint 1 Backlog ---\n")
for i, item in enumerate(backlog, 1):
    print(f"  [{i}] {item['Priority']}")
    print(f"      Backlog Item       : {item['Backlog Item']}")
    print(f"      Reason for Priority: {item['Reason']}")
    print(f"      Expected Deliverable: {item['Expected Deliverable']}")
    print()

# ============================================================================
print("\n" + "=" * 80)
print("SPRINT 1 EDA SCRIPT — COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"""
Output files generated:
  1. {os.path.join(OUTPUT_DIR, 'q1b_distribution.png')}
  2. {os.path.join(OUTPUT_DIR, 'q1b_relationship.png')}
  3. {os.path.join(OUTPUT_DIR, 'q1b_categorical.png')}
  4. {CLEANED_PATH}
""")
