"""
Sprint 2 - Baseline Model (Linear Regression)
===============================================
Q2(a): Feature Engineering - Lag Features & MinMaxScaler
Q2(b): Baseline Model - Linear Regression

Dataset: Labour Force Survey (Monthly), 2010-2026
Target Variable: u_rate (Unemployment Rate)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "lfs_month.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("SPRINT 2 - BASELINE MODEL")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"\nDataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string(index=False))

# ============================================================================
# Q2(a) - FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("Q2(a) - FEATURE ENGINEERING")
print("=" * 80)

# -------------------------------------------------------
# Step 1: Feature Creation - Lag Features
# -------------------------------------------------------
print("\n" + "-" * 60)
print("Step 1: Feature Creation - Lag Features")
print("-" * 60)

print("""
Rationale:
Lag features capture temporal autocorrelation in unemployment rate time series,
enabling the model to use past values for prediction. By including lags at 1, 3,
and 6 months, the model can learn short-term momentum (lag1), quarterly patterns
(lag3), and semi-annual trends (lag6) in unemployment dynamics.
""")

# Create lag features
df['u_rate_lag1'] = df['u_rate'].shift(1)
df['u_rate_lag3'] = df['u_rate'].shift(3)
df['u_rate_lag6'] = df['u_rate'].shift(6)

# Extract month and year from date
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

print("Created features:")
print("  - u_rate_lag1 : Unemployment rate lagged by 1 month")
print("  - u_rate_lag3 : Unemployment rate lagged by 3 months")
print("  - u_rate_lag6 : Unemployment rate lagged by 6 months")
print("  - month       : Month extracted from date (1-12)")
print("  - year        : Year extracted from date")

print(f"\nSample of lag features (first 8 rows):")
print(df[['date', 'u_rate', 'u_rate_lag1', 'u_rate_lag3', 'u_rate_lag6', 'month', 'year']].head(8).to_string(index=False))

# Drop NaN rows created by lagging
rows_before = len(df)
df = df.dropna().reset_index(drop=True)
rows_after = len(df)
print(f"\nRows before dropping NaN (from lags): {rows_before}")
print(f"Rows after dropping NaN:              {rows_after}")
print(f"Rows removed:                         {rows_before - rows_after}")

# -------------------------------------------------------
# Step 2: Scaling - MinMaxScaler
# -------------------------------------------------------
print("\n" + "-" * 60)
print("Step 2: Scaling - MinMaxScaler")
print("-" * 60)

print("""
Rationale:
Scaling normalizes feature ranges to [0,1], preventing features with larger
magnitudes from dominating the model and improving convergence. Labour force
variables (e.g., lf ~12,000) have much larger scales than rates (e.g., u_rate ~3-5),
so MinMaxScaler ensures all features contribute proportionally.
""")

# Define feature columns
feature_cols = ['lf', 'lf_employed', 'lf_outside', 'p_rate', 'ep_ratio',
                'u_rate_lag1', 'u_rate_lag3', 'u_rate_lag6', 'month', 'year']
target_col = 'u_rate'

# Show BEFORE scaling
print("BEFORE Scaling (first 5 rows of features):")
print(df[feature_cols].head().to_string(index=False))
print(f"\nFeature ranges BEFORE scaling:")
for col in feature_cols:
    print(f"  {col:15s} : min = {df[col].min():10.2f}, max = {df[col].max():10.2f}")

# Apply MinMaxScaler
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

# Show AFTER scaling
print(f"\nAFTER Scaling (first 5 rows of features):")
print(df_scaled[feature_cols].head().to_string(index=False))
print(f"\nFeature ranges AFTER scaling:")
for col in feature_cols:
    print(f"  {col:15s} : min = {df_scaled[col].min():10.4f}, max = {df_scaled[col].max():10.4f}")

# Save scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved to: {scaler_path}")

# ============================================================================
# Q2(b) - BASELINE MODEL: LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("Q2(b) - BASELINE MODEL: LINEAR REGRESSION")
print("=" * 80)

print("""
Model Selection Rationale:
Linear Regression was selected as the baseline model because it is simple,
interpretable, computationally efficient, and provides a benchmark for comparing
more complex models. It can reveal linear relationships between labour force
variables and unemployment rate. As a deterministic, closed-form solution, it
requires no hyperparameter tuning and serves as the minimum performance threshold
that improved models must surpass.
""")

# -------------------------------------------------------
# Prepare features and target
# -------------------------------------------------------
X = df_scaled[feature_cols].values
y = df_scaled[target_col].values

# Chronological train/test split (80/20) - NO random shuffle for time series
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = df['date'].values[split_idx:]
dates_all = df['date'].values

print(f"Total samples:    {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test samples:     {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Train period:     {pd.Timestamp(df['date'].iloc[0]).strftime('%Y-%m')} to {pd.Timestamp(df['date'].iloc[split_idx-1]).strftime('%Y-%m')}")
print(f"Test period:      {pd.Timestamp(df['date'].iloc[split_idx]).strftime('%Y-%m')} to {pd.Timestamp(df['date'].iloc[len(df)-1]).strftime('%Y-%m')}")
print(f"\nNote: Chronological split used (NOT random) to preserve time series order.")

# -------------------------------------------------------
# Train Linear Regression
# -------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# -------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------
print("\n" + "-" * 60)
print("Model Evaluation")
print("-" * 60)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n{'Metric':<20} {'Train':>12} {'Test':>12}")
print("-" * 44)
print(f"{'MAE':<20} {train_mae:>12.4f} {test_mae:>12.4f}")
print(f"{'RMSE':<20} {train_rmse:>12.4f} {test_rmse:>12.4f}")
print(f"{'R² Score':<20} {train_r2:>12.4f} {test_r2:>12.4f}")

# Print model coefficients
print(f"\nModel Coefficients:")
for fname, coef in zip(feature_cols, lr_model.coef_):
    print(f"  {fname:15s} : {coef:+.6f}")
print(f"  {'Intercept':15s} : {lr_model.intercept_:+.6f}")

# -------------------------------------------------------
# Plot: Actual vs Predicted on Test Set
# -------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

ax.plot(dates_test, y_test, 'b-o', markersize=4, linewidth=1.5,
        label='Actual', alpha=0.9)
ax.plot(dates_test, y_test_pred, 'r--s', markersize=4, linewidth=1.5,
        label='Predicted (Linear Regression)', alpha=0.9)

ax.set_title('Q2(b) Baseline Model — Actual vs Predicted Unemployment Rate (Test Set)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Unemployment Rate (u_rate)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')

# Add metrics annotation
metrics_text = f'MAE = {test_mae:.4f}\nRMSE = {test_rmse:.4f}\nR² = {test_r2:.4f}'
ax.text(0.98, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

plt.xticks(rotation=45)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "q2b_baseline_predictions.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to: {plot_path}")

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "baseline_model.joblib")
joblib.dump(lr_model, model_path)
print(f"Model saved to: {model_path}")

# Also save the processed dataframe and metadata for Sprint 3
metadata = {
    'feature_cols': feature_cols,
    'target_col': target_col,
    'split_idx': split_idx,
    'scaler': scaler
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "sprint2_metadata.joblib"))
# Save the processed (unscaled) dataframe for Sprint 3 reproducibility
df_original_processed = pd.read_csv(DATA_PATH)
df_original_processed['date'] = pd.to_datetime(df_original_processed['date'])
df_original_processed = df_original_processed.sort_values('date').reset_index(drop=True)
df_original_processed['u_rate_lag1'] = df_original_processed['u_rate'].shift(1)
df_original_processed['u_rate_lag3'] = df_original_processed['u_rate'].shift(3)
df_original_processed['u_rate_lag6'] = df_original_processed['u_rate'].shift(6)
df_original_processed['month'] = df_original_processed['date'].dt.month
df_original_processed['year'] = df_original_processed['date'].dt.year
df_original_processed = df_original_processed.dropna().reset_index(drop=True)
df_original_processed.to_csv(os.path.join(MODEL_DIR, "processed_data.csv"), index=False)

print(f"Metadata saved to: {os.path.join(MODEL_DIR, 'sprint2_metadata.joblib')}")
print(f"Processed data saved to: {os.path.join(MODEL_DIR, 'processed_data.csv')}")

print("\n" + "=" * 80)
print("SPRINT 2 COMPLETE")
print("=" * 80)
