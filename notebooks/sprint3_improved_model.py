"""
Sprint 3 - Improved Model (Random Forest with GridSearchCV)
=============================================================
Q2(c): Improved Model - Random Forest Regressor + GridSearchCV
Q2(d): Model Comparison - Sprint 2 vs Sprint 3

Dataset: Labour Force Survey (Monthly), 2010-2026
Target Variable: u_rate (Unemployment Rate)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
# LOAD AND PREPARE DATA (same pipeline as Sprint 2)
# ============================================================================
print("=" * 80)
print("SPRINT 3 - IMPROVED MODEL")
print("=" * 80)

# Load raw data and apply same feature engineering as Sprint 2
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Create lag features and date components (same as Sprint 2)
df['u_rate_lag1'] = df['u_rate'].shift(1)
df['u_rate_lag3'] = df['u_rate'].shift(3)
df['u_rate_lag6'] = df['u_rate'].shift(6)
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Drop NaN from lags
df = df.dropna().reset_index(drop=True)

print(f"\nDataset loaded and processed: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

# Define features and target
feature_cols = ['lf', 'lf_employed', 'lf_outside', 'p_rate', 'ep_ratio',
                'u_rate_lag1', 'u_rate_lag3', 'u_rate_lag6', 'month', 'year']
target_col = 'u_rate'

# Chronological train/test split (80/20)
split_idx = int(len(df) * 0.8)

print(f"\nTraining samples: {split_idx} ({split_idx/len(df)*100:.1f}%)")
print(f"Test samples:     {len(df)-split_idx} ({(len(df)-split_idx)/len(df)*100:.1f}%)")
print(f"Train period:     {df['date'].iloc[0].strftime('%Y-%m')} to {df['date'].iloc[split_idx-1].strftime('%Y-%m')}")
print(f"Test period:      {df['date'].iloc[split_idx].strftime('%Y-%m')} to {df['date'].iloc[len(df)-1].strftime('%Y-%m')}")

dates_test = df['date'].values[split_idx:]

# -------------------------------------------------------
# Prepare SCALED data (for baseline LR comparison - same as Sprint 2)
# -------------------------------------------------------
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

X_scaled = df_scaled[feature_cols].values
y_scaled = df_scaled[target_col].values

X_train_scaled = X_scaled[:split_idx]
X_test_scaled = X_scaled[split_idx:]
y_train_scaled = y_scaled[:split_idx]
y_test_scaled = y_scaled[split_idx:]

# -------------------------------------------------------
# Prepare UNSCALED data (for Random Forest - trees are scale-invariant)
# -------------------------------------------------------
# Note: Random Forest (tree-based) does NOT need feature scaling.
# Using raw features avoids extrapolation issues where scaled test features
# fall outside the [0,1] range learned during training.
X_raw = df[feature_cols].values
y_raw = df[target_col].values

X_train_raw = X_raw[:split_idx]
X_test_raw = X_raw[split_idx:]
y_train_raw = y_raw[:split_idx]
y_test_raw = y_raw[split_idx:]

print(f"\nNote: Random Forest uses UNSCALED features (tree-based models are")
print(f"      scale-invariant). Baseline LR uses scaled features for comparison.")

# ============================================================================
# Q2(c) - IMPROVED MODEL: RANDOM FOREST WITH GridSearchCV
# ============================================================================
print("\n" + "=" * 80)
print("Q2(c) - IMPROVED MODEL: RANDOM FOREST WITH GridSearchCV")
print("=" * 80)

print("""
Improvement Description:
The improved model uses Random Forest Regressor, a non-linear ensemble method
that captures complex interactions between features. GridSearchCV was used for
hyperparameter tuning with 5-fold cross-validation to find the optimal
configuration, reducing overfitting and improving generalization.

Key advantages over Linear Regression:
  1. Captures non-linear relationships between features and u_rate
  2. Handles feature interactions automatically (e.g., lf x ep_ratio)
  3. Robust to outliers and noisy data
  4. Built-in feature importance for interpretability
  5. Ensemble averaging reduces variance and overfitting
  6. Scale-invariant — no feature scaling required
""")

# -------------------------------------------------------
# GridSearchCV Setup
# -------------------------------------------------------
print("-" * 60)
print("Hyperparameter Tuning with GridSearchCV")
print("-" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])
print(f"\nParameter Grid:")
print(f"  n_estimators    : {param_grid['n_estimators']}")
print(f"  max_depth       : {param_grid['max_depth']}")
print(f"  min_samples_split: {param_grid['min_samples_split']}")
print(f"\nTotal combinations: {total_combinations}")
print(f"Cross-validation: 5-fold (TimeSeriesSplit)")
print(f"Scoring metric: neg_mean_squared_error")
print(f"Total fits: {total_combinations * 5}")
print(f"\nRunning GridSearchCV... (this may take a moment)")

# Use TimeSeriesSplit for proper time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
grid_search.fit(X_train_raw, y_train_raw)

# -------------------------------------------------------
# Best Parameters
# -------------------------------------------------------
print(f"\nGridSearchCV Complete!")
print(f"\nBest Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param:20s} : {value}")
print(f"\nBest CV Score (neg_MSE): {grid_search.best_score_:.6f}")
print(f"Best CV Score (RMSE):    {np.sqrt(-grid_search.best_score_):.6f}")

# Get best model
rf_best = grid_search.best_estimator_

# -------------------------------------------------------
# Predictions (on raw/unscaled data)
# -------------------------------------------------------
y_train_pred_rf = rf_best.predict(X_train_raw)
y_test_pred_rf = rf_best.predict(X_test_raw)

# -------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------
print("\n" + "-" * 60)
print("Improved Model Evaluation")
print("-" * 60)

rf_train_mae = mean_absolute_error(y_train_raw, y_train_pred_rf)
rf_train_rmse = np.sqrt(mean_squared_error(y_train_raw, y_train_pred_rf))
rf_train_r2 = r2_score(y_train_raw, y_train_pred_rf)

rf_test_mae = mean_absolute_error(y_test_raw, y_test_pred_rf)
rf_test_rmse = np.sqrt(mean_squared_error(y_test_raw, y_test_pred_rf))
rf_test_r2 = r2_score(y_test_raw, y_test_pred_rf)

print(f"\n{'Metric':<20} {'Train':>12} {'Test':>12}")
print("-" * 44)
print(f"{'MAE':<20} {rf_train_mae:>12.4f} {rf_test_mae:>12.4f}")
print(f"{'RMSE':<20} {rf_train_rmse:>12.4f} {rf_test_rmse:>12.4f}")
print(f"{'R² Score':<20} {rf_train_r2:>12.4f} {rf_test_r2:>12.4f}")

# -------------------------------------------------------
# Plot: Actual vs Predicted (Improved Model)
# -------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

ax.plot(dates_test, y_test_raw, 'b-o', markersize=4, linewidth=1.5,
        label='Actual', alpha=0.9)
ax.plot(dates_test, y_test_pred_rf, 'g--^', markersize=4, linewidth=1.5,
        label='Predicted (Random Forest)', alpha=0.9)

ax.set_title('Q2(c) Improved Model — Actual vs Predicted Unemployment Rate (Test Set)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Unemployment Rate (u_rate)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')

metrics_text = f'MAE = {rf_test_mae:.4f}\nRMSE = {rf_test_rmse:.4f}\nR² = {rf_test_r2:.4f}'
ax.text(0.98, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

plt.xticks(rotation=45)
plt.tight_layout()

plot_path_pred = os.path.join(OUTPUT_DIR, "q2c_improved_predictions.png")
plt.savefig(plot_path_pred, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPrediction plot saved to: {plot_path_pred}")

# -------------------------------------------------------
# Plot: Feature Importance
# -------------------------------------------------------
importances = rf_best.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
               color=colors, edgecolor='white', linewidth=0.5)

ax.set_title('Q2(c) Random Forest — Feature Importance',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)

# Add value labels on bars
for bar, val in zip(bars, importance_df['Importance']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()

plot_path_imp = os.path.join(OUTPUT_DIR, "q2c_feature_importance.png")
plt.savefig(plot_path_imp, dpi=150, bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to: {plot_path_imp}")

# Print feature importance ranking
print(f"\nFeature Importance Ranking:")
for i, (_, row) in enumerate(importance_df.iloc[::-1].iterrows(), 1):
    print(f"  {i:2d}. {row['Feature']:15s} : {row['Importance']:.4f}")

# -------------------------------------------------------
# Save Improved Model
# -------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "improved_model.joblib")
joblib.dump(rf_best, model_path)
print(f"\nImproved model saved to: {model_path}")

# ============================================================================
# Q2(d) - MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("Q2(d) - MODEL COMPARISON: SPRINT 2 vs SPRINT 3")
print("=" * 80)

# -------------------------------------------------------
# Load Sprint 2 baseline model and evaluate on SCALED test set
# (Baseline was trained on scaled data)
# -------------------------------------------------------
lr_model = joblib.load(os.path.join(MODEL_DIR, "baseline_model.joblib"))

# Baseline predictions (on scaled data)
y_train_pred_lr_scaled = lr_model.predict(X_train_scaled)
y_test_pred_lr_scaled = lr_model.predict(X_test_scaled)

# Baseline metrics (on raw u_rate since target was never scaled)
lr_train_mae = mean_absolute_error(y_train_raw, y_train_pred_lr_scaled)
lr_train_rmse = np.sqrt(mean_squared_error(y_train_raw, y_train_pred_lr_scaled))
lr_train_r2 = r2_score(y_train_raw, y_train_pred_lr_scaled)

lr_test_mae = mean_absolute_error(y_test_raw, y_test_pred_lr_scaled)
lr_test_rmse = np.sqrt(mean_squared_error(y_test_raw, y_test_pred_lr_scaled))
lr_test_r2 = r2_score(y_test_raw, y_test_pred_lr_scaled)

# -------------------------------------------------------
# Q2(d)(i) - Comparison Table
# -------------------------------------------------------
print("\n" + "-" * 60)
print("Q2(d)(i) - Model Comparison Table")
print("-" * 60)

print(f"""
+-------------------------+-----------------------------+-----------------------------+
|         Aspect          |    Sprint 2 (Baseline)      |    Sprint 3 (Improved)      |
+-------------------------+-----------------------------+-----------------------------+
| Model Used              | Linear Regression           | Random Forest Regressor     |
+-------------------------+-----------------------------+-----------------------------+
| Evaluation Metrics      | MAE, RMSE, R²               | MAE, RMSE, R²               |
+-------------------------+-----------------------------+-----------------------------+
| MAE  (Train / Test)     | {lr_train_mae:.4f} / {lr_test_mae:.4f}         | {rf_train_mae:.4f} / {rf_test_mae:.4f}         |
| RMSE (Train / Test)     | {lr_train_rmse:.4f} / {lr_test_rmse:.4f}         | {rf_train_rmse:.4f} / {rf_test_rmse:.4f}         |
| R²   (Train / Test)     | {lr_train_r2:.4f} / {lr_test_r2:.4f}         | {rf_train_r2:.4f} / {rf_test_r2:.4f}         |
+-------------------------+-----------------------------+-----------------------------+
| Improvement Introduced  | Baseline (no tuning)        | Hyperparameter tuning +     |
|                         |                             | non-linear ensemble method  |
+-------------------------+-----------------------------+-----------------------------+
""")

# Compute improvement percentages
mae_improvement = ((lr_test_mae - rf_test_mae) / lr_test_mae) * 100
rmse_improvement = ((lr_test_rmse - rf_test_rmse) / lr_test_rmse) * 100
r2_improvement = rf_test_r2 - lr_test_r2

print(f"Improvement Summary (Test Set):")
print(f"  MAE  reduction: {mae_improvement:+.2f}%")
print(f"  RMSE reduction: {rmse_improvement:+.2f}%")
print(f"  R²   change:    {r2_improvement:+.4f}")

# -------------------------------------------------------
# Q2(d)(ii) - Key Findings from Sprint 2
# -------------------------------------------------------
print("\n" + "-" * 60)
print("Q2(d)(ii) - Key Findings from Sprint 2 (Baseline)")
print("-" * 60)

print(f"""
1. LINEAR RELATIONSHIPS IDENTIFIED:
   The Linear Regression baseline revealed that labour force variables have
   measurable linear relationships with the unemployment rate. The lag features
   (especially u_rate_lag1) were strong predictors, confirming temporal
   autocorrelation in unemployment data.

2. BASELINE PERFORMANCE ESTABLISHED:
   The baseline achieved R² = {lr_test_r2:.4f} on the test set, indicating that
   linear combinations of features explain {lr_test_r2*100:.1f}% of variance in u_rate.
   MAE of {lr_test_mae:.4f} and RMSE of {lr_test_rmse:.4f} set the benchmark to beat.

3. FEATURE ENGINEERING VALUE:
   Lag features and temporal components (month, year) proved essential for
   capturing time-dependent patterns. Without them, the model would miss
   the autoregressive nature of unemployment dynamics.

4. LIMITATIONS OBSERVED:
   Linear Regression assumes a linear relationship between features and target,
   which may not capture complex interactions (e.g., non-linear effects of
   labour force participation on unemployment during economic shocks like COVID-19).
   The model also cannot capture interaction effects between features.
""")

# -------------------------------------------------------
# Q2(d)(iii) - How Sprint 3 Improved the Model
# -------------------------------------------------------
print("-" * 60)
print("Q2(d)(iii) - How Sprint 3 Improved the Model")
print("-" * 60)

print(f"""
1. NON-LINEAR MODELLING:
   Random Forest Regressor replaced Linear Regression, enabling the model to
   capture non-linear relationships and complex interactions between features.
   This is critical for modelling unemployment dynamics during structural breaks
   (e.g., COVID-19 pandemic in 2020-2021).

2. HYPERPARAMETER OPTIMISATION:
   GridSearchCV with 5-fold TimeSeriesSplit cross-validation systematically
   searched {total_combinations} parameter combinations to find the optimal configuration:
   - Best n_estimators:     {grid_search.best_params_['n_estimators']}
   - Best max_depth:        {grid_search.best_params_['max_depth']}
   - Best min_samples_split: {grid_search.best_params_['min_samples_split']}
   TimeSeriesSplit preserves temporal ordering in each fold, preventing data
   leakage and providing more realistic performance estimates.

3. ENSEMBLE ADVANTAGE:
   Random Forest aggregates predictions from multiple decision trees, reducing
   variance and improving robustness compared to a single linear model.

4. QUANTIFIED IMPROVEMENT:
   - MAE changed by {mae_improvement:+.2f}% (from {lr_test_mae:.4f} to {rf_test_mae:.4f})
   - RMSE changed by {rmse_improvement:+.2f}% (from {lr_test_rmse:.4f} to {rf_test_rmse:.4f})
   - R² changed from {lr_test_r2:.4f} to {rf_test_r2:.4f} ({r2_improvement:+.4f})

5. INTERPRETABILITY RETAINED:
   Feature importance analysis reveals which variables drive predictions most,
   maintaining model interpretability despite increased complexity.
""")

# -------------------------------------------------------
# Plot: Side-by-side Comparison Chart
# -------------------------------------------------------
fig = plt.figure(figsize=(16, 10), dpi=150)
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# --- Subplot 1: Actual vs Both Predictions ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(dates_test, y_test_raw, 'b-o', markersize=4, linewidth=1.8,
         label='Actual', alpha=0.9, zorder=3)
ax1.plot(dates_test, y_test_pred_lr_scaled, 'r--s', markersize=3, linewidth=1.2,
         label='Linear Regression (Sprint 2)', alpha=0.7)
ax1.plot(dates_test, y_test_pred_rf, 'g--^', markersize=3, linewidth=1.2,
         label='Random Forest (Sprint 3)', alpha=0.7)
ax1.set_title('Actual vs Predicted — Both Models on Test Set',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Unemployment Rate (u_rate)', fontsize=10)
ax1.legend(fontsize=9, loc='upper left')
ax1.tick_params(axis='x', rotation=45)

# --- Subplot 2: Metric Comparison Bar Chart ---
ax2 = fig.add_subplot(gs[1, 0])
metrics_labels = ['MAE', 'RMSE', 'R²']
lr_values = [lr_test_mae, lr_test_rmse, lr_test_r2]
rf_values = [rf_test_mae, rf_test_rmse, rf_test_r2]

x_pos = np.arange(len(metrics_labels))
width = 0.3

bars1 = ax2.bar(x_pos - width/2, lr_values, width, label='Sprint 2 (Linear Regression)',
                color='#e74c3c', alpha=0.85, edgecolor='white')
bars2 = ax2.bar(x_pos + width/2, rf_values, width, label='Sprint 3 (Random Forest)',
                color='#2ecc71', alpha=0.85, edgecolor='white')

ax2.set_title('Test Set Metric Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_labels, fontsize=10)
ax2.legend(fontsize=8, loc='upper right')

# Add value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

# --- Subplot 3: Residuals Comparison ---
ax3 = fig.add_subplot(gs[1, 1])
residuals_lr = y_test_raw - y_test_pred_lr_scaled
residuals_rf = y_test_raw - y_test_pred_rf

ax3.scatter(y_test_pred_lr_scaled, residuals_lr, c='#e74c3c', alpha=0.6, s=30,
            label='Sprint 2 (LR)', edgecolors='white', linewidth=0.5)
ax3.scatter(y_test_pred_rf, residuals_rf, c='#2ecc71', alpha=0.6, s=30,
            label='Sprint 3 (RF)', edgecolors='white', linewidth=0.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_title('Residuals: Predicted vs Error', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted u_rate', fontsize=10)
ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')

fig.suptitle('Q2(d) Model Comparison — Sprint 2 vs Sprint 3',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()

plot_path_comp = os.path.join(OUTPUT_DIR, "q2d_comparison.png")
plt.savefig(plot_path_comp, dpi=150, bbox_inches='tight')
plt.close()
print(f"Comparison plot saved to: {plot_path_comp}")

print("\n" + "=" * 80)
print("SPRINT 3 COMPLETE")
print("=" * 80)
print(f"\nAll outputs saved:")
print(f"  Models:  {MODEL_DIR}")
print(f"  Plots:   {OUTPUT_DIR}")
print(f"\nFiles generated:")
print(f"  - {os.path.join(MODEL_DIR, 'improved_model.joblib')}")
print(f"  - {plot_path_pred}")
print(f"  - {plot_path_imp}")
print(f"  - {plot_path_comp}")
