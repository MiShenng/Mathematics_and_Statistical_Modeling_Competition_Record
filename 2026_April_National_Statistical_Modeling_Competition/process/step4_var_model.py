#!/usr/bin/env python3
"""
Phase 4: VAR Model, Impulse Response, and Variance Decomposition
==================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = Path("/Users/kongfei/Desktop/统计建模数据集_副本")
OUTPUT_DIR = BASE_DIR / "output" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load panel and GRA results
panel = pd.read_csv(BASE_DIR / "output" / "data" / "weekly_panel.csv",
                     index_col='date', parse_dates=True)
panel_valid = panel.dropna(subset=['Y']).copy()

with open(BASE_DIR / "output" / "phase3" / "gra_summary.json") as f:
    gra_summary = json.load(f)

screened = gra_summary['screened_factors']
print("=" * 60)
print("Phase 4: VAR Model")
print("=" * 60)
print(f"GRA-screened factors: {screened}")

# ============================================================
# 1. Determine differencing based on ADF results
# ============================================================
print("\n--- 1. Stationarity & Differencing ---")

# Re-run ADF to determine which variables need differencing
vars_for_var = ['Y'] + screened
diff_needed = {}

for var in vars_for_var:
    series = panel_valid[var].dropna()
    try:
        result = adfuller(series, maxlag=4, autolag='AIC')
        stationary = result[1] < 0.05
    except:
        stationary = False
    diff_needed[var] = not stationary
    status = "→ DIFF" if not stationary else "→ LEVEL (stationary)"
    print(f"  {var}: ADF p={result[1]:.4f} {status}")

# Build differenced dataset
df_var = panel_valid[vars_for_var].copy()
for var in vars_for_var:
    if diff_needed[var]:
        df_var[var] = df_var[var].diff()

df_var = df_var.dropna()
print(f"\nVAR dataset: {df_var.shape[0]} obs × {df_var.shape[1]} vars")
print(f"Variables (after differencing): {list(df_var.columns)}")

# Verify all differenced series are stationary
print("\nPost-differencing ADF check:")
for var in df_var.columns:
    result = adfuller(df_var[var].dropna(), maxlag=4, autolag='AIC')
    print(f"  {var}: ADF stat={result[0]:.4f}, p={result[1]:.4f} {'✓' if result[1]<0.05 else '✗'}")


# ============================================================
# 2. VAR Lag Selection by BIC
# ============================================================
print("\n--- 2. Lag Selection ---")

# Limit to top 5-6 variables per proposal recommendation
# If more than 6 variables + Y, select top 5 by GRA grade
if len(screened) > 5:
    # Load GRA grades to pick top 5
    gra_grades = pd.read_csv(BASE_DIR / "output" / "phase3" / "gra_time_lag_grades.csv", index_col=0)
    top5 = gra_grades.loc[screened, 'gamma(tau*)'].sort_values(ascending=False).head(5).index.tolist()
    print(f"  Limiting VAR to top 5 factors: {top5}")
    vars_in_var = ['Y'] + top5
else:
    vars_in_var = ['Y'] + screened
    top5 = screened

df_var_model = df_var[vars_in_var].copy()

model = VAR(df_var_model)
# Try lags 1 and 2 (limited by sample size)
max_lag = min(2, len(df_var_model) // (len(vars_in_var) * 3))
max_lag = max(1, max_lag)

try:
    lag_order = model.select_order(maxlags=max_lag)
    print(f"\n  Lag selection results:")
    print(lag_order.summary())
    optimal_lag = lag_order.bic
except:
    optimal_lag = 1
    print(f"  Defaulting to lag = {optimal_lag}")

# Use BIC-optimal lag, but at least 1
if optimal_lag < 1:
    optimal_lag = 1
print(f"\n  Selected lag order (BIC): p = {optimal_lag}")


# ============================================================
# 3. VAR Estimation
# ============================================================
print("\n--- 3. VAR Estimation ---")

var_result = model.fit(optimal_lag)
print(var_result.summary())

# Stability check
print("\n  Stability check (all eigenvalues inside unit circle):")
eigenvalues = np.abs(np.linalg.eigvals(np.array(var_result.coefs).reshape(
    optimal_lag * len(vars_in_var), len(vars_in_var))[:len(vars_in_var), :]))
roots = var_result.roots
print(f"  Max eigenvalue modulus: {max(np.abs(roots)):.4f}")
is_stable = all(np.abs(roots) < 1)
print(f"  Stable: {'Yes ✓' if is_stable else 'No ✗'}")


# ============================================================
# 4. Granger Causality Tests
# ============================================================
print("\n--- 4. Granger Causality Tests ---")

granger_results = {}
for factor in top5:
    if factor in df_var_model.columns:
        try:
            test_data = df_var_model[['Y', factor]].dropna()
            result = grangercausalitytests(test_data, maxlag=optimal_lag, verbose=False)
            p_val = result[optimal_lag][0]['ssr_ftest'][1]
            f_stat = result[optimal_lag][0]['ssr_ftest'][0]
            granger_results[factor] = {'F_stat': round(f_stat, 4), 'p_value': round(p_val, 4),
                                        'Significant': 'Yes' if p_val < 0.05 else 'No'}
        except Exception as e:
            granger_results[factor] = {'F_stat': np.nan, 'p_value': np.nan, 'Significant': f'Error: {e}'}

granger_df = pd.DataFrame(granger_results).T
granger_df.index.name = 'Factor'
print(granger_df)
granger_df.to_csv(OUTPUT_DIR / "granger_causality.csv")


# ============================================================
# 5. Impulse Response Functions (IRF)
# ============================================================
print("\n--- 5. Impulse Response Functions ---")

irf = var_result.irf(periods=12)

# Plot IRF for each factor's effect on Y
y_idx = vars_in_var.index('Y')
n_factors = len(top5)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Orthogonalized IRF: Factor Shocks → Storage Price (Y)', fontsize=13, y=0.98)

for idx, factor in enumerate(top5):
    ax = axes[idx // 3, idx % 3]
    f_idx = vars_in_var.index(factor)
    
    irf_vals = irf.irfs[:, y_idx, f_idx]
    lower = irf.ci()[:, y_idx, f_idx, 0] if hasattr(irf, 'ci') else None
    upper = irf.ci()[:, y_idx, f_idx, 1] if hasattr(irf, 'ci') else None
    
    horizons = np.arange(len(irf_vals))
    ax.plot(horizons, irf_vals, 'b-', linewidth=2)
    
    # Try confidence intervals
    try:
        ci = irf.ci(alpha=0.05)  # returns lower and upper
        ax.fill_between(horizons, ci[:, y_idx, f_idx, 0], ci[:, y_idx, f_idx, 1],
                        alpha=0.2, color='blue')
    except:
        pass
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f'Shock: {factor} → Y', fontsize=10)
    ax.set_xlabel('Weeks ahead')
    ax.set_ylabel('Response')
    ax.grid(True, alpha=0.3)

# Remove empty subplot if odd number
if n_factors < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "irf_factor_to_Y.png")
plt.close()
print("  Saved: irf_factor_to_Y.png")


# ============================================================
# 6. Forecast Error Variance Decomposition (FEVD)
# ============================================================
print("\n--- 6. FEVD ---")

fevd = var_result.fevd(6)
print(fevd.summary())

# Extract FEVD for Y at h=1,3,6
fevd_data = fevd.decomp
y_fevd = {}
for h in [0, 2, 5]:  # 0-indexed → h=1,3,6
    actual_h = h + 1
    row = fevd_data[h, y_idx, :]  # FEVD of Y at horizon h
    y_fevd[f'h={actual_h}'] = {vars_in_var[i]: round(row[i] * 100, 2) for i in range(len(vars_in_var))}

fevd_df = pd.DataFrame(y_fevd).T
print(f"\nFEVD of Y (%):")
print(fevd_df)
fevd_df.to_csv(OUTPUT_DIR / "fevd_Y.csv")

# FEVD bar chart
fig, ax = plt.subplots(figsize=(10, 6))
fevd_df.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', alpha=0.85)
ax.set_title('Forecast Error Variance Decomposition of Y', fontsize=12)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Variance Share (%)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.savefig(OUTPUT_DIR / "fevd_bar_chart.png")
plt.close()
print("  Saved: fevd_bar_chart.png")

# Extract FEVD ranking for cross-validation later
fevd_h6 = fevd_df.loc['h=6'].drop('Y', errors='ignore')
fevd_rank = fevd_h6.rank(ascending=False).astype(int)
print(f"\nFEVD ranking (h=6): {dict(fevd_rank)}")


# ============================================================
# 7. Iran Conflict Counterfactual IRF
# ============================================================
print("\n--- 7. Iran Conflict Counterfactual IRF ---")

# Inject shocks: X4 (oil +$40/bbl) and X3 if in model
# The shock magnitude in the differenced model
# For X4: if differenced, shock = 40 (dollar jump)
# For X3: if in model and differenced, shock = 200 (EPU spike)

shock_desc = {}
shock_vector = np.zeros(len(vars_in_var))

if 'X4' in vars_in_var:
    f_idx = vars_in_var.index('X4')
    # Get std of X4 in the model
    x4_std = df_var_model['X4'].std()
    shock_magnitude = 40.0  # $40/bbl jump
    shock_vector[f_idx] = shock_magnitude
    shock_desc['X4'] = f"+${shock_magnitude}/bbl"

if 'X3' in vars_in_var:
    f_idx = vars_in_var.index('X3')
    shock_magnitude = 200.0  # EPU spike
    shock_vector[f_idx] = shock_magnitude
    shock_desc['X3'] = f"+{shock_magnitude} EPU points"

print(f"  Shock description: {shock_desc}")

# Custom IRF with given shock
# IRF at horizon s = Phi_s @ shock_vector
ma_coefs = irf.irfs  # shape (periods+1, K, K)
y_response = np.zeros(13)
for s in range(13):
    y_response[s] = ma_coefs[s, y_idx, :] @ shock_vector

# Cumulative response (since we're in differences)
y_cumulative = np.cumsum(y_response)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(13), y_response, 'r-o', linewidth=2, markersize=5)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('Iran Conflict Shock → Y (Period Response)')
axes[0].set_xlabel('Weeks after conflict')
axes[0].set_ylabel('ΔY response')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(13), y_cumulative, 'r-o', linewidth=2, markersize=5)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('Iran Conflict Shock → Y (Cumulative Response)')
axes[1].set_xlabel('Weeks after conflict')
axes[1].set_ylabel('Cumulative ΔY')
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Counterfactual: {shock_desc}', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "iran_conflict_counterfactual_irf.png")
plt.close()
print("  Saved: iran_conflict_counterfactual_irf.png")


# ============================================================
# Save VAR summary for later phases
# ============================================================
var_summary = {
    'vars_in_var': vars_in_var,
    'optimal_lag': int(optimal_lag),
    'is_stable': bool(is_stable),
    'diff_needed': {k: bool(v) for k, v in diff_needed.items()},
    'fevd_rank_h6': {k: int(v) for k, v in fevd_rank.items()} if len(fevd_rank) > 0 else {},
}
with open(OUTPUT_DIR / "var_summary.json", 'w') as f:
    json.dump(var_summary, f, indent=2)

print(f"\n{'='*60}")
print("Phase 4 Complete!")
print(f"{'='*60}")
