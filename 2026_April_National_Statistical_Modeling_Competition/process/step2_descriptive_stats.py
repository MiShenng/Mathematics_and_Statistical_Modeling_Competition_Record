#!/usr/bin/env python3
"""
Phase 2: Descriptive Statistics and Data Exploration
=====================================================
Descriptive stats, time-series plots, correlation heatmap,
ADF unit-root tests, X5-X6 correlation verification, M5 VIF verification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = Path("/Users/kongfei/Desktop/统计建模数据集_副本")
OUTPUT_DIR = BASE_DIR / "output" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load weekly panel
panel = pd.read_csv(BASE_DIR / "output" / "data" / "weekly_panel.csv", 
                     index_col='date', parse_dates=True)

# Working subset: rows with valid Y
panel_valid = panel.dropna(subset=['Y']).copy()
VARS_MAIN = ['Y', 'X1', 'X2', 'X3', 'X4', 'X6', 'M1', 'M2', 'M3', 'M4']

print("=" * 60)
print("Phase 2: Descriptive Statistics")
print("=" * 60)
print(f"Full panel: {panel.shape[0]} weeks")
print(f"Valid Y panel: {panel_valid.shape[0]} weeks")


# ============================================================
# 1. Descriptive Statistics Table
# ============================================================
print("\n--- 1. Descriptive Statistics ---")
desc = panel_valid[VARS_MAIN].describe().T
desc['skewness'] = panel_valid[VARS_MAIN].skew()
desc['kurtosis'] = panel_valid[VARS_MAIN].kurtosis()
desc = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]
print(desc.round(4))
desc.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", float_format='%.4f')


# ============================================================
# 2. Time Series Plots (all 10 variables)
# ============================================================
print("\n--- 2. Time Series Plots ---")

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('All Variables — Weekly Time Series (Apr 2025 – Feb 2026)', fontsize=14, y=0.98)

var_labels = {
    'Y':  'Y: Composite Storage Price Index',
    'X1': 'X1: Hyperscaler CapEx (M USD)',
    'X2': 'X2: Manufacturer COGS (summed)',
    'X3': 'X3: US EPU Index',
    'X4': 'X4: Brent Crude Oil ($/bbl)',
    'X6': 'X6: Production-Cut Aftershock (%)',
    'M1': 'M1: HBM Capacity Crowding (%)',
    'M2': 'M2: Supply-Demand Gap (neg DIO YoY)',
    'M3': 'M3: Contract Liabilities (M USD)',
    'M4': 'M4: Semiconductor PPI',
}

for idx, var in enumerate(VARS_MAIN):
    ax = axes[idx // 2, idx % 2]
    data = panel_valid[var]
    ax.plot(data.index, data.values, 'b-', linewidth=1.5)
    ax.set_title(var_labels.get(var, var), fontsize=11)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "time_series_all_variables.png")
plt.close()
print("  Saved: time_series_all_variables.png")


# ============================================================
# 3. Correlation Heatmap
# ============================================================
print("\n--- 3. Correlation Heatmap ---")

corr = panel_valid[VARS_MAIN].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            mask=mask, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Pearson Correlation'})
ax.set_title('Correlation Heatmap — All Modeling Variables', fontsize=13)
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()
print("  Saved: correlation_heatmap.png")
print(f"\n  Correlation matrix:\n{corr.round(3)}")


# ============================================================
# 4. ADF Unit Root Tests
# ============================================================
print("\n--- 4. ADF Unit Root Tests ---")

adf_results = []
for var in VARS_MAIN:
    series = panel_valid[var].dropna()
    if len(series) < 10:
        adf_results.append({'Variable': var, 'Level_ADF_stat': np.nan,
                            'Level_pvalue': np.nan, 'Level_stationary': 'N/A',
                            'Diff_ADF_stat': np.nan, 'Diff_pvalue': np.nan,
                            'Diff_stationary': 'N/A', 'Integration_order': 'N/A'})
        continue
    
    # Level test
    try:
        adf_level = adfuller(series, maxlag=4, autolag='AIC')
        level_stat, level_pval = adf_level[0], adf_level[1]
        level_stationary = level_pval < 0.05
    except Exception:
        level_stat, level_pval, level_stationary = np.nan, np.nan, False
    
    # First difference test
    diff_series = series.diff().dropna()
    try:
        adf_diff = adfuller(diff_series, maxlag=4, autolag='AIC')
        diff_stat, diff_pval = adf_diff[0], adf_diff[1]
        diff_stationary = diff_pval < 0.05
    except Exception:
        diff_stat, diff_pval, diff_stationary = np.nan, np.nan, False
    
    if level_stationary:
        order = 'I(0)'
    elif diff_stationary:
        order = 'I(1)'
    else:
        order = 'I(2) or higher'
    
    adf_results.append({
        'Variable': var,
        'Level_ADF_stat': round(level_stat, 4) if not np.isnan(level_stat) else np.nan,
        'Level_pvalue': round(level_pval, 4) if not np.isnan(level_pval) else np.nan,
        'Level_stationary': 'Yes' if level_stationary else 'No',
        'Diff_ADF_stat': round(diff_stat, 4) if not np.isnan(diff_stat) else np.nan,
        'Diff_pvalue': round(diff_pval, 4) if not np.isnan(diff_pval) else np.nan,
        'Diff_stationary': 'Yes' if diff_stationary else 'No',
        'Integration_order': order,
    })

adf_df = pd.DataFrame(adf_results)
print(adf_df.to_string(index=False))
adf_df.to_csv(OUTPUT_DIR / "adf_unit_root_tests.csv", index=False)


# ============================================================
# 5. X5 ↔ X6 Correlation Verification
# ============================================================
print("\n--- 5. X5 ↔ X6 Correlation Verification ---")

x5 = panel_valid['X5_verify']
x6 = panel_valid['X6']

pearson_r, pearson_p = stats.pearsonr(x5, x6)
spearman_r, spearman_p = stats.spearmanr(x5, x6)
kendall_t, kendall_p = stats.kendalltau(x5, x6)

print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.6f})")
print(f"  Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.6f})")
print(f"  Kendall  τ = {kendall_t:.4f}  (p = {kendall_p:.6f})")
print(f"  → Both X5 and X6 are deterministic time functions, highly collinear.")
print(f"    X5 is dropped per proposal.")

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x5, x6, alpha=0.7, edgecolors='k', linewidths=0.5)
ax.set_xlabel('X5 (Fab Countdown, weeks)')
ax.set_ylabel('X6 (Production-Cut Aftershock, %)')
ax.set_title(f'X5 vs X6 Collinearity\nPearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}')
ax.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "x5_x6_collinearity.png")
plt.close()


# ============================================================
# 6. M5 VIF Verification (M2 vs hypothetical M5)
# ============================================================
print("\n--- 6. M5 VIF Verification ---")

# M5 would be DIO level; M2 is negated DIO YoY change
# Compute Micron DIO for VIF check
x2_dir = BASE_DIR / "所有数据集" / "自变量" / "X2"
inv_df = pd.read_csv(x2_dir / "Micron Technology（Inventories）_Clean.csv")
cogs_df = pd.read_csv(x2_dir / "Micron Technology（Cost of Goods & Services）_Clean.csv")
merged = inv_df.merge(cogs_df, on=['Ticker', 'Period'])
merged['DIO'] = (merged['Inventories'] / merged['Cost of Goods & Services']) * 91
merged['DIO_lag4'] = merged['DIO'].shift(4)
merged['DIO_yoy'] = (merged['DIO'] - merged['DIO_lag4']) / merged['DIO_lag4']
merged['M2'] = -merged['DIO_yoy']
merged['M5'] = merged['DIO']

valid_m = merged.dropna(subset=['M2', 'M5'])
if len(valid_m) >= 3:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler
    
    # Simple VIF: regress M5 on M2 and all X variables
    # For demonstration, compute correlation between M2 and M5
    r_m2_m5, p_m2_m5 = stats.pearsonr(valid_m['M2'], valid_m['M5'])
    vif_approx = 1 / (1 - r_m2_m5**2) if abs(r_m2_m5) < 1 else float('inf')
    print(f"  M2-M5 Pearson r = {r_m2_m5:.4f}")
    print(f"  Approximate VIF(M5) = {vif_approx:.2f}")
    if vif_approx > 10:
        print(f"  → VIF > 10: severe multicollinearity confirmed. M5 is dropped.")
    else:
        print(f"  → VIF < 10 in bivariate case, but per proposal, M5 is dropped.")
else:
    print("  Insufficient data for VIF check.")


# ============================================================
# 7. Distribution Plots
# ============================================================
print("\n--- 7. Distribution Plots ---")

fig, axes = plt.subplots(5, 2, figsize=(14, 18))
fig.suptitle('Variable Distributions', fontsize=14, y=0.98)

for idx, var in enumerate(VARS_MAIN):
    ax = axes[idx // 2, idx % 2]
    data = panel_valid[var].dropna()
    ax.hist(data, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.2f}')
    ax.set_title(var_labels.get(var, var), fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distributions.png")
plt.close()
print("  Saved: distributions.png")


print("\n" + "=" * 60)
print("Phase 2 Complete! All outputs in:", OUTPUT_DIR)
print("=" * 60)
