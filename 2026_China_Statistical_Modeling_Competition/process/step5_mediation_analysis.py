#!/usr/bin/env python3
"""
Phase 5: Mediation Analysis (Baron-Kenny with HAC Standard Errors)
===================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = Path("/Users/kongfei/Desktop/统计建模数据集_副本")
OUTPUT_DIR = BASE_DIR / "output" / "phase5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load panel
panel = pd.read_csv(BASE_DIR / "output" / "data" / "weekly_panel.csv",
                     index_col='date', parse_dates=True)
panel_valid = panel.dropna(subset=['Y']).copy()

# Core X variables and mediators (updated after GRA fix)
X_VARS = ['X3', 'X4', 'X6']  # GRA-screened X variables
M_VARS = ['M1', 'M2', 'M3', 'M4']  # All mediators

print("=" * 60)
print("Phase 5: Mediation Analysis")
print("=" * 60)


# ============================================================
# Helper: OLS with Newey-West HAC standard errors
# ============================================================
def ols_hac(y, X, max_lags=None):
    """Run OLS with Newey-West HAC standard errors."""
    X_const = add_constant(X)
    model = OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return model


# ============================================================
# 1. Engle-Granger Cointegration Tests
# ============================================================
print("\n--- 1. Cointegration Tests ---")

coint_results = []
all_vars = X_VARS + M_VARS + ['Y']
for var in all_vars:
    if var == 'Y':
        continue
    try:
        t_stat, p_val, crit_vals = coint(panel_valid['Y'], panel_valid[var])
        coint_results.append({
            'Pair': f'Y ~ {var}',
            'EG_t_stat': round(t_stat, 4),
            'p_value': round(p_val, 4),
            'Cointegrated': 'Yes' if p_val < 0.05 else 'No'
        })
    except Exception as e:
        coint_results.append({'Pair': f'Y ~ {var}', 'EG_t_stat': np.nan,
                             'p_value': np.nan, 'Cointegrated': f'Error'})

coint_df = pd.DataFrame(coint_results)
print(coint_df.to_string(index=False))
coint_df.to_csv(OUTPUT_DIR / "cointegration_tests.csv", index=False)


# ============================================================
# 2. Baron-Kenny Mediation Analysis
# ============================================================
print("\n--- 2. Baron-Kenny Mediation Analysis ---")

# Bandwidth for Newey-West
T = len(panel_valid)
nw_lags = int(np.floor(4 * (T / 100) ** (2/9)))
print(f"  Newey-West bandwidth: {nw_lags}")

mediation_results = []

for xi in X_VARS:
    for mj in M_VARS:
        try:
            # Data
            y = panel_valid['Y'].values
            x = panel_valid[xi].values
            m = panel_valid[mj].values
            
            # Step 1: Total effect (Y = c*X)
            step1 = ols_hac(y, pd.DataFrame({xi: x}), max_lags=nw_lags)
            c_total = step1.params.iloc[1]
            c_total_se = step1.bse.iloc[1]
            c_total_p = step1.pvalues.iloc[1]
            dw1 = durbin_watson(step1.resid)
            
            # Step 2: Mediator equation (M = a*X)
            step2 = ols_hac(m, pd.DataFrame({xi: x}), max_lags=nw_lags)
            a = step2.params.iloc[1]
            a_se = step2.bse.iloc[1]
            a_p = step2.pvalues.iloc[1]
            
            # Step 3: Mediated effect (Y = c'*X + b*M)
            step3 = ols_hac(y, pd.DataFrame({xi: x, mj: m}), max_lags=nw_lags)
            c_prime = step3.params.iloc[1]  # Direct effect
            b = step3.params.iloc[2]        # Mediator effect
            b_se = step3.bse.iloc[2]
            b_p = step3.pvalues.iloc[2]
            c_prime_p = step3.pvalues.iloc[1]
            dw3 = durbin_watson(step3.resid)
            
            # Indirect effect = a * b
            indirect = a * b
            
            # Sobel test
            sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
            z_sobel = indirect / sobel_se if sobel_se > 0 else 0
            sobel_p = 2 * (1 - stats.norm.cdf(abs(z_sobel)))
            
            # Mediation classification
            if c_total_p > 0.05:
                classification = "No total effect"
            elif a_p > 0.05 or b_p > 0.05:
                classification = "No mediation"
            elif c_prime_p > 0.05:
                classification = "Full mediation"
            else:
                classification = "Partial mediation"
            
            mediation_results.append({
                'X': xi, 'M': mj,
                'c_total': round(float(c_total), 6), 'c_total_p': round(float(c_total_p), 4),
                'a': round(float(a), 6), 'a_p': round(float(a_p), 4),
                'b': round(float(b), 6), 'b_p': round(float(b_p), 4),
                'c_prime': round(float(c_prime), 6), 'c_prime_p': round(float(c_prime_p), 4),
                'indirect_ab': round(float(indirect), 6),
                'Z_Sobel': round(float(z_sobel), 4), 'Sobel_p': round(float(sobel_p), 4),
                'DW_step1': round(float(dw1), 4), 'DW_step3': round(float(dw3), 4),
                'Classification': classification,
            })
        except Exception as e:
            import traceback
            mediation_results.append({
                'X': xi, 'M': mj, 'Classification': f'Error: {e}'
            })

med_df = pd.DataFrame(mediation_results)
print("\nMediation Results:")
key_cols = ['X', 'M', 'c_total', 'c_prime', 'indirect_ab', 'Z_Sobel', 'Sobel_p', 'Classification']
available_cols = [c for c in key_cols if c in med_df.columns]
print(med_df[available_cols].to_string(index=False))
med_df.to_csv(OUTPUT_DIR / "mediation_results.csv", index=False)


# ============================================================
# 3. Effect Decomposition Summary
# ============================================================
print("\n--- 3. Effect Decomposition ---")

if 'Sobel_p' in med_df.columns:
    significant_med = med_df[med_df['Sobel_p'].astype(float) < 0.05]
    if len(significant_med) > 0:
        print("\nSignificant mediation paths (Sobel p < 0.05):")
        print(significant_med[available_cols].to_string(index=False))
    else:
        print("  No mediation paths significant at 5% level.")
        med_sorted = med_df.dropna(subset=['Sobel_p']).sort_values('Sobel_p')
        print("\n  Top 5 mediation paths by Sobel p-value:")
        print(med_sorted[available_cols].head(5).to_string(index=False))
else:
    print("  Sobel_p column not available.")


# ============================================================
# 4. Visualization: Mediation Effect Heatmap
# ============================================================
print("\n--- 4. Generating Plots ---")

# Sobel Z-statistic heatmap
pivot = med_df.pivot(index='X', columns='M', values='Z_Sobel')
fig, ax = plt.subplots(figsize=(8, 6))
import seaborn as sns
sns.heatmap(pivot.astype(float), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Sobel Z-statistic'})
ax.set_title('Mediation Analysis: Sobel Z-statistics\n(|Z| > 1.96 is significant at 5%)', fontsize=11)
plt.savefig(OUTPUT_DIR / "mediation_sobel_heatmap.png")
plt.close()

# Effect decomposition bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Effect Decomposition by Core Variable', fontsize=13, y=0.98)

for idx, xi in enumerate(X_VARS):
    ax = axes[idx // 2, idx % 2]
    subset = med_df[med_df['X'] == xi].dropna(subset=['c_total', 'c_prime', 'indirect_ab'])
    if len(subset) == 0:
        ax.set_title(f'{xi}: No valid results')
        continue
    
    x_pos = np.arange(len(subset))
    width = 0.25
    ax.bar(x_pos - width, subset['c_total'].astype(float), width, label='Total (c)', color='steelblue', alpha=0.8)
    ax.bar(x_pos, subset['c_prime'].astype(float), width, label='Direct (c\')', color='orange', alpha=0.8)
    ax.bar(x_pos + width, subset['indirect_ab'].astype(float), width, label='Indirect (a×b)', color='green', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subset['M'].values)
    ax.set_title(f'{xi} → Y via Mediators')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "effect_decomposition.png")
plt.close()

print("  Saved: mediation_sobel_heatmap.png")
print("  Saved: effect_decomposition.png")

print(f"\n{'='*60}")
print("Phase 5 Complete!")
print(f"{'='*60}")
