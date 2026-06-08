#!/usr/bin/env python3
"""
Phase 3: Grey Relational Analysis (GRA)
========================================
Factor screening, lag identification, robustness checks.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = Path("/Users/kongfei/Desktop/统计建模数据集_副本")
OUTPUT_DIR = BASE_DIR / "output" / "phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load panel
panel = pd.read_csv(BASE_DIR / "output" / "data" / "weekly_panel.csv",
                     index_col='date', parse_dates=True)
panel_valid = panel.dropna(subset=['Y']).copy()

FACTORS = ['X1', 'X2', 'X3', 'X4', 'X6', 'M1', 'M2', 'M3', 'M4']
RHO_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
MAX_LAG = 6
rho_default = 0.5

print("=" * 60)
print("Phase 3: Grey Relational Analysis")
print("=" * 60)


# ============================================================
# GRA Core Functions
# ============================================================
def gra_normalize(series):
    """Normalize by first observation (初值像法)."""
    return series / series.iloc[0]


def grey_relational_coefficient(ref, comp, rho=0.5):
    """Compute grey relational coefficient at each time point."""
    delta = np.abs(ref.values - comp.values)
    delta_min = delta.min()
    delta_max = delta.max()
    xi = (delta_min + rho * delta_max) / (delta + rho * delta_max)
    return xi


def grey_relational_grade(ref, comp, rho=0.5):
    """Compute grey relational grade (average of coefficients)."""
    xi = grey_relational_coefficient(ref, comp, rho)
    return xi.mean()


def time_lag_gra(ref_series, comp_series, max_tau, rho=0.5):
    """
    Compute grey relational grade at each lag tau.
    ref_series[k+tau] is compared with comp_series[k].
    
    IMPORTANT: Normalize ONCE globally before computing lags.
    Do NOT re-normalize at each lag — that artificially favors tau=0
    for trending series.
    """
    # Global normalization (初值像法) — done ONCE on the full series
    ref_norm_full = ref_series / ref_series.iloc[0]
    comp_norm_full = comp_series / comp_series.iloc[0]
    
    grades = {}
    for tau in range(max_tau + 1):
        if tau == 0:
            ref_norm = ref_norm_full.values
            comp_norm = comp_norm_full.values
        else:
            # Y(k+tau) vs X(k): X at time k influences Y at time k+tau
            ref_norm = ref_norm_full.iloc[tau:].values
            comp_norm = comp_norm_full.iloc[:-tau].values
        
        n = len(ref_norm)
        if n < 5:
            grades[tau] = np.nan
            continue
        
        delta = np.abs(ref_norm - comp_norm)
        delta_min = delta.min()
        delta_max = delta.max()
        if delta_max == 0:
            grades[tau] = 1.0
        else:
            xi = (delta_min + rho * delta_max) / (delta + rho * delta_max)
            grades[tau] = xi.mean()
    
    return grades


# ============================================================
# 1. GRA at zero lag under multiple rho values
# ============================================================
print("\n--- 1. GRA Grades at Zero Lag (5 rho values) ---")

Y_norm = gra_normalize(panel_valid['Y'])

gra_results = {}
for rho in RHO_VALUES:
    grades = {}
    for factor in FACTORS:
        f_norm = gra_normalize(panel_valid[factor])
        grades[factor] = grey_relational_grade(Y_norm, f_norm, rho)
    gra_results[rho] = grades

gra_df = pd.DataFrame(gra_results)
gra_df.columns = [f'rho={r}' for r in RHO_VALUES]
gra_df.index.name = 'Factor'

# Rank at each rho
rank_df = gra_df.rank(ascending=False).astype(int)
rank_df.columns = [f'Rank(rho={r})' for r in RHO_VALUES]

print("\nGrey Relational Grades:")
print(gra_df.round(4))
print("\nRankings:")
print(rank_df)


# ============================================================
# 2. Kendall's W Concordance for rho robustness
# ============================================================
print("\n--- 2. Kendall's W (rho robustness) ---")

m = len(RHO_VALUES)  # number of rankings
n = len(FACTORS)     # number of factors
ranks_matrix = rank_df.values  # shape (n, m)

# Sum of ranks for each factor across m rankings
R_i = ranks_matrix.sum(axis=1)
R_bar = R_i.mean()
S = np.sum((R_i - R_bar) ** 2)
W = 12 * S / (m**2 * (n**3 - n))
chi2 = m * (n - 1) * W
p_val = 1 - stats.chi2.cdf(chi2, n - 1)

print(f"  Kendall's W = {W:.4f}")
print(f"  Chi-square  = {chi2:.4f}")
print(f"  p-value     = {p_val:.6f}")
if W > 0.7:
    print("  → W > 0.7: Rankings are stable across rho values ✓")
else:
    print("  → W < 0.7: Some instability in rankings across rho values")


# ============================================================
# 3. Time-Lag GRA: Characteristic Lag tau_i*
#    Use FIRST DIFFERENCES to remove common trend bias.
#    Level-based GRA always gives tau*=0 when all series trend together.
# ============================================================
print("\n--- 3. Time-Lag GRA (on first-differenced data) ---")

# First-difference all series to remove trend
Y_diff = panel_valid['Y'].diff().dropna()

lag_results = {}
char_lags = {}

for factor in FACTORS:
    f_diff = panel_valid[factor].diff().dropna()
    
    # Align Y_diff and f_diff (same index after diff)
    common_idx = Y_diff.index.intersection(f_diff.index)
    y_d = Y_diff.loc[common_idx]
    f_d = f_diff.loc[common_idx]
    
    # For differenced data, use range normalization (min-max) instead of
    # initial-value normalization (which doesn't work for diffs that can be negative)
    def range_normalize(s):
        smin, smax = s.min(), s.max()
        if smax == smin:
            return s * 0 + 0.5
        return (s - smin) / (smax - smin)
    
    y_norm = range_normalize(y_d)
    f_norm = range_normalize(f_d)
    
    grades = {}
    for tau in range(MAX_LAG + 1):
        if tau == 0:
            ref = y_norm.values
            comp = f_norm.values
        else:
            # X at time k influences Y at time k+tau
            ref = y_norm.iloc[tau:].values
            comp = f_norm.iloc[:-tau].values
        
        n = len(ref)
        if n < 5:
            grades[tau] = np.nan
            continue
        
        delta = np.abs(ref - comp)
        delta_min = delta.min()
        delta_max = delta.max()
        if delta_max == 0:
            grades[tau] = 1.0
        else:
            xi = (delta_min + rho_default * delta_max) / (delta + rho_default * delta_max)
            grades[tau] = xi.mean()
    
    lag_results[factor] = grades
    # Find best lag
    valid_grades = {k: v for k, v in grades.items() if not np.isnan(v)}
    if valid_grades:
        best_tau = max(valid_grades, key=valid_grades.get)
    else:
        best_tau = 0
    char_lags[factor] = best_tau

lag_df = pd.DataFrame(lag_results).T
lag_df.columns = [f'tau={t}' for t in range(MAX_LAG + 1)]
lag_df['tau*'] = pd.Series(char_lags)
lag_df['gamma(tau*)'] = [lag_results[f][char_lags[f]] for f in FACTORS]

print("\nTime-Lag Grey Relational Grades (ρ=0.5, first-differenced):")
print(lag_df.round(4))


# ============================================================
# 4. Factor Screening: gamma(tau*) >= mean(gamma)
# ============================================================
print("\n--- 4. Factor Screening ---")

mean_gamma = lag_df['gamma(tau*)'].mean()
print(f"  Mean gamma(tau*) = {mean_gamma:.4f}")

screened = lag_df[lag_df['gamma(tau*)'] >= mean_gamma].copy()
screened_factors = list(screened.index)
dropped_factors = [f for f in FACTORS if f not in screened_factors]

print(f"  Threshold: gamma >= {mean_gamma:.4f}")
print(f"  RETAINED factors ({len(screened_factors)}): {screened_factors}")
print(f"  DROPPED  factors ({len(dropped_factors)}): {dropped_factors}")

# Save screening results
screening_df = lag_df[['tau*', 'gamma(tau*)']].copy()
screening_df['Retained'] = [f in screened_factors for f in FACTORS]
screening_df.to_csv(OUTPUT_DIR / "gra_screening_results.csv")


# ============================================================
# 5. Visualization
# ============================================================
print("\n--- 5. Generating Plots ---")

# 5a. GRA grades at zero lag, bar chart for each rho
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(FACTORS))
width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, rho in enumerate(RHO_VALUES):
    vals = [gra_results[rho][f] for f in FACTORS]
    ax.bar(x + i * width, vals, width, label=f'ρ={rho}', color=colors[i], alpha=0.8)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(FACTORS, fontsize=10)
ax.set_ylabel('Grey Relational Grade')
ax.set_title('GRA Grades at Zero Lag — Robustness Across ρ Values')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.savefig(OUTPUT_DIR / "gra_zero_lag_rho_robustness.png")
plt.close()

# 5b. Time-lag profiles
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Time-Lag Grey Relational Grades (ρ=0.5)', fontsize=13, y=0.98)

for idx, factor in enumerate(FACTORS):
    ax = axes[idx // 3, idx % 3]
    taus = list(range(MAX_LAG + 1))
    grades = [lag_results[factor][t] for t in taus]
    ax.bar(taus, grades, color='steelblue', alpha=0.8)
    best = char_lags[factor]
    ax.bar(best, lag_results[factor][best], color='red', alpha=0.9)
    ax.set_title(f'{factor} (τ*={best})', fontsize=10)
    ax.set_xlabel('Lag τ (weeks)')
    ax.set_ylabel('γ(τ)')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gra_time_lag_profiles.png")
plt.close()

# 5c. Factor screening summary
fig, ax = plt.subplots(figsize=(10, 5))
gammas = lag_df['gamma(tau*)'].sort_values(ascending=True)
colors = ['green' if f in screened_factors else 'gray' for f in gammas.index]
ax.barh(gammas.index, gammas.values, color=colors, alpha=0.8, edgecolor='black')
ax.axvline(mean_gamma, color='red', linestyle='--', linewidth=2, label=f'Threshold γ̄={mean_gamma:.4f}')
ax.set_xlabel('Grey Relational Grade γ(τ*)')
ax.set_title('Factor Screening — GRA at Characteristic Lag')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.savefig(OUTPUT_DIR / "gra_factor_screening.png")
plt.close()

print("  Saved: gra_zero_lag_rho_robustness.png")
print("  Saved: gra_time_lag_profiles.png")
print("  Saved: gra_factor_screening.png")

# Save all results
gra_df.to_csv(OUTPUT_DIR / "gra_grades_zero_lag.csv")
rank_df.to_csv(OUTPUT_DIR / "gra_rankings.csv")
lag_df.to_csv(OUTPUT_DIR / "gra_time_lag_grades.csv")

# Save a summary for later phases
summary = {
    'screened_factors': screened_factors,
    'char_lags': {f: int(char_lags[f]) for f in screened_factors},
    'kendall_W': float(W),
}
import json
with open(OUTPUT_DIR / "gra_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("Phase 3 Complete!")
print(f"Screened factors: {screened_factors}")
print(f"Characteristic lags: {char_lags}")
print(f"{'='*60}")
