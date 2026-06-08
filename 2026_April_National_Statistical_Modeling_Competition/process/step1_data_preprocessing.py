#!/usr/bin/env python3
"""
Phase 1: Data Preprocessing
============================
Read all raw datasets, construct variables per revised_full_proposal.md,
harmonize to a Friday-anchored weekly grid, and output weekly_panel.csv.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path("/Users/kongfei/Desktop/统计建模数据集_副本")
DATA_DIR = BASE_DIR / "所有数据集"
OUTPUT_DIR = BASE_DIR / "output" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Y weights (market share)
W_NAND = 0.11111
W_DRAM = 0.56667
W_SSD  = 0.32222

# Weekly grid: Friday-anchored, covering Y range with buffer for lags
# Y data: April 2025 – February 2026
# Add ~8 weeks buffer before April for lag analysis
GRID_START = pd.Timestamp("2025-02-07")  # First Friday of Feb 2025
GRID_END   = pd.Timestamp("2026-02-27")  # Last Friday of Feb 2026

def build_weekly_grid(start, end):
    """Generate Friday-anchored weekly dates."""
    # Find the first Friday >= start
    first_friday = start + pd.Timedelta(days=(4 - start.weekday()) % 7)
    return pd.date_range(first_friday, end, freq='W-FRI')

WEEKLY_DATES = build_weekly_grid(GRID_START, GRID_END)
print(f"Weekly grid: {WEEKLY_DATES[0].date()} to {WEEKLY_DATES[-1].date()}, {len(WEEKLY_DATES)} weeks")


# ============================================================
# Helper: Quarterly step function with publication lag
# ============================================================
def quarterly_step(quarterly_values, weekly_dates, lag_days=45):
    """
    Convert quarterly data to weekly via step function with publication lag.
    
    quarterly_values: dict {pd.Timestamp(quarter_end): value}
    """
    # Sort by publication date
    pub_items = sorted(
        [(qend + pd.Timedelta(days=lag_days), val) for qend, val in quarterly_values.items()]
    )
    
    result = pd.Series(index=weekly_dates, dtype=float)
    for date in weekly_dates:
        available = [(pub, val) for pub, val in pub_items if pub <= date]
        if available:
            result[date] = available[-1][1]
        else:
            result[date] = np.nan
    return result


def parse_period_to_qend(period_str):
    """Convert '2024 Q1' to quarter end date."""
    parts = period_str.strip().split()
    year = int(parts[0])
    q = int(parts[1].replace('Q', ''))
    month = q * 3
    if month == 3:
        return pd.Timestamp(year, 3, 31)
    elif month == 6:
        return pd.Timestamp(year, 6, 30)
    elif month == 9:
        return pd.Timestamp(year, 9, 30)
    else:
        return pd.Timestamp(year, 12, 31)


# ============================================================
# 1. Dependent Variable Y: Composite Storage Price Index
# ============================================================
def build_Y():
    """Build composite price index from monthly data, interpolate to weekly."""
    print("\n=== Building Y (Composite Storage Price Index) ===")
    
    df = pd.read_csv(DATA_DIR / "因变量.csv", index_col=0)
    
    # Map month columns to dates
    months_cn = ['4月','5月','6月','7月','8月','9月','10月','11月','12月','1月','2月']
    # April 2025 to Feb 2026
    month_dates = [
        pd.Timestamp(2025, 4, 15), pd.Timestamp(2025, 5, 15),
        pd.Timestamp(2025, 6, 15), pd.Timestamp(2025, 7, 15),
        pd.Timestamp(2025, 8, 15), pd.Timestamp(2025, 9, 15),
        pd.Timestamp(2025, 10, 15), pd.Timestamp(2025, 11, 15),
        pd.Timestamp(2025, 12, 15), pd.Timestamp(2026, 1, 15),
        pd.Timestamp(2026, 2, 15),
    ]
    
    # Extract prices
    nand = df.loc['512Gb TLC'].values.astype(float)
    dram = df.loc['DDR5 16Gb Major'].values.astype(float)
    ssd  = df.loc['OEM SSD 512GB PCIe 4.0'].values.astype(float)
    
    # Normalize to 100 at first observation
    nand_norm = nand / nand[0] * 100
    dram_norm = dram / dram[0] * 100
    ssd_norm  = ssd  / ssd[0]  * 100
    
    # Composite index
    Y_monthly = W_NAND * nand_norm + W_DRAM * dram_norm + W_SSD * ssd_norm
    
    print(f"  Monthly Y range: {Y_monthly.min():.2f} to {Y_monthly.max():.2f}")
    
    # Cubic spline interpolation to weekly
    month_ordinals = np.array([(d - month_dates[0]).days for d in month_dates], dtype=float)
    cs = CubicSpline(month_ordinals, Y_monthly)
    
    weekly_ordinals = np.array([(d - month_dates[0]).days for d in WEEKLY_DATES], dtype=float)
    Y_weekly = pd.Series(index=WEEKLY_DATES, dtype=float)
    
    for i, (date, ordinal) in enumerate(zip(WEEKLY_DATES, weekly_ordinals)):
        if month_ordinals[0] <= ordinal <= month_ordinals[-1]:
            Y_weekly[date] = cs(ordinal)
        else:
            Y_weekly[date] = np.nan
    
    valid = Y_weekly.dropna()
    print(f"  Weekly Y: {valid.index[0].date()} to {valid.index[-1].date()}, {len(valid)} weeks")
    return Y_weekly


# ============================================================
# 2. X1: Hyperscaler Capital Expenditure (summed)
# ============================================================
def build_X1():
    """Sum CapEx across 5 hyperscalers, convert to weekly step function."""
    print("\n=== Building X1 (Hyperscaler CapEx) ===")
    
    x1_dir = DATA_DIR / "自变量" / "财报（X1）"
    companies = ['NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    all_capex = {}
    for company in companies:
        fpath = x1_dir / f"{company}_CapEx.csv"
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            period = row['Period']
            val = abs(float(row['Capital Expenditures']))  # CapEx is negative
            if period not in all_capex:
                all_capex[period] = 0.0
            all_capex[period] += val
    
    # Convert to {quarter_end: value}
    quarterly = {}
    for period, val in sorted(all_capex.items()):
        qend = parse_period_to_qend(period)
        quarterly[qend] = val
    
    # Use last 10 quarters where data is most complete
    sorted_items = sorted(quarterly.items())
    # Find quarters where at least 4 companies contributed
    # For simplicity, use whatever we have
    if len(sorted_items) > 10:
        sorted_items = sorted_items[-10:]
    quarterly = dict(sorted_items)
    
    X1 = quarterly_step(quarterly, WEEKLY_DATES)
    print(f"  X1 quarters used: {len(quarterly)}")
    print(f"  X1 range: {X1.dropna().min():.0f} to {X1.dropna().max():.0f}")
    return X1


# ============================================================
# 3. X2: Manufacturer COGS (summed across 3 manufacturers)
# ============================================================
def build_X2():
    """Sum COGS across SK Hynix, Micron, Samsung."""
    print("\n=== Building X2 (Manufacturer COGS) ===")
    
    x2_dir = DATA_DIR / "自变量" / "X2"
    cogs_files = {
        'HY9H.F': 'HY9H.F（Cost of Goods & Services）_Clean.csv',
        'Micron': 'Micron Technology（Cost of Goods & Services）_Clean.csv',
        'SSNLF': 'SSNLF（Cost of Goods & Services）_Clean.csv',
    }
    
    all_cogs = {}
    for company, fname in cogs_files.items():
        df = pd.read_csv(x2_dir / fname)
        for _, row in df.iterrows():
            period = row['Period']
            val = float(row['Cost of Goods & Services'])
            if period not in all_cogs:
                all_cogs[period] = 0.0
            all_cogs[period] += val
    
    # Use last 10 quarters
    quarterly = {}
    for period, val in sorted(all_cogs.items()):
        qend = parse_period_to_qend(period)
        quarterly[qend] = val
    
    sorted_items = sorted(quarterly.items())
    if len(sorted_items) > 10:
        sorted_items = sorted_items[-10:]
    quarterly = dict(sorted_items)
    
    X2 = quarterly_step(quarterly, WEEKLY_DATES)
    print(f"  X2 quarters used: {len(quarterly)}")
    return X2


# ============================================================
# 4. X3: US Economic Policy Uncertainty (daily -> weekly mean)
# ============================================================
def build_X3():
    """Daily EPU -> weekly mean."""
    print("\n=== Building X3 (US EPU) ===")
    
    df = pd.read_csv(DATA_DIR / "自变量" / "US EPU（2024.1-2026.4）（X3）.csv")
    df['date'] = pd.to_datetime(
        df.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    )
    df = df.set_index('date').sort_index()
    
    X3 = pd.Series(index=WEEKLY_DATES, dtype=float)
    for friday in WEEKLY_DATES:
        week_start = friday - pd.Timedelta(days=6)  # Saturday
        week_data = df.loc[week_start:friday, 'daily_policy_index']
        if len(week_data) > 0:
            X3[friday] = week_data.mean()
        else:
            X3[friday] = np.nan
    
    # Forward fill any remaining NaN
    X3 = X3.ffill()
    print(f"  X3 weekly range: {X3.dropna().min():.1f} to {X3.dropna().max():.1f}")
    return X3


# ============================================================
# 5. X4: Brent Crude Oil Price (daily -> weekly mean)
# ============================================================
def build_X4():
    """Daily Brent -> weekly mean, forward fill holidays."""
    print("\n=== Building X4 (Brent Crude Oil) ===")
    
    df = pd.read_csv(DATA_DIR / "自变量" / "布伦特原油价格（X4）.csv")
    df['date'] = pd.to_datetime(df['observation_date'])
    df['DCOILBRENTEU'] = pd.to_numeric(df['DCOILBRENTEU'], errors='coerce')
    df = df.set_index('date').sort_index()
    # Forward fill missing (holidays)
    df['DCOILBRENTEU'] = df['DCOILBRENTEU'].ffill()
    
    X4 = pd.Series(index=WEEKLY_DATES, dtype=float)
    for friday in WEEKLY_DATES:
        week_start = friday - pd.Timedelta(days=6)
        week_data = df.loc[week_start:friday, 'DCOILBRENTEU'].dropna()
        if len(week_data) > 0:
            X4[friday] = week_data.mean()
        else:
            X4[friday] = np.nan
    
    X4 = X4.ffill()
    print(f"  X4 weekly range: {X4.dropna().min():.2f} to {X4.dropna().max():.2f}")
    return X4


# ============================================================
# 6. X6: Legacy Production-Cut Aftershock (deterministic)
# ============================================================
def build_X6():
    """X6(t) = 5 + 30 * exp(-lambda * w), lambda = ln2/78, w = weeks since 2024-01-01."""
    print("\n=== Building X6 (Production-Cut Aftershock) ===")
    
    origin = pd.Timestamp("2024-01-01")
    lam = np.log(2) / 78
    
    X6 = pd.Series(index=WEEKLY_DATES, dtype=float)
    for friday in WEEKLY_DATES:
        w = (friday - origin).days / 7.0
        X6[friday] = 5 + 30 * np.exp(-lam * w)
    
    print(f"  X6 range: {X6.min():.3f} to {X6.max():.3f}")
    return X6


# ============================================================
# 7. M1: HBM Capacity Crowding (annual -> cubic spline to weekly)
# ============================================================
def build_M1():
    """Annual HBM bit output share -> cubic spline interpolation to weekly."""
    print("\n=== Building M1 (HBM Capacity Crowding) ===")
    
    # HBM share of DRAM bit output (from TrendForce report)
    annual_data = {
        pd.Timestamp("2023-07-01"): 2.0,   # 2023: 2%
        pd.Timestamp("2024-07-01"): 5.0,   # 2024: 5%
        pd.Timestamp("2025-07-01"): 8.0,   # 2025: 8% (forecast)
        pd.Timestamp("2026-07-01"): 12.0,  # 2026: extrapolated ~12%
    }
    
    dates = list(annual_data.keys())
    values = list(annual_data.values())
    ordinals = [(d - dates[0]).days for d in dates]
    
    cs = CubicSpline(ordinals, values)
    
    M1 = pd.Series(index=WEEKLY_DATES, dtype=float)
    for friday in WEEKLY_DATES:
        ord_val = (friday - dates[0]).days
        M1[friday] = cs(ord_val)
    
    print(f"  M1 range: {M1.min():.2f}% to {M1.max():.2f}%")
    return M1


# ============================================================
# 8. M2: NAND Supply-Demand Gap (Micron DIO YoY change, negated)
# ============================================================
def build_M2():
    """Micron DIO = (Inventory/COGS)*91, YoY change rate negated."""
    print("\n=== Building M2 (Supply-Demand Gap) ===")
    
    x2_dir = DATA_DIR / "自变量" / "X2"
    
    inv_df = pd.read_csv(x2_dir / "Micron Technology（Inventories）_Clean.csv")
    cogs_df = pd.read_csv(x2_dir / "Micron Technology（Cost of Goods & Services）_Clean.csv")
    
    # Merge on Period
    merged = inv_df.merge(cogs_df, on=['Ticker', 'Period'])
    merged['DIO'] = (merged['Inventories'] / merged['Cost of Goods & Services']) * 91
    
    # Parse period to quarter end
    merged['qend'] = merged['Period'].apply(parse_period_to_qend)
    merged = merged.sort_values('qend')
    
    # YoY change: compare with 4 quarters ago
    merged['DIO_lag4'] = merged['DIO'].shift(4)
    merged['DIO_yoy'] = (merged['DIO'] - merged['DIO_lag4']) / merged['DIO_lag4']
    merged['M2'] = -merged['DIO_yoy']  # Negated
    
    # Filter to rows with valid M2
    valid = merged.dropna(subset=['M2'])
    quarterly = {row['qend']: row['M2'] for _, row in valid.iterrows()}
    
    M2 = quarterly_step(quarterly, WEEKLY_DATES)
    print(f"  M2 quarters used: {len(quarterly)}")
    print(f"  M2 range: {M2.dropna().min():.4f} to {M2.dropna().max():.4f}")
    return M2


# ============================================================
# 9. M3: Hyperscaler Procurement Lockup (Contract Liabilities)
# ============================================================
def build_M3():
    """Micron Contract Liabilities, quarterly step function."""
    print("\n=== Building M3 (Contract Liabilities) ===")
    
    df = pd.read_csv(DATA_DIR / "中间变量" / "合同负债财务数据历史汇总（M3）.csv")
    
    quarterly = {}
    for _, row in df.iterrows():
        qend = parse_period_to_qend(row['Period'])
        quarterly[qend] = float(row['Contract Liabilities (Million USD)'])
    
    M3 = quarterly_step(quarterly, WEEKLY_DATES)
    print(f"  M3 range: {M3.dropna().min():.0f} to {M3.dropna().max():.0f}")
    return M3


# ============================================================
# 10. M4: Semiconductor PPI (monthly -> linear interpolation to weekly)
# ============================================================
def build_M4():
    """Monthly PPI -> linear interpolation to weekly."""
    print("\n=== Building M4 (Semiconductor PPI) ===")
    
    df = pd.read_csv(
        DATA_DIR / "中间变量" / "工业生产者价格指数：半导体及有关器件制造业（M4）（24.1-26.2）.csv"
    )
    df['date'] = pd.to_datetime(df['observation_date'])
    df = df.set_index('date').sort_index()
    
    # Linear interpolation to daily, then resample to weekly
    daily = df['PCU334413334413'].resample('D').interpolate(method='linear')
    
    M4 = pd.Series(index=WEEKLY_DATES, dtype=float)
    for friday in WEEKLY_DATES:
        if friday in daily.index:
            M4[friday] = daily[friday]
        else:
            # Find nearest date
            nearest = daily.index[daily.index.get_indexer([friday], method='nearest')[0]]
            M4[friday] = daily[nearest]
    
    print(f"  M4 range: {M4.dropna().min():.3f} to {M4.dropna().max():.3f}")
    return M4


# ============================================================
# Main: Build all variables and merge into weekly panel
# ============================================================
def main():
    print("=" * 60)
    print("Phase 1: Data Preprocessing")
    print("=" * 60)
    
    # Build all variables
    Y  = build_Y()
    X1 = build_X1()
    X2 = build_X2()
    X3 = build_X3()
    X4 = build_X4()
    X6 = build_X6()
    M1 = build_M1()
    M2 = build_M2()
    M3 = build_M3()
    M4 = build_M4()
    
    # Assemble panel
    panel = pd.DataFrame({
        'date': WEEKLY_DATES,
        'Y':  Y.values,
        'X1': X1.values,
        'X2': X2.values,
        'X3': X3.values,
        'X4': X4.values,
        'X6': X6.values,
        'M1': M1.values,
        'M2': M2.values,
        'M3': M3.values,
        'M4': M4.values,
    })
    panel = panel.set_index('date')
    
    # Also compute X5 for verification (removed variable)
    # X5: weeks remaining until SK Hynix M15X mass-production (2025-11-30)
    target_date = pd.Timestamp("2025-11-30")
    panel['X5_verify'] = [(target_date - d).days / 7.0 for d in WEEKLY_DATES]
    panel.loc[panel['X5_verify'] < 0, 'X5_verify'] = 0
    
    # Save
    out_path = OUTPUT_DIR / "weekly_panel.csv"
    panel.to_csv(out_path, float_format='%.6f')
    
    print("\n" + "=" * 60)
    print("Panel Summary")
    print("=" * 60)
    print(f"Shape: {panel.shape}")
    print(f"\nDate range: {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"\nMissing values:")
    print(panel.isnull().sum())
    print(f"\nDescriptive statistics:")
    print(panel.describe().round(4))
    print(f"\nSaved to: {out_path}")
    
    return panel


if __name__ == "__main__":
    main()
