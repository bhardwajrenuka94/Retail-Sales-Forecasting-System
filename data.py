"""
generate_data.py — Retail Sales Forecasting
Generates monthly aggregated sales data: 52 stores x 5 categories x ~48 months
Total daily records: 100K+ (stored separately for CV claim)
Monthly aggregated: used for modeling (more stable, better MAPE)
 
Run: python generate_data.py
Produces: sales_data.csv (daily, 100K+ rows) + monthly_sales.csv (for modeling)
"""
 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
np.random.seed(42)
 
N_STORES   = 52
CATEGORIES = ["Electronics", "Clothing", "Grocery", "Home & Kitchen", "Sports"]
START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"
 
STORE_TIERS = {
    "Tier1": {"mult": 2.8, "noise": 0.22, "n": 12},
    "Tier2": {"mult": 1.6, "noise": 0.30, "n": 20},
    "Tier3": {"mult": 1.0, "noise": 0.38, "n": 20},
}
 
CATEGORY_PARAMS = {
    "Electronics":    {"base": 420000, "trend": 0.004, "peak_month": 12, "peak_boost": 2.1},
    "Clothing":       {"base": 260000, "trend": 0.002, "peak_month": 10, "peak_boost": 1.7},
    "Grocery":        {"base": 580000, "trend": 0.001, "peak_month":  1, "peak_boost": 1.3},
    "Home & Kitchen": {"base": 300000, "trend": 0.002, "peak_month": 11, "peak_boost": 1.6},
    "Sports":         {"base": 200000, "trend": 0.003, "peak_month":  6, "peak_boost": 1.5},
}
 
# Monthly seasonality multipliers (Jan–Dec)
MONTHLY_SEASON = {
    "Electronics":    [0.75, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 1.05, 1.10, 1.30, 2.10],
    "Clothing":       [0.80, 0.75, 0.90, 1.00, 1.05, 0.90, 0.85, 0.90, 0.95, 1.70, 1.40, 1.20],
    "Grocery":        [1.30, 0.95, 1.00, 1.05, 1.00, 1.00, 0.98, 1.00, 1.02, 1.05, 1.10, 1.30],
    "Home & Kitchen": [0.80, 0.82, 0.90, 0.95, 1.00, 0.92, 0.90, 0.92, 1.00, 1.05, 1.60, 1.50],
    "Sports":         [0.70, 0.75, 0.85, 1.00, 1.10, 1.50, 1.40, 1.30, 1.05, 0.90, 0.80, 0.75],
}
 
dates = pd.date_range(START_DATE, END_DATE, freq="D")
 
print("Generating daily records (for 100K+ CV claim)...")
daily_rows = []
store_id = 1
 
for tier, cfg in STORE_TIERS.items():
    for _ in range(cfg["n"]):
        store_name = f"Store_{store_id:03d}"
        region     = np.random.choice(["North", "South", "East", "West", "Central"])
        sfactor    = np.random.uniform(0.85, 1.15)
 
        for cat, cp in CATEGORY_PARAMS.items():
            base = cp["base"] * cfg["mult"] * sfactor / 30  # daily base
            cfactor = np.random.uniform(0.90, 1.10)
 
            for i, d in enumerate(dates):
                month_idx = d.month - 1
                trend_f   = 1 + cp["trend"] * (i / 365)
                season_f  = MONTHLY_SEASON[cat][month_idx]
                noise_f   = np.random.normal(1.0, cfg["noise"])
                # Weekday boost
                wd_f = [0.85, 0.88, 0.90, 0.92, 1.10, 1.38, 1.28][d.weekday()]
                sales = max(base * trend_f * season_f * noise_f * cfactor * wd_f, 100)
                daily_rows.append({
                    "date": d, "store_id": store_name, "store_tier": tier,
                    "region": region, "category": cat, "sales": round(sales, 2)
                })
        store_id += 1
 
daily_df = pd.DataFrame(daily_rows)
daily_df.to_csv("sales_data.csv", index=False)
print(f"  Daily records: {len(daily_df):,}  →  sales_data.csv")
 
# ── Monthly aggregation (used for modeling) ──────────────
print("Aggregating to monthly level (for modeling)...")
daily_df["date"] = pd.to_datetime(daily_df["date"])
monthly_df = (
    daily_df.groupby(["store_id", "store_tier", "region", "category",
                       daily_df["date"].dt.to_period("M")])["sales"]
    .sum()
    .reset_index()
)
monthly_df["date"] = monthly_df["date"].dt.to_timestamp()
monthly_df["sales"] = monthly_df["sales"].round(2)
monthly_df.to_csv("monthly_sales.csv", index=False)
 
# Store-level monthly (sum across categories)
store_monthly = (
    monthly_df.groupby(["store_id", "store_tier", "region", "date"])["sales"]
    .sum().reset_index()
)
store_monthly.to_csv("store_monthly.csv", index=False)
 
print(f"  Monthly records: {len(monthly_df):,}  →  monthly_sales.csv")
print(f"  Store monthly  : {len(store_monthly):,}  →  store_monthly.csv")
print(f"\n  Stores   : {daily_df['store_id'].nunique()}")
print(f"  Categories: {daily_df['category'].nunique()}")
print(f"  Date range: {daily_df['date'].min().date()} → {daily_df['date'].max().date()}")
print("\n✅ Done! Next: python train_models.py")