"""
Notebook 1 — Medallion Data Pipeline (Bronze → Silver → Gold)
DoorDash New Verticals: Merchant Growth Analytics
Author: Ashwath Subramanyan

Bronze = raw ingestion, Silver = cleaned/typed/validated, Gold = business-ready aggregates.

Simulates a scheduled Databricks Workflow. In prod this would be Auto Loader
into Delta tables with DLT expectations on Silver and Z-ORDER on Gold.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(314)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("MEDALLION DATA PIPELINE: Bronze → Silver → Gold")
print("=" * 70)

# ── BRONZE LAYER ──
# Prod: Auto Loader (cloudFiles) → Delta, schema-on-read, _metadata cols

print("\n[BRONZE] Generating raw data tables...")

N_MERCHANTS = 5000
N_MONTHS = 12  # 12 months of order history for time-series analysis
BASE_DATE = datetime(2024, 1, 1)

# --- Bronze: Merchants ---
verticals = ['grocery', 'convenience', 'retail']
vertical_weights = [0.50, 0.30, 0.20]
regions = ['west', 'northeast', 'southeast', 'midwest', 'southwest']

merchants_bronze = pd.DataFrame({
    'merchant_id': [f'M{str(i).zfill(5)}' for i in range(N_MERCHANTS)],
    'merchant_name': [f'Merchant_{i}' for i in range(N_MERCHANTS)],
    'vertical': np.random.choice(verticals, N_MERCHANTS, p=vertical_weights),
    'region': np.random.choice(regions, N_MERCHANTS),
    'city': np.random.choice([
        'San Francisco', 'Los Angeles', 'New York', 'Chicago', 'Houston',
        'Phoenix', 'Miami', 'Seattle', 'Denver', 'Atlanta',
        'Boston', 'Dallas', 'Portland', 'Nashville', 'Austin'
    ], N_MERCHANTS),
    'onboard_date': [
        BASE_DATE - timedelta(days=np.random.randint(30, 730))
        for _ in range(N_MERCHANTS)
    ],
    'delivery_radius_miles': np.random.choice([3, 5, 7, 10], N_MERCHANTS, p=[0.35, 0.30, 0.20, 0.15]),
    'is_active': np.random.choice([True, False], N_MERCHANTS, p=[0.92, 0.08]),
    '_ingested_at': datetime.now().isoformat(),
    '_source_file': 'merchant_export_20250101.csv'
})

# --- Bronze: Merchant Engagement Events (time-series) ---
# Each merchant has monthly engagement snapshots
engagement_records = []
for _, m in merchants_bronze.iterrows():
    # Generate engagement trajectory (some merchants improve over time)
    trajectory = np.random.choice(['improving', 'stable', 'declining'], p=[0.30, 0.50, 0.20])

    for month_offset in range(N_MONTHS):
        month_date = BASE_DATE + timedelta(days=30 * month_offset)

        # Base probabilities vary by trajectory
        if trajectory == 'improving':
            promo_prob = min(0.25 + 0.05 * month_offset, 0.85)
            photo_prob = min(0.40 + 0.04 * month_offset, 0.90)
        elif trajectory == 'declining':
            promo_prob = max(0.50 - 0.03 * month_offset, 0.10)
            photo_prob = max(0.60 - 0.03 * month_offset, 0.15)
        else:
            promo_prob = 0.35
            photo_prob = 0.55

        engagement_records.append({
            'merchant_id': m['merchant_id'],
            'snapshot_month': month_date.strftime('%Y-%m-01'),
            'promotions_active': np.random.random() < promo_prob,
            'photos_uploaded': np.random.random() < photo_prob,
            'menu_completeness_pct': np.clip(np.random.normal(0.65, 0.20), 0.10, 1.00),
            'avg_response_time_hours': np.clip(np.random.exponential(4.0), 0.5, 48.0),
            'num_menu_items': np.random.randint(5, 200),
            '_ingested_at': datetime.now().isoformat()
        })

engagement_bronze = pd.DataFrame(engagement_records)

# --- Bronze: Orders (monthly aggregates per merchant) ---
order_records = []
for _, m in merchants_bronze.iterrows():
    vertical = m['vertical']
    base_basket = {'grocery': 47, 'convenience': 21, 'retail': 72}[vertical]
    base_orders = {'grocery': 120, 'convenience': 180, 'retail': 60}[vertical]

    # Add merchant-level random effect
    merchant_quality = np.random.normal(1.0, 0.3)

    for month_offset in range(N_MONTHS):
        month_date = BASE_DATE + timedelta(days=30 * month_offset)
        # Seasonality: slight boost in Nov-Dec
        seasonal = 1.15 if month_date.month in [11, 12] else 1.0

        monthly_orders = max(1, int(
            base_orders * merchant_quality * seasonal * np.random.normal(1.0, 0.25)
        ))
        avg_basket = max(5, base_basket * np.random.normal(1.0, 0.15))

        order_records.append({
            'merchant_id': m['merchant_id'],
            'order_month': month_date.strftime('%Y-%m-01'),
            'total_orders': monthly_orders,
            'total_gmv': round(monthly_orders * avg_basket, 2),
            'avg_basket_size': round(avg_basket, 2),
            'cancelled_orders': max(0, int(monthly_orders * np.random.beta(2, 20))),
            'avg_delivery_time_min': np.clip(np.random.normal(35, 10), 15, 90),
            'customer_rating_avg': np.clip(np.random.normal(4.2, 0.5), 1.0, 5.0),
            '_ingested_at': datetime.now().isoformat()
        })

orders_bronze = pd.DataFrame(order_records)

print(f"  merchants_bronze:   {len(merchants_bronze):,} rows")
print(f"  engagement_bronze:  {len(engagement_bronze):,} rows")
print(f"  orders_bronze:      {len(orders_bronze):,} rows")

# Save Bronze
merchants_bronze.to_parquet(os.path.join(OUTPUT_DIR, 'bronze_merchants.parquet'), index=False)
engagement_bronze.to_parquet(os.path.join(OUTPUT_DIR, 'bronze_engagement.parquet'), index=False)
orders_bronze.to_parquet(os.path.join(OUTPUT_DIR, 'bronze_orders.parquet'), index=False)

# ── SILVER LAYER ──
# Prod: DLT with expectations, SCD2 on merchant dims, type casting + dedup

print("\n[SILVER] Applying transformations and validation...")

# --- Silver: Merchants ---
merchants_silver = merchants_bronze.copy()
merchants_silver['onboard_date'] = pd.to_datetime(merchants_silver['onboard_date'])
merchants_silver['tenure_days'] = (pd.Timestamp(BASE_DATE + timedelta(days=365)) - merchants_silver['onboard_date']).dt.days
merchants_silver['tenure_months'] = (merchants_silver['tenure_days'] / 30).astype(int)
merchants_silver['delivery_radius_bucket'] = merchants_silver['delivery_radius_miles'].map({
    3: 'narrow (3mi)', 5: 'standard (5mi)', 7: 'wide (7mi)', 10: 'extra_wide (10mi)'
})
# Drop ingestion metadata columns for Silver
merchants_silver = merchants_silver.drop(columns=['_ingested_at', '_source_file'])

# Data quality checks (simulating DLT expectations)
dq_checks = {
    'no_null_merchant_id': merchants_silver['merchant_id'].notna().all(),
    'valid_vertical': merchants_silver['vertical'].isin(verticals).all(),
    'valid_region': merchants_silver['region'].isin(regions).all(),
    'positive_tenure': (merchants_silver['tenure_days'] > 0).all(),
    'unique_merchant_id': merchants_silver['merchant_id'].nunique() == len(merchants_silver),
}
print(f"  Data quality checks: {sum(dq_checks.values())}/{len(dq_checks)} passed")
for check, passed in dq_checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"    {status} {check}")

# --- Silver: Engagement (latest snapshot per merchant + historical) ---
engagement_silver = engagement_bronze.copy()
engagement_silver['snapshot_month'] = pd.to_datetime(engagement_silver['snapshot_month'])
engagement_silver['menu_completeness_pct'] = engagement_silver['menu_completeness_pct'].round(3)
engagement_silver['avg_response_time_hours'] = engagement_silver['avg_response_time_hours'].round(2)
engagement_silver = engagement_silver.drop(columns=['_ingested_at'])

# Latest engagement snapshot per merchant
latest_engagement = (
    engagement_silver
    .sort_values('snapshot_month')
    .groupby('merchant_id')
    .tail(1)
    .reset_index(drop=True)
)

# First promotion date per merchant (for cohort analysis)
# Generate realistic promo timing RELATIVE TO ONBOARD DATE
# Distribution: 15% early (0-30d), 20% mid (31-90d), 25% late (91-180d),
#               30% very late (180d+), 10% never promoted
np.random.seed(314)
promo_timing_records = []
for _, m in merchants_silver.iterrows():
    cohort_draw = np.random.random()
    if cohort_draw < 0.15:      # early adopters
        days_offset = np.random.randint(1, 31)
    elif cohort_draw < 0.35:    # mid adopters
        days_offset = np.random.randint(31, 91)
    elif cohort_draw < 0.60:    # late adopters
        days_offset = np.random.randint(91, 181)
    elif cohort_draw < 0.90:    # very late
        days_offset = np.random.randint(181, 500)
    else:                        # never promoted
        days_offset = None

    if days_offset is not None:
        promo_timing_records.append({
            'merchant_id': m['merchant_id'],
            'first_promo_date': m['onboard_date'] + timedelta(days=days_offset)
        })

promo_dates = pd.DataFrame(promo_timing_records)

# --- Silver: Orders ---
orders_silver = orders_bronze.copy()
orders_silver['order_month'] = pd.to_datetime(orders_silver['order_month'])
orders_silver['cancellation_rate'] = (
    orders_silver['cancelled_orders'] / orders_silver['total_orders']
).round(4)
orders_silver = orders_silver.drop(columns=['_ingested_at'])

print(f"  merchants_silver:   {len(merchants_silver):,} rows")
print(f"  engagement_silver:  {len(engagement_silver):,} rows (full history)")
print(f"  latest_engagement:  {len(latest_engagement):,} rows (current snapshot)")
print(f"  orders_silver:      {len(orders_silver):,} rows")

# Save Silver
merchants_silver.to_parquet(os.path.join(OUTPUT_DIR, 'silver_merchants.parquet'), index=False)
engagement_silver.to_parquet(os.path.join(OUTPUT_DIR, 'silver_engagement.parquet'), index=False)
orders_silver.to_parquet(os.path.join(OUTPUT_DIR, 'silver_orders.parquet'), index=False)

# ── GOLD LAYER ──
# Prod: materialized Delta with Z-ORDER on merchant_id, refreshed daily 6am PT
# Downstream: Tableau/Looker dashboards + ad-hoc SQL

print("\n[GOLD] Building fact and dimension tables...")

# --- Gold: Merchant Performance Fact Table ---
# Aggregate 12 months of orders per merchant
merchant_agg = (
    orders_silver
    .groupby('merchant_id')
    .agg(
        total_orders_12mo=('total_orders', 'sum'),
        total_gmv_12mo=('total_gmv', 'sum'),
        avg_monthly_gmv=('total_gmv', 'mean'),
        avg_monthly_orders=('total_orders', 'mean'),
        avg_basket_size=('avg_basket_size', 'mean'),
        avg_delivery_time=('avg_delivery_time_min', 'mean'),
        avg_customer_rating=('customer_rating_avg', 'mean'),
        avg_cancellation_rate=('cancellation_rate', 'mean'),
        months_active=('order_month', 'nunique'),
        # Growth metrics
        first_month_gmv=('total_gmv', 'first'),
        last_month_gmv=('total_gmv', 'last'),
    )
    .reset_index()
)
merchant_agg['gmv_growth_rate'] = (
    (merchant_agg['last_month_gmv'] - merchant_agg['first_month_gmv'])
    / merchant_agg['first_month_gmv'].clip(lower=1)
).round(4)

# Join merchant attributes + latest engagement + promo cohorts
gold_merchants = (
    merchants_silver
    .merge(merchant_agg, on='merchant_id', how='left')
    .merge(latest_engagement[['merchant_id', 'promotions_active', 'photos_uploaded',
                               'menu_completeness_pct', 'avg_response_time_hours',
                               'num_menu_items']], on='merchant_id', how='left')
    .merge(promo_dates, on='merchant_id', how='left')
)

# ── Derived Features ──

# Engagement score (0-5 scale based on behavioral signals)
gold_merchants['engagement_score'] = (
    gold_merchants['promotions_active'].astype(int) +
    gold_merchants['photos_uploaded'].astype(int) +
    (gold_merchants['menu_completeness_pct'] >= 0.80).astype(int) +
    (gold_merchants['avg_response_time_hours'] < 2.0).astype(int) +
    (gold_merchants['delivery_radius_miles'] >= 5).astype(int)
)

# Promotion timing cohort (needed for high-growth calibration below)
gold_merchants['days_to_first_promo'] = (
    gold_merchants['first_promo_date'] - gold_merchants['onboard_date']
).dt.days

gold_merchants['promo_cohort'] = pd.cut(
    gold_merchants['days_to_first_promo'],
    bins=[-1, 30, 90, 180, float('inf')],
    labels=['early_0_30d', 'mid_31_90d', 'late_91_180d', 'very_late_180d+'],
    right=True
).astype(str)
gold_merchants.loc[gold_merchants['first_promo_date'].isna(), 'promo_cohort'] = 'never_promoted'

# High-growth classification using CALIBRATED LOG-ODDS formulation
# This embeds realistic driver effects so the logistic regression in
# Notebook 2 can recover them. Odds ratios are calibrated to industry
# benchmarks from DoorDash's public merchant success research.
#
# True odds ratios (what we want the model to recover):
#   Promo activation:    2.3x  (β = ln(2.3) ≈ 0.83)
#   Photo upload:        1.6x  (β = ln(1.6) ≈ 0.47)
#   Wide radius (≥5mi):  1.5x  (β = ln(1.5) ≈ 0.41)
#   Complete menu (≥80%): 1.3x (β = ln(1.3) ≈ 0.26)
#   Fast response (<2h):  1.2x (β = ln(1.2) ≈ 0.18)
#   Early promo (0-30d):  1.8x (β = ln(1.8) ≈ 0.59) — cohort effect

from scipy.special import expit
log_odds = (
    -1.5  # intercept (baseline ~18% high-growth rate)
    + 0.83 * gold_merchants['promotions_active'].astype(float)
    + 0.47 * gold_merchants['photos_uploaded'].astype(float)
    + 0.41 * (gold_merchants['delivery_radius_miles'] >= 5).astype(float)
    + 0.26 * (gold_merchants['menu_completeness_pct'] >= 0.80).astype(float)
    + 0.18 * (gold_merchants['avg_response_time_hours'] < 2.0).astype(float)
    + 0.59 * (gold_merchants['days_to_first_promo'].fillna(999) <= 30).astype(float)
    + np.random.normal(0, 0.5, len(gold_merchants))  # noise for realistic AUC
)
growth_prob = expit(log_odds)
gold_merchants['growth_probability'] = growth_prob
gold_merchants['is_high_growth'] = np.random.binomial(1, growth_prob).astype(bool)

print(f"  High-growth rate: {gold_merchants['is_high_growth'].mean():.1%}")
print(f"  (Calibrated with log-odds; target OR for promos ≈ 2.3x)")

# Segmentation matrix (GMV potential × Engagement level)
gmv_median = gold_merchants['avg_monthly_gmv'].median()
gold_merchants['gmv_tier'] = np.where(gold_merchants['avg_monthly_gmv'] >= gmv_median, 'high_gmv', 'low_gmv')
gold_merchants['engagement_tier'] = np.where(gold_merchants['engagement_score'] >= 3, 'high_engagement', 'low_engagement')
gold_merchants['segment'] = gold_merchants.apply(
    lambda r: {
        ('high_gmv', 'low_engagement'): 'ACTIVATE',
        ('high_gmv', 'high_engagement'): 'PROTECT',
        ('low_gmv', 'low_engagement'): 'MONITOR',
        ('low_gmv', 'high_engagement'): 'NURTURE'
    }.get((r['gmv_tier'], r['engagement_tier']), 'UNKNOWN'),
    axis=1
)

# A/B test eligibility flag
# Eligible: ACTIVATE segment merchants who are NOT already high-growth
gold_merchants['ab_test_eligible'] = (
    (gold_merchants['segment'] == 'ACTIVATE') &
    (~gold_merchants['is_high_growth'])
)

# CUPED pre-experiment covariate (prior-period GMV for variance reduction)
# Uses first 6 months as pre-period, last 6 as experiment period
pre_period = orders_silver[orders_silver['order_month'] < '2024-07-01']
pre_period_gmv = pre_period.groupby('merchant_id')['total_gmv'].mean().reset_index()
pre_period_gmv.columns = ['merchant_id', 'cuped_pre_gmv']
gold_merchants = gold_merchants.merge(pre_period_gmv, on='merchant_id', how='left')

print(f"  gold_merchants: {len(gold_merchants):,} rows, {len(gold_merchants.columns)} columns")
print(f"\n  Segment distribution:")
print(gold_merchants['segment'].value_counts().to_string(header=False))
print(f"\n  Promo cohort distribution:")
print(gold_merchants['promo_cohort'].value_counts().to_string(header=False))
print(f"\n  A/B test eligible: {gold_merchants['ab_test_eligible'].sum():,} merchants")

# Save Gold
gold_merchants.to_parquet(os.path.join(OUTPUT_DIR, 'gold_merchants.parquet'), index=False)
orders_silver.to_parquet(os.path.join(OUTPUT_DIR, 'gold_orders_monthly.parquet'), index=False)
engagement_silver.to_parquet(os.path.join(OUTPUT_DIR, 'gold_engagement_history.parquet'), index=False)

# ── PIPELINE SUMMARY ──
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)

catalog = {
    'pipeline_run': datetime.now().isoformat(),
    'architecture': 'Medallion (Bronze → Silver → Gold)',
    'tables': {
        'bronze': {
            'merchants': {'rows': len(merchants_bronze), 'cols': len(merchants_bronze.columns)},
            'engagement': {'rows': len(engagement_bronze), 'cols': len(engagement_bronze.columns)},
            'orders': {'rows': len(orders_bronze), 'cols': len(orders_bronze.columns)},
        },
        'silver': {
            'merchants': {'rows': len(merchants_silver), 'cols': len(merchants_silver.columns)},
            'engagement': {'rows': len(engagement_silver), 'cols': len(engagement_silver.columns)},
            'orders': {'rows': len(orders_silver), 'cols': len(orders_silver.columns)},
        },
        'gold': {
            'merchants': {'rows': len(gold_merchants), 'cols': len(gold_merchants.columns)},
        }
    },
    'data_quality': dq_checks,
    'key_metrics': {
        'total_merchants': int(len(gold_merchants)),
        'active_merchants': int(gold_merchants['is_active'].sum()),
        'total_12mo_gmv': float(gold_merchants['total_gmv_12mo'].sum()),
        'ab_test_eligible': int(gold_merchants['ab_test_eligible'].sum()),
    }
}

with open(os.path.join(OUTPUT_DIR, 'pipeline_catalog.json'), 'w') as f:
    json.dump(catalog, f, indent=2, default=str)

print(f"\nData files saved to: {OUTPUT_DIR}/")
print(f"Total 12-month GMV: ${catalog['key_metrics']['total_12mo_gmv']:,.0f}")
print(f"A/B test eligible pool: {catalog['key_metrics']['ab_test_eligible']:,} merchants")
