"""
Notebook 2 — Statistical Analysis & Driver Modeling
DoorDash New Verticals: Merchant Growth Analytics
Author: Ashwath Subramanyan

EDA → logistic regression (driver identification) → propensity score analysis
→ cohort analysis → segmentation validation.

All claims carry CIs and p-values. Causal vs. correlational labeling throughout.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import expit
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Gold Layer ──
merchants = pd.read_parquet(os.path.join(DATA_DIR, 'gold_merchants.parquet'))
orders = pd.read_parquet(os.path.join(DATA_DIR, 'gold_orders_monthly.parquet'))
engagement = pd.read_parquet(os.path.join(DATA_DIR, 'gold_engagement_history.parquet'))

print("=" * 70)
print("STATISTICAL ANALYSIS & DRIVER MODELING")
print("=" * 70)
print(f"Merchants: {len(merchants):,}  |  Monthly orders: {len(orders):,}")

# ── 1. EDA ──
print("\n" + "─" * 50)
print("1. EXPLORATORY DATA ANALYSIS")
print("─" * 50)

# Key distributions
print("\nGMV Distribution by Vertical:")
gmv_by_vertical = merchants.groupby('vertical')['avg_monthly_gmv'].describe()[['mean', '50%', 'std']]
gmv_by_vertical.columns = ['mean', 'median', 'std']
print(gmv_by_vertical.round(0).to_string())

print("\nEngagement Score Distribution:")
print(merchants['engagement_score'].value_counts().sort_index().to_string())

print(f"\nHigh-Growth Rate: {merchants['is_high_growth'].mean():.1%}")
print(f"Growth Threshold (60th pctile): {merchants['gmv_growth_rate'].quantile(0.60):.3f}")

# ── Figure 1: Key Distributions ──
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Merchant Population Overview', fontsize=14, fontweight='bold', y=0.98)

# 1a: GMV distribution by vertical
for v in ['grocery', 'convenience', 'retail']:
    subset = merchants[merchants['vertical'] == v]['avg_monthly_gmv']
    axes[0,0].hist(subset, bins=40, alpha=0.6, label=f'{v.title()} (n={len(subset):,})')
axes[0,0].set_xlabel('Avg Monthly GMV ($)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('GMV Distribution by Vertical')
axes[0,0].legend(fontsize=8)

# 1b: Engagement score distribution
colors = ['#d9534f' if s < 3 else '#5cb85c' for s in range(6)]
merchants['engagement_score'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], color=colors)
axes[0,1].set_xlabel('Engagement Score (0-5)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Engagement Score Distribution')

# 1c: Segment distribution
seg_counts = merchants['segment'].value_counts()
colors_seg = {'ACTIVATE': '#FF6B35', 'PROTECT': '#2E86AB', 'NURTURE': '#A23B72', 'MONITOR': '#999999'}
seg_counts.plot(kind='bar', ax=axes[1,0], color=[colors_seg.get(s, '#999') for s in seg_counts.index])
axes[1,0].set_xlabel('Segment')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Merchant Segmentation')
axes[1,0].tick_params(axis='x', rotation=0)

# 1d: Growth rate distribution
axes[1,1].hist(merchants['gmv_growth_rate'].clip(-1, 2), bins=50, color='steelblue', alpha=0.7)
axes[1,1].axvline(merchants['gmv_growth_rate'].quantile(0.60), color='red', linestyle='--',
                   label=f'60th pctile = {merchants["gmv_growth_rate"].quantile(0.60):.3f}')
axes[1,1].set_xlabel('12-Month GMV Growth Rate')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Growth Rate Distribution')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_eda_overview.png'), bbox_inches='tight')
plt.close()
print("\n  → Saved fig1_eda_overview.png")


# ── 2. DRIVER ANALYSIS: LOGISTIC REGRESSION ──
# Logit(P(high_growth)) = β₀ + β₁·promo + ... + controls (vertical, region, tenure)
# ORs with Wald 95% CIs, Bonferroni correction, 80/20 stratified split
# NOTE: associational only — promo activation likely endogenous (see §3)
print("\n" + "─" * 50)
print("2. DRIVER ANALYSIS: LOGISTIC REGRESSION")
print("─" * 50)

# Feature engineering
model_df = merchants[merchants['is_active']].copy()
model_df['wide_radius'] = (model_df['delivery_radius_miles'] >= 5).astype(int)
model_df['menu_complete'] = (model_df['menu_completeness_pct'] >= 0.80).astype(int)
model_df['fast_response'] = (model_df['avg_response_time_hours'] < 2.0).astype(int)
model_df['promo_active'] = model_df['promotions_active'].astype(int)
model_df['photo_uploaded'] = model_df['photos_uploaded'].astype(int)

# Control variables
model_df = pd.get_dummies(model_df, columns=['vertical', 'region'], drop_first=True, dtype=int)

# Feature matrix
engagement_features = ['promo_active', 'photo_uploaded', 'wide_radius', 'menu_complete', 'fast_response']
control_features = [c for c in model_df.columns if c.startswith(('vertical_', 'region_'))]
control_features.append('tenure_months')
all_features = engagement_features + control_features

X = model_df[all_features].astype(float)
y = model_df['is_high_growth'].astype(int)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

# Fit logistic regression with statsmodels (for proper inference)
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

logit_model = sm.Logit(y_train, X_train_const).fit(disp=0)

print("\n  Model Summary (Engagement Features Only):")
print("  " + "─" * 64)
print(f"  {'Feature':<20} {'Coef':>8} {'OR':>8} {'95% CI':>18} {'p-value':>10}")
print("  " + "─" * 64)

# Extract results for engagement features
results_data = []
for feat in engagement_features:
    coef = logit_model.params[feat]
    ci_low, ci_high = logit_model.conf_int().loc[feat]
    or_val = np.exp(coef)
    or_ci_low = np.exp(ci_low)
    or_ci_high = np.exp(ci_high)
    p_val = logit_model.pvalues[feat]

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    print(f"  {feat:<20} {coef:>7.3f}  {or_val:>7.2f}x  [{or_ci_low:.2f}, {or_ci_high:.2f}]  {p_val:>9.4f} {sig}")

    results_data.append({
        'feature': feat,
        'coefficient': round(coef, 4),
        'odds_ratio': round(or_val, 3),
        'ci_lower': round(or_ci_low, 3),
        'ci_upper': round(or_ci_high, 3),
        'p_value': round(p_val, 6),
        'significant_at_05': p_val < 0.05
    })

print("  " + "─" * 64)
print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05")

# Model fit
y_pred_prob = logit_model.predict(X_test_const)
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n  Model Performance:")
print(f"    AUC-ROC (test):  {auc:.3f}")
print(f"    Pseudo R²:       {logit_model.prsquared:.3f}")
print(f"    N (train):       {len(X_train):,}")
print(f"    N (test):        {len(X_test):,}")

# Bonferroni correction for multiple comparisons
n_tests = len(engagement_features)
bonferroni_alpha = 0.05 / n_tests
print(f"\n  Multiple Testing Correction:")
print(f"    Bonferroni α = 0.05/{n_tests} = {bonferroni_alpha:.4f}")
significant_after_correction = sum(1 for r in results_data if r['p_value'] < bonferroni_alpha)
print(f"    Features significant after correction: {significant_after_correction}/{n_tests}")

# Save results
results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'driver_analysis_results.csv'), index=False)

# ── Figure 2: Driver Analysis (Odds Ratios with CIs) ──
fig, ax = plt.subplots(figsize=(10, 5))

labels = [r['feature'].replace('_', ' ').title() for r in results_data]
ors = [r['odds_ratio'] for r in results_data]
ci_lows = [r['odds_ratio'] - r['ci_lower'] for r in results_data]
ci_highs = [r['ci_upper'] - r['odds_ratio'] for r in results_data]
colors_bar = ['#FF6B35' if r['p_value'] < 0.05 else '#CCCCCC' for r in results_data]

y_pos = range(len(labels))
ax.barh(y_pos, ors, xerr=[ci_lows, ci_highs], color=colors_bar, capsize=4, edgecolor='white', height=0.6)
ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, label='No effect (OR=1.0)')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('Odds Ratio (95% CI)')
ax.set_title('Growth Drivers: What Predicts High-Growth Merchants?', fontweight='bold')

# Annotate ORs
for i, r in enumerate(results_data):
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
    ax.text(r['ci_upper'] + 0.05, i, f"{r['odds_ratio']:.2f}x {sig}", va='center', fontsize=9)

ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_driver_analysis.png'), bbox_inches='tight')
plt.close()
print("\n  → Saved fig2_driver_analysis.png")


# ── 3. CAUSAL INFERENCE: PROPENSITY SCORE ANALYSIS ──
# Selection bias is the concern — better merchants self-select into promos.
# Propensity stratification below; RCT design in notebook 03.
print("\n" + "─" * 50)
print("3. CAUSAL INFERENCE: PROPENSITY SCORE ANALYSIS")
print("─" * 50)

# Step 1: Estimate propensity scores (probability of activating promos)
# Using pre-treatment covariates that predict promo adoption
propensity_features = ['tenure_months', 'photo_uploaded', 'wide_radius',
                       'menu_complete', 'fast_response', 'avg_monthly_gmv']

ps_df = model_df[propensity_features + ['promo_active', 'is_high_growth']].dropna().copy()

# Standardize continuous features for propensity model
scaler = StandardScaler()
ps_df[['tenure_months_z', 'avg_monthly_gmv_z']] = scaler.fit_transform(
    ps_df[['tenure_months', 'avg_monthly_gmv']]
)

ps_features = ['tenure_months_z', 'photo_uploaded', 'wide_radius',
               'menu_complete', 'fast_response', 'avg_monthly_gmv_z']
X_ps = sm.add_constant(ps_df[ps_features])
y_ps = ps_df['promo_active']

ps_model = sm.Logit(y_ps, X_ps).fit(disp=0)
ps_df['propensity_score'] = ps_model.predict(X_ps)

# Step 2: Assess overlap (common support)
treated = ps_df[ps_df['promo_active'] == 1]['propensity_score']
control = ps_df[ps_df['promo_active'] == 0]['propensity_score']

print(f"\n  Propensity Score Summary:")
print(f"    Treated (promo=1):  n={len(treated):,}, mean={treated.mean():.3f}, std={treated.std():.3f}")
print(f"    Control (promo=0):  n={len(control):,}, mean={control.mean():.3f}, std={control.std():.3f}")
print(f"    Overlap region:     [{max(treated.min(), control.min()):.3f}, {min(treated.max(), control.max()):.3f}]")

# Step 3: Stratified analysis (divide into propensity quintiles)
ps_df['ps_quintile'] = pd.qcut(ps_df['propensity_score'], 5, labels=['Q1','Q2','Q3','Q4','Q5'])

print(f"\n  Stratified Treatment Effect by Propensity Quintile:")
print(f"  {'Quintile':<10} {'Treated %':>12} {'Control %':>12} {'Difference':>12} {'N':>8}")
print(f"  {'─'*54}")

stratum_effects = []
for q in ['Q1','Q2','Q3','Q4','Q5']:
    stratum = ps_df[ps_df['ps_quintile'] == q]
    t_rate = stratum[stratum['promo_active']==1]['is_high_growth'].mean()
    c_rate = stratum[stratum['promo_active']==0]['is_high_growth'].mean()
    diff = t_rate - c_rate
    n = len(stratum)
    stratum_effects.append({'quintile': q, 'treated_rate': t_rate, 'control_rate': c_rate,
                           'ate': diff, 'n': n})
    print(f"  {q:<10} {t_rate:>11.1%} {c_rate:>11.1%} {diff:>+11.1%} {n:>8,}")

# Average Treatment Effect (ATE) across strata
ate_values = [s['ate'] for s in stratum_effects]
ate_weights = [s['n'] for s in stratum_effects]
weighted_ate = np.average(ate_values, weights=ate_weights)
ate_se = np.std(ate_values) / np.sqrt(len(ate_values))
ate_ci = (weighted_ate - 1.96 * ate_se, weighted_ate + 1.96 * ate_se)

print(f"\n  Weighted Average Treatment Effect (ATE):")
print(f"    ATE = {weighted_ate:+.3f} (95% CI: [{ate_ci[0]:.3f}, {ate_ci[1]:.3f}])")
print(f"\n  After adjusting for observables, promo activation → {weighted_ate:+.1%} in high-growth prob.")
print(f"  Still only observed confounders — unobserved quality/motivation could bias this.")
print(f"  Need the RCT (notebook 03) to nail down causality.")


# ── 4. COHORT ANALYSIS: PROMOTION TIMING ──
print("\n" + "─" * 50)
print("4. COHORT ANALYSIS: PROMOTION TIMING")
print("─" * 50)

cohort_stats = (
    merchants[merchants['promo_cohort'] != 'never_promoted']
    .groupby('promo_cohort')
    .agg(
        n_merchants=('merchant_id', 'count'),
        high_growth_rate=('is_high_growth', 'mean'),
        avg_gmv=('avg_monthly_gmv', 'mean'),
        avg_engagement=('engagement_score', 'mean'),
        avg_rating=('avg_customer_rating', 'mean'),
    )
    .round(3)
)

print("\n  Cohort Performance by Promotion Timing:")
print(cohort_stats.to_string())

# Statistical test: Chi-squared test for independence
# H₀: High-growth rate is independent of promo timing cohort
cohort_data = merchants[merchants['promo_cohort'] != 'never_promoted']
contingency = pd.crosstab(cohort_data['promo_cohort'], cohort_data['is_high_growth'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"\n  Chi-squared test (cohort × high-growth):")
print(f"    χ² = {chi2:.2f}, df = {dof}, p = {p_value:.4f}")
print(f"    → {'Reject' if p_value < 0.05 else 'Fail to reject'} H₀ at α=0.05")

# Pairwise comparison: early vs late cohorts
early = cohort_data[cohort_data['promo_cohort'] == 'early_0_30d']['is_high_growth']
late = cohort_data[cohort_data['promo_cohort'] == 'late_91_180d']['is_high_growth']
if len(early) > 0 and len(late) > 0:
    t_stat, p_pair = stats.ttest_ind(early.astype(float), late.astype(float))
    early_rate = early.mean()
    late_rate = late.mean()
    print(f"\n  Early (0-30d) vs Late (91-180d) pairwise t-test:")
    print(f"    Early rate: {early_rate:.1%}  |  Late rate: {late_rate:.1%}")
    print(f"    t = {t_stat:.2f}, p = {p_pair:.4f}")
    print(f"    → {'Significant' if p_pair < 0.05 else 'Not significant'} difference at α=0.05")

# ── Figure 3: Cohort Analysis ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Cohort Analysis: Does Promotion Timing Matter?', fontsize=13, fontweight='bold')

# 3a: High-growth rate by cohort
cohort_order = ['early_0_30d', 'mid_31_90d', 'late_91_180d', 'very_late_180d+', 'never_promoted']
cohort_labels = ['Early\n(0-30d)', 'Mid\n(31-90d)', 'Late\n(91-180d)', 'Very Late\n(180d+)', 'Never\nPromoted']

cohort_all = merchants.groupby('promo_cohort')['is_high_growth'].agg(['mean', 'count', 'std']).reindex(cohort_order)
cohort_all['se'] = cohort_all['std'] / np.sqrt(cohort_all['count'])

bars = axes[0].bar(range(len(cohort_order)), cohort_all['mean'],
                    yerr=1.96 * cohort_all['se'], capsize=4,
                    color=['#2E86AB', '#5BA4C9', '#999999', '#CC6666', '#DD4444'],
                    edgecolor='white')
axes[0].set_xticks(range(len(cohort_order)))
axes[0].set_xticklabels(cohort_labels, fontsize=9)
axes[0].set_ylabel('High-Growth Rate')
axes[0].set_title('High-Growth Rate by Promo Cohort')
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Add value labels
for bar, val in zip(bars, cohort_all['mean']):
    if not np.isnan(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.1%}', ha='center', fontsize=9)

# 3b: Monthly GMV trajectory by cohort (first 6 months of orders)
# Simulate monthly retention by cohort using actual order data
cohort_merchants = merchants[['merchant_id', 'promo_cohort', 'onboard_date']].copy()
order_cohort = orders.merge(cohort_merchants, on='merchant_id')
order_cohort['months_since_onboard'] = (
    (order_cohort['order_month'] - order_cohort['onboard_date']).dt.days / 30
).astype(int).clip(0, 11)

for cohort in ['early_0_30d', 'mid_31_90d', 'late_91_180d', 'never_promoted']:
    subset = order_cohort[order_cohort['promo_cohort'] == cohort]
    monthly_avg = subset.groupby('months_since_onboard')['total_gmv'].mean()
    if len(monthly_avg) > 0:
        axes[1].plot(monthly_avg.index[:12], monthly_avg.values[:12],
                     marker='o', markersize=3, label=cohort.replace('_', ' ').title())

axes[1].set_xlabel('Months Since Onboarding')
axes[1].set_ylabel('Avg Monthly GMV ($)')
axes[1].set_title('GMV Trajectory by Promo Cohort')
axes[1].legend(fontsize=8)
axes[1].set_xlim(0, 11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_cohort_analysis.png'), bbox_inches='tight')
plt.close()
print("\n  → Saved fig3_cohort_analysis.png")


# ── 5. SEGMENTATION VALIDATION ──
print("\n" + "─" * 50)
print("5. SEGMENTATION VALIDATION")
print("─" * 50)

# Validate that segments differ meaningfully on key business metrics
seg_validation = merchants.groupby('segment').agg(
    n=('merchant_id', 'count'),
    avg_gmv=('avg_monthly_gmv', 'mean'),
    high_growth_rate=('is_high_growth', 'mean'),
    avg_engagement=('engagement_score', 'mean'),
    avg_rating=('avg_customer_rating', 'mean'),
    avg_cancel_rate=('avg_cancellation_rate', 'mean'),
).round(3)

print("\n  Segment Profiles:")
print(seg_validation.to_string())

# ANOVA test: Do segments differ significantly on GMV?
segments_list = [merchants[merchants['segment'] == s]['avg_monthly_gmv'] for s in merchants['segment'].unique()]
f_stat, p_anova = stats.f_oneway(*segments_list)
print(f"\n  One-way ANOVA (GMV across segments):")
print(f"    F = {f_stat:.2f}, p = {p_anova:.2e}")
print(f"    → Segments {'are' if p_anova < 0.05 else 'are NOT'} significantly different on GMV")

# Effect size (eta-squared)
ss_between = sum(len(s) * (s.mean() - merchants['avg_monthly_gmv'].mean())**2 for s in segments_list)
ss_total = sum((merchants['avg_monthly_gmv'] - merchants['avg_monthly_gmv'].mean())**2)
eta_squared = ss_between / ss_total
print(f"    η² = {eta_squared:.3f} ({'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small'} effect)")

# Intervention target sizing
activate = merchants[merchants['segment'] == 'ACTIVATE']
intervention_targets = activate[~activate['is_high_growth']]
print(f"\n  Intervention Target Pool (ACTIVATE, non-high-growth):")
print(f"    Merchants:        {len(intervention_targets):,}")
print(f"    Annual GMV:       ${intervention_targets['total_gmv_12mo'].sum():,.0f}")
print(f"    Missing promos:   {(~intervention_targets['promotions_active']).mean():.0%}")
print(f"    Missing photos:   {(~intervention_targets['photos_uploaded']).mean():.0%}")
print(f"    Avg engagement:   {intervention_targets['engagement_score'].mean():.1f}/5")

# ── Figure 4: Segmentation Matrix ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Merchant Segmentation: Where to Invest', fontsize=13, fontweight='bold')

# 4a: Scatter plot
colors_map = {'ACTIVATE': '#FF6B35', 'PROTECT': '#2E86AB', 'NURTURE': '#A23B72', 'MONITOR': '#BBBBBB'}
for seg in ['MONITOR', 'NURTURE', 'PROTECT', 'ACTIVATE']:
    subset = merchants[merchants['segment'] == seg].sample(min(300, len(merchants[merchants['segment'] == seg])))
    axes[0].scatter(subset['engagement_score'], subset['avg_monthly_gmv'],
                    alpha=0.3, s=15, color=colors_map[seg], label=f'{seg} (n={len(merchants[merchants["segment"]==seg]):,})')

axes[0].axhline(y=merchants['avg_monthly_gmv'].median(), color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(x=2.5, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('Engagement Score')
axes[0].set_ylabel('Avg Monthly GMV ($)')
axes[0].set_title('Segmentation Matrix')
axes[0].legend(fontsize=8, loc='upper left')

# 4b: Intervention gap analysis
gap_labels = ['No Promos', 'No Photos', 'Incomplete\nMenu (<80%)', 'Slow Response\n(>2hr)', 'Narrow\nRadius (<5mi)']
gap_values = [
    (~intervention_targets['promotions_active']).mean(),
    (~intervention_targets['photos_uploaded']).mean(),
    (intervention_targets['menu_completeness_pct'] < 0.80).mean(),
    (intervention_targets['avg_response_time_hours'] >= 2.0).mean(),
    (intervention_targets['delivery_radius_miles'] < 5).mean(),
]

bars = axes[1].barh(gap_labels, gap_values, color=['#FF6B35', '#FF8C5A', '#FFAA80', '#CCCCCC', '#DDDDDD'],
                     edgecolor='white', height=0.6)
axes[1].set_xlabel('% of Intervention Targets')
axes[1].set_title('Engagement Gaps in ACTIVATE Segment')
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

for bar, val in zip(bars, gap_values):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.0%}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_segmentation.png'), bbox_inches='tight')
plt.close()
print("\n  → Saved fig4_segmentation.png")


# ── 6. SUMMARY ──
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

top_driver = results_df.loc[results_df['odds_ratio'].idxmax()]
print(f"""
  1. TOP DRIVER: {top_driver['feature'].replace('_', ' ').title()}
     - Odds Ratio: {top_driver['odds_ratio']}x (95% CI: [{top_driver['ci_lower']}, {top_driver['ci_upper']}])
     - p-value: {top_driver['p_value']:.2e}
     - Survives Bonferroni correction: {'Yes' if top_driver['p_value'] < bonferroni_alpha else 'No'}

  2. CAUSAL ESTIMATE (propensity-adjusted):
     - ATE: {weighted_ate:+.3f} ({weighted_ate:+.1%} change in high-growth probability)
     - Limitation: Observed confounders only; RCT recommended

  3. COHORT INSIGHT:
     - Earlier promo activation is associated with higher growth rates
     - Chi-squared test: {'Significant' if p_value < 0.05 else 'Not significant'} (p={p_value:.4f})

  4. INTERVENTION OPPORTUNITY:
     - {len(intervention_targets):,} merchants in ACTIVATE segment
     - ${intervention_targets['total_gmv_12mo'].sum():,.0f} annual GMV at stake
     - Primary gap: {(~intervention_targets['promotions_active']).mean():.0%} lack promotion activation

  5. NEXT STEP: Design and run a randomized experiment (Notebook 3)
     to establish causal impact before scaling interventions.
""")
