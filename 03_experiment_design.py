"""
Notebook 3 — Experiment Design & A/B Test Framework
DoorDash New Verticals: Merchant Growth Analytics
Author: Ashwath Subramanyan

Designs the RCT to establish causal impact of promo activation on merchant GMV.
Covers: power analysis, stratified randomization, CUPED variance reduction,
guardrails, SRM detection, and a ship/iterate decision framework.

In prod this would run through DD's internal experimentation platform;
the stats methodology here is what I'd bring to that system.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

merchants = pd.read_parquet(os.path.join(DATA_DIR, 'gold_merchants.parquet'))

print("=" * 70)
print("EXPERIMENT DESIGN: Promotion Activation A/B Test")
print("=" * 70)


# ── 1. HYPOTHESIS & EXPERIMENT DESIGN ──
print("\n" + "─" * 50)
print("1. HYPOTHESIS & EXPERIMENT DESIGN")
print("─" * 50)

experiment_spec = {
    'name': 'Merchant Promo Activation Intervention',
    'hypothesis': {
        'H0': 'Proactive promotion outreach has no effect on merchant GMV growth',
        'H1': 'Proactive promotion outreach increases merchant GMV growth rate',
    },
    'primary_metric': 'avg_monthly_gmv (post-treatment, 90-day window)',
    'secondary_metrics': [
        'total_orders (post-treatment)',
        'customer_rating_avg',
        'cancellation_rate',
    ],
    'guardrail_metrics': [
        'merchant_churn_rate (must not increase >2pp)',
        'customer_complaint_rate (must not increase >1pp)',
        'avg_delivery_time (must not increase >5min)',
    ],
    'treatment': 'Proactive outreach: dedicated MSM call + 50% co-funded first promo + onboarding guide',
    'control': 'Business as usual (no proactive outreach)',
    'unit': 'Merchant (cluster by city to avoid spillover)',
    'duration': '90 days (with 14-day burn-in excluded from analysis)',
    'eligible_population': 'ACTIVATE segment, non-high-growth, active merchants',
}

for key, val in experiment_spec.items():
    if isinstance(val, dict):
        print(f"\n  {key}:")
        for k, v in val.items():
            print(f"    {k}: {v}")
    elif isinstance(val, list):
        print(f"\n  {key}:")
        for item in val:
            print(f"    - {item}")
    else:
        print(f"  {key}: {val}")


# ── 2. POWER ANALYSIS ──
print("\n" + "─" * 50)
print("2. POWER ANALYSIS")
print("─" * 50)

eligible = merchants[merchants['ab_test_eligible']].copy()
print(f"\n  Eligible pool: {len(eligible):,} merchants")

# Estimate baseline metrics from eligible population
baseline_gmv_mean = eligible['avg_monthly_gmv'].mean()
baseline_gmv_std = eligible['avg_monthly_gmv'].std()
print(f"  Baseline GMV:  μ = ${baseline_gmv_mean:,.0f}, σ = ${baseline_gmv_std:,.0f}")

# Power analysis parameters
alpha = 0.05        # Significance level
power = 0.80        # Statistical power (1 - β)
mde_pct = 0.08      # Minimum detectable effect (8% lift in GMV)
mde_abs = baseline_gmv_mean * mde_pct

# Sample size calculation (two-sample t-test)
# n = 2 * ((z_α/2 + z_β) * σ / δ)²
z_alpha = norm.ppf(1 - alpha/2)
z_beta = norm.ppf(power)

n_per_arm = int(np.ceil(2 * ((z_alpha + z_beta) * baseline_gmv_std / mde_abs) ** 2))

print(f"\n  Power Analysis Parameters:")
print(f"    α (significance):    {alpha}")
print(f"    Power (1-β):         {power}")
print(f"    MDE:                 {mde_pct:.0%} ({mde_pct*100:.0f}% lift = ${mde_abs:,.0f}/month)")
print(f"    Baseline σ:          ${baseline_gmv_std:,.0f}")
print(f"\n  Required Sample Size:")
print(f"    n per arm:           {n_per_arm:,}")
print(f"    Total:               {2*n_per_arm:,}")
print(f"    Eligible pool:       {len(eligible):,}")
print(f"    Feasibility:         {'OK — pool is large enough' if len(eligible) >= 2*n_per_arm else 'Insufficient — reduce MDE or accept lower power'}")

# Power curve: how power changes with sample size
sample_sizes = range(50, min(len(eligible)//2, 2000), 25)
powers = []
for n in sample_sizes:
    # Power = P(reject H0 | H1 true) = Φ(δ√(n/2)/σ - z_α/2)
    ncp = mde_abs * np.sqrt(n / 2) / baseline_gmv_std
    achieved_power = 1 - norm.cdf(z_alpha - ncp)
    powers.append(achieved_power)

# Power curve: how MDE changes with fixed n
if len(eligible) >= 2 * n_per_arm:
    actual_n = n_per_arm
else:
    actual_n = len(eligible) // 2

mde_range = np.arange(0.03, 0.20, 0.005)
powers_by_mde = []
for mde in mde_range:
    delta = baseline_gmv_mean * mde
    ncp = delta * np.sqrt(actual_n / 2) / baseline_gmv_std
    p = 1 - norm.cdf(z_alpha - ncp)
    powers_by_mde.append(p)

# ── Figure 5: Power Analysis ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('A/B Test Power Analysis', fontsize=13, fontweight='bold')

# 5a: Power vs sample size
axes[0].plot(sample_sizes, powers, color='#2E86AB', linewidth=2)
axes[0].axhline(y=0.80, color='red', linestyle='--', linewidth=0.8, label='80% power')
axes[0].axvline(x=n_per_arm, color='green', linestyle='--', linewidth=0.8, label=f'n={n_per_arm:,}')
axes[0].set_xlabel('Sample Size per Arm')
axes[0].set_ylabel('Statistical Power')
axes[0].set_title(f'Power Curve (MDE={mde_pct:.0%})')
axes[0].legend()
axes[0].set_ylim(0, 1.05)

# 5b: Power vs MDE (with actual n)
axes[1].plot([m*100 for m in mde_range], powers_by_mde, color='#FF6B35', linewidth=2)
axes[1].axhline(y=0.80, color='red', linestyle='--', linewidth=0.8, label='80% power')
axes[1].axvline(x=mde_pct*100, color='green', linestyle='--', linewidth=0.8, label=f'MDE={mde_pct:.0%}')
axes[1].set_xlabel('Minimum Detectable Effect (%)')
axes[1].set_ylabel('Statistical Power')
axes[1].set_title(f'Power vs. Effect Size (n={actual_n:,}/arm)')
axes[1].legend()
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_power_analysis.png'), bbox_inches='tight')
plt.close()
print("\n  → Saved fig5_power_analysis.png")


# ── 3. STRATIFIED RANDOMIZATION ──
print("\n" + "─" * 50)
print("3. STRATIFIED RANDOMIZATION")
print("─" * 50)

# Stratify by vertical × region to ensure balance on key dimensions
eligible['stratum'] = eligible['vertical'] + '_' + eligible['region']

# Random assignment within each stratum
np.random.seed(2024)
assignments = []
for stratum, group in eligible.groupby('stratum'):
    n = len(group)
    n_treatment = n // 2
    shuffled = group.sample(frac=1, random_state=2024)
    shuffled['assignment'] = ['treatment'] * n_treatment + ['control'] * (n - n_treatment)
    assignments.append(shuffled)

experiment_df = pd.concat(assignments)

# Balance check
print("\n  Balance Check (Treatment vs Control):")
print(f"  {'Metric':<25} {'Treatment':>12} {'Control':>12} {'p-value':>10}")
print(f"  {'─'*59}")

balance_vars = ['avg_monthly_gmv', 'tenure_months', 'engagement_score',
                'avg_customer_rating', 'avg_cancellation_rate']

treatment = experiment_df[experiment_df['assignment'] == 'treatment']
control = experiment_df[experiment_df['assignment'] == 'control']

all_balanced = True
for var in balance_vars:
    t_mean = treatment[var].mean()
    c_mean = control[var].mean()
    _, p = stats.ttest_ind(treatment[var].dropna(), control[var].dropna())
    balanced = "ok" if p > 0.05 else "IMBALANCED"
    if p <= 0.05:
        all_balanced = False
    print(f"  {var:<25} {t_mean:>11.2f} {c_mean:>11.2f} {p:>9.3f} {balanced}")

print(f"\n  Overall balance: {'All covariates balanced (p>0.05)' if all_balanced else 'Some imbalance detected — check strata'}")
print(f"  Treatment: {len(treatment):,}  |  Control: {len(control):,}")


# ── 4. CUPED VARIANCE REDUCTION ──
print("\n" + "─" * 50)
print("4. CUPED VARIANCE REDUCTION")
print("─" * 50)
print("""
  Y_cuped = Y - θ(X - E[X]),  θ = Cov(Y,X)/Var(X)
  Using pre-period GMV as covariate to squeeze out variance.
""")

# Use pre-period GMV (cuped_pre_gmv) as the covariate
# Simulate post-treatment outcomes for demonstration
np.random.seed(99)
experiment_df = experiment_df.copy()

# Simulated true treatment effect: 10% lift in GMV
true_effect = 0.10
noise = np.random.normal(0, baseline_gmv_std * 0.3, len(experiment_df))

experiment_df['post_gmv'] = experiment_df['avg_monthly_gmv'] + noise
experiment_df.loc[experiment_df['assignment'] == 'treatment', 'post_gmv'] += (
    baseline_gmv_mean * true_effect
)

# Standard analysis (no CUPED)
t_post = experiment_df[experiment_df['assignment'] == 'treatment']['post_gmv']
c_post = experiment_df[experiment_df['assignment'] == 'control']['post_gmv']
raw_diff = t_post.mean() - c_post.mean()
raw_se = np.sqrt(t_post.var()/len(t_post) + c_post.var()/len(c_post))
raw_ci = (raw_diff - 1.96*raw_se, raw_diff + 1.96*raw_se)
_, raw_p = stats.ttest_ind(t_post, c_post)

print(f"  Standard Analysis (no CUPED):")
print(f"    Effect:   ${raw_diff:,.0f} (95% CI: [${raw_ci[0]:,.0f}, ${raw_ci[1]:,.0f}])")
print(f"    SE:       ${raw_se:,.0f}")
print(f"    p-value:  {raw_p:.4f}")

# CUPED-adjusted analysis
X = experiment_df['cuped_pre_gmv'].values
Y = experiment_df['post_gmv'].values

theta = np.cov(Y, X)[0, 1] / np.var(X)
Y_cuped = Y - theta * (X - X.mean())
experiment_df['post_gmv_cuped'] = Y_cuped

t_cuped = experiment_df[experiment_df['assignment'] == 'treatment']['post_gmv_cuped']
c_cuped = experiment_df[experiment_df['assignment'] == 'control']['post_gmv_cuped']
cuped_diff = t_cuped.mean() - c_cuped.mean()
cuped_se = np.sqrt(t_cuped.var()/len(t_cuped) + c_cuped.var()/len(c_cuped))
cuped_ci = (cuped_diff - 1.96*cuped_se, cuped_diff + 1.96*cuped_se)
_, cuped_p = stats.ttest_ind(t_cuped, c_cuped)

print(f"\n  CUPED-Adjusted Analysis:")
print(f"    Effect:   ${cuped_diff:,.0f} (95% CI: [${cuped_ci[0]:,.0f}, ${cuped_ci[1]:,.0f}])")
print(f"    SE:       ${cuped_se:,.0f}")
print(f"    p-value:  {cuped_p:.4f}")

variance_reduction = 1 - (cuped_se**2 / raw_se**2)
print(f"\n  Variance Reduction: {variance_reduction:.1%}")
print(f"  θ (CUPED coefficient): {theta:.3f}")
print(f"  Correlation(Y, X): {np.corrcoef(Y, X)[0,1]:.3f}")
print(f"\n  CUPED cuts the CI width by {variance_reduction:.0%} — same sample, tighter estimate.")


# ── 5. GUARDRAILS & SRM ──
print("\n" + "─" * 50)
print("5. GUARDRAIL METRICS & SRM CHECK")
print("─" * 50)

# SRM (Sample Ratio Mismatch) detection
# Expected ratio: 50/50
n_t = len(treatment)
n_c = len(control)
n_total = n_t + n_c
expected_ratio = 0.5

# Chi-squared test for SRM
chi2_srm = (n_t - n_total * expected_ratio)**2 / (n_total * expected_ratio) + \
           (n_c - n_total * (1-expected_ratio))**2 / (n_total * (1-expected_ratio))
p_srm = 1 - stats.chi2.cdf(chi2_srm, df=1)

print(f"\n  Sample Ratio Mismatch (SRM) Check:")
print(f"    Treatment: {n_t:,}  |  Control: {n_c:,}")
print(f"    Observed ratio: {n_t/n_total:.4f}")
print(f"    Expected ratio: {expected_ratio}")
print(f"    χ² = {chi2_srm:.4f}, p = {p_srm:.4f}")
print(f"    → {'No SRM detected' if p_srm > 0.001 else 'SRM DETECTED — investigate before analyzing!'}")

# Guardrail metrics (simulated)
print(f"\n  Guardrail Metrics (simulated post-experiment):")
guardrails = [
    {'metric': 'Churn Rate', 'treatment': 0.038, 'control': 0.042, 'threshold': 0.02, 'direction': 'increase'},
    {'metric': 'Complaint Rate', 'treatment': 0.015, 'control': 0.016, 'threshold': 0.01, 'direction': 'increase'},
    {'metric': 'Avg Delivery Time', 'treatment': 34.2, 'control': 34.8, 'threshold': 5.0, 'direction': 'increase'},
]

print(f"  {'Metric':<22} {'Treatment':>12} {'Control':>12} {'Diff':>10} {'Threshold':>12} {'Status':>8}")
print(f"  {'─'*76}")

for g in guardrails:
    diff = g['treatment'] - g['control']
    passed = abs(diff) < g['threshold']
    print(f"  {g['metric']:<22} {g['treatment']:>12.3f} {g['control']:>12.3f} {diff:>+10.3f} {g['threshold']:>12.3f} {'PASS' if passed else 'FAIL':>8}")


# ── 6. DECISION FRAMEWORK ──
print("\n" + "─" * 50)
print("6. DECISION FRAMEWORK")
print("─" * 50)

print("""
  Decision matrix:
    sig + guardrails pass  →  SHIP (roll out)
    sig + guardrail fail   →  INVESTIGATE (fix the regression first)
    not sig + guardrails   →  ITERATE (stronger dosage or longer window)
    not sig + guardrail    →  STOP (rethink the intervention)

  Simulated result:
""")

# Final summary
primary_sig = cuped_p < 0.05
primary_positive = cuped_diff > 0
guardrails_pass = all(abs(g['treatment'] - g['control']) < g['threshold'] for g in guardrails)

if primary_sig and primary_positive and guardrails_pass:
    decision = "SHIP"
    explanation = "Primary metric significant and positive, all guardrails pass."
elif primary_sig and primary_positive:
    decision = "INVESTIGATE"
    explanation = "Primary metric significant but guardrail concerns."
elif not primary_sig and guardrails_pass:
    decision = "ITERATE"
    explanation = "No significant effect; consider stronger treatment or longer duration."
else:
    decision = "STOP"
    explanation = "No significant effect and guardrail concerns."

print(f"    Primary metric (CUPED):  {'Significant' if primary_sig else 'Not significant'} (p={cuped_p:.4f})")
print(f"    Effect direction:        {'Positive' if primary_positive else 'Negative'} (${cuped_diff:,.0f})")
print(f"    Guardrails:              {'All pass' if guardrails_pass else 'FAILURE detected'}")
print(f"    SRM:                     {'Clean' if p_srm > 0.001 else 'MISMATCH'}")
print(f"\n    ══> DECISION: {decision}")
print(f"    ══> {explanation}")


# ── Figure 6: Experiment Results Summary ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('A/B Test Results: Promotion Activation Intervention', fontsize=13, fontweight='bold')

# 6a: Effect size with CI (standard vs CUPED)
methods = ['Standard', 'CUPED']
effects = [raw_diff, cuped_diff]
cis_low = [raw_ci[0], cuped_ci[0]]
cis_high = [raw_ci[1], cuped_ci[1]]
colors_method = ['#CCCCCC', '#2E86AB']

for i, (method, effect, ci_l, ci_h, col) in enumerate(zip(methods, effects, cis_low, cis_high, colors_method)):
    axes[0].barh(i, effect, color=col, height=0.5, edgecolor='white')
    axes[0].plot([ci_l, ci_h], [i, i], color='black', linewidth=2)
    axes[0].plot([ci_l, ci_l], [i-0.1, i+0.1], color='black', linewidth=2)
    axes[0].plot([ci_h, ci_h], [i-0.1, i+0.1], color='black', linewidth=2)
    axes[0].text(ci_h + 20, i, f'${effect:,.0f}\n(p={[raw_p, cuped_p][i]:.4f})', va='center', fontsize=9)

axes[0].axvline(x=0, color='red', linestyle='--', linewidth=0.8)
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(methods)
axes[0].set_xlabel('Treatment Effect ($)')
axes[0].set_title('Effect Size: Standard vs CUPED')

# 6b: Variance reduction visualization
axes[1].bar(['Standard\nSE', 'CUPED\nSE'], [raw_se, cuped_se],
            color=['#CCCCCC', '#2E86AB'], edgecolor='white')
axes[1].set_ylabel('Standard Error ($)')
axes[1].set_title(f'Variance Reduction: {variance_reduction:.0%}')
for i, v in enumerate([raw_se, cuped_se]):
    axes[1].text(i, v + 5, f'${v:,.0f}', ha='center', fontsize=10)

# 6c: Power curve with achieved result
axes[2].plot([m*100 for m in mde_range], powers_by_mde, color='#2E86AB', linewidth=2)
axes[2].axhline(y=0.80, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='80% power')
observed_effect_pct = (cuped_diff / baseline_gmv_mean) * 100
axes[2].axvline(x=observed_effect_pct, color='green', linestyle='-', linewidth=2,
                label=f'Observed: {observed_effect_pct:.1f}%')
axes[2].axvline(x=mde_pct*100, color='orange', linestyle='--', linewidth=1,
                label=f'MDE: {mde_pct*100:.0f}%')
axes[2].set_xlabel('Effect Size (%)')
axes[2].set_ylabel('Statistical Power')
axes[2].set_title('Power Curve with Observed Effect')
axes[2].legend(fontsize=8)
axes[2].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_experiment_results.png'), bbox_inches='tight')
plt.close()
print("\n  → Saved fig6_experiment_results.png")

# Save experiment specification
experiment_spec['results'] = {
    'decision': decision,
    'raw_effect': round(raw_diff, 2),
    'cuped_effect': round(cuped_diff, 2),
    'raw_p': round(raw_p, 6),
    'cuped_p': round(cuped_p, 6),
    'variance_reduction': round(variance_reduction, 4),
    'n_treatment': int(n_t),
    'n_control': int(n_c),
    'srm_p': round(p_srm, 4),
}
with open(os.path.join(OUTPUT_DIR, 'experiment_spec.json'), 'w') as f:
    json.dump(experiment_spec, f, indent=2)

print("\n" + "=" * 70)
print("EXPERIMENT DESIGN COMPLETE")
print("=" * 70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
