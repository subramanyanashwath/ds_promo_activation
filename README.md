# Promotion Activation Is the #1 Growth Lever for New Vertical Merchants

**Merchants who activate promotions are 2.2x more likely to become high-growth** — and 55% of the merchant base hasn't activated a single one. This project identifies the intervention, sizes the opportunity, and designs the experiment to prove it.

Portfolio Project by **Ashwath Subramanyan** | Data Scientist (Analytics)

---

## Why This Matters Now

DoorDash's new verticals (grocery, convenience, retail) are the company's next growth engine — GOV grew 25% YoY in Q3 2025 while Instacart managed only 10%. But growth at the platform level masks a merchant-level problem: **most new-vertical merchants stall in the activation funnel.** They onboard, but never activate promotions, expand their delivery radius, or complete their menu. The gap between top-performing and underperforming merchants is widening, and the funnel stages where merchants drop off aren't obvious from top-line metrics alone.

This project maps the full merchant growth funnel — **Onboard → Activate Promos → Expand Radius → Upload Content → Achieve High Growth** — identifies where merchants stall, sizes the opportunity, and designs the experiment to prove the fix works.

## The Insight

Using a dataset of 5,000 new-vertical merchants across grocery, convenience, and retail, I built a full analytics pipeline — from raw ingestion through causal inference to experiment design — and found a clear answer:

**Promotion activation is the single strongest predictor of merchant growth**, with an odds ratio of 2.21 (95% CI [1.92, 2.54], p<0.001). Merchants who run promotions early (within 30 days of onboarding) reach a 52.3% high-growth rate, compared to 39.6% for late adopters — and this isn't just correlation. Propensity score stratification estimates a causal Average Treatment Effect of +18.5% on high-growth probability.

The other significant drivers, in order: wide delivery radius (OR=1.40), photo uploads (OR=1.39), menu completeness (OR=1.35), and response time (OR=1.18). All five survive significance testing; three survive Bonferroni correction at α=0.01.

## The Opportunity

A 2×2 segmentation on GMV potential × engagement level reveals four merchant segments. The one that matters most:

| Segment | Merchants | Annual GMV | Profile |
|---------|-----------|------------|---------|
| **ACTIVATE** | 1,452 | $108.0M | High GMV potential, low engagement — **intervention targets** |
| PROTECT | 1,103 | — | High GMV, high engagement — retention focus |
| NURTURE | 1,047 | — | Low GMV, high engagement — long-term bets |
| MONITOR | 1,453 | — | Low GMV, low engagement — watch list |

The ACTIVATE segment represents **$108.0M in annual GMV** across merchants who have the order volume and basket size to grow, but haven't activated the engagement levers that drive it. 80% of the total merchant base lacks active promotions — the single highest-impact lever.

## What I'd Recommend

**1. Make promotion activation a Day-1 onboarding milestone.** The data shows early promo adopters (0–30 days) reach 52.3% high-growth rates vs. 39.6% for late adopters. This is the lowest-friction, highest-impact intervention available. Hypothesized lift: 50–70% more promo activation.

**2. Expand the default delivery radius from 3mi to 5mi.** Wide radius merchants show 1.40x growth odds with no degradation in customer ratings or cancellation rates. This is a configuration change, not a behavior change — making it an ideal default nudge. Hypothesized lift: 15–25% more orders.

**3. A/B test before scaling.** I designed the experiment: n=83/arm, 90 days, stratified by vertical and region. CUPED pre-experiment adjustment using historical GMV reduces variance by **79.7%**, making this an efficient test. Guardrail metrics (churn, complaints, delivery time) and SRM detection are built in. The simulated result: **SHIP** (p<0.001, all guardrails pass, +$600/month per merchant in GMV lift).

**4. Scale with AI-powered merchant recommendations.** A RAG-based system (ChromaDB + DSPy) can generate personalized growth memos for every ACTIVATE merchant — "Your delivery radius is below the 5mi threshold that 1.40x growth merchants use. Here's how to expand it." This scales MSM coverage 10x without adding headcount, and every recommendation traces back to the driver analysis.

## How I Validated It

This isn't a single model — it's a layered analytical framework covering **funnel analysis, regression modeling, cohort analysis, time series analysis, hypothesis testing, A/B experiment design, and user segmentation** — designed to withstand scrutiny at each step:

**Data Pipeline (Medallion Architecture):** Bronze → Silver → Gold processing 5,000 merchants, 60K engagement records, and 60K orders. Five data quality gates. Designed for Databricks Delta Lake with production patterns (Auto Loader, DLT expectations, Z-ORDER optimization).

**Driver Analysis:** Logistic regression with vertical and region controls. I chose logistic regression over gradient boosting deliberately — the goal isn't to predict which merchants will grow (a black-box prediction), it's to tell Merchant Strategy Managers *which levers to pull* in their next merchant conversation. Interpretable odds ratios serve that stakeholder need directly. AUC of 0.633 is appropriate for a driver-identification model; higher AUC via ensemble methods would sacrifice the interpretability that makes findings actionable. All results include 95% CIs and explicit causal vs. correlational labeling.

**Causal Inference:** Propensity score stratification to estimate ATE while controlling for observed confounders (vertical, region, tenure, menu size). The limitation: unobserved confounders could still bias estimates. That's why I designed the A/B test — to get clean causal evidence before scaling.

**Cohort Analysis:** Five promotion-timing cohorts (early/mid/late/very late/never) with chi-squared omnibus test (p<0.001) and pairwise t-tests confirming early adopters significantly outperform late adopters. **Time series analysis** of GMV trajectories shows the growth gap between early and late adopters widens monotonically over the first 12 months, confirming that early promo activation has compounding — not just one-time — effects on merchant growth.

**Experiment Design:** Full A/B test framework with power analysis (n=83/arm, MDE=8%, α=0.05, β=0.80), stratified randomization by vertical and region, CUPED variance reduction using pre-experiment GMV as a covariate, guardrail metrics (merchant churn, customer complaints, delivery time), SRM detection (p=0.75, confirming balanced assignment), and a Ship/Investigate/Iterate/Stop decision matrix tied to significance thresholds and guardrail pass rates.

## What I'd Do With Real Data

This project uses synthetic data calibrated to DoorDash's public benchmarks. With access to real merchant data, I'd extend the analysis in three directions:

**Survival analysis for churn prediction.** The current framework identifies growth drivers but doesn't model *when* merchants churn. A Cox proportional hazards model on real tenure data would let us identify at-risk merchants before they leave and quantify the retention value of each engagement lever.

**Difference-in-differences on policy changes.** DoorDash has rolled out onboarding changes, radius defaults, and promo programs at different times in different markets. A DiD design on real policy rollouts would give us causal estimates without needing a new experiment — and would validate or challenge the propensity score estimates.

**Heterogeneous treatment effects by vertical.** The current model controls for vertical but doesn't estimate *how differently* promotions work across grocery vs. convenience vs. retail. A causal forest or stratified analysis on real data could reveal that promo activation is a 3x lever in convenience but only 1.5x in grocery — changing how we prioritize interventions by vertical.

**LTV integration.** The current GMV growth metric captures 12-month behavior but doesn't weight by margin or account for customer acquisition costs. Integrating DoorDash's LTV model would let us re-rank the ACTIVATE segment by profit potential rather than revenue potential, likely shifting priority toward higher-margin verticals.

## Project Structure

```
├── notebooks/
│   ├── 01_medallion_data_pipeline.py    # Bronze → Silver → Gold ETL
│   ├── 02_statistical_analysis.py       # Driver analysis, causal inference, cohorts
│   └── 03_experiment_design.py          # A/B test design with CUPED
├── data/                                # Parquet files (Bronze/Silver/Gold layers)
├── outputs/                             # Visualizations and analysis results
└── docs/                                # Documentation and reference materials
```

## Tech Stack

Python, Databricks Medallion Architecture (Bronze/Silver/Gold), statsmodels, scipy, scikit-learn, matplotlib. Designed for Databricks Workflows with Delta Lake, Auto Loader, and DLT patterns.

## Data Transparency

All data is synthetic, with merchant behaviors calibrated to DoorDash's public benchmarks. Market context uses real Q3 2025 earnings data (DoorDash +25% YoY GOV growth, Instacart +10% YoY). Effect sizes are generated via a calibrated log-odds formulation with industry-benchmarked odds ratios. Every statistical claim includes confidence intervals, p-values, and explicit causal vs. correlational labeling.

