# Doordash Merchant Growth Analytics - Exact Numbers & Metrics

This directory contains comprehensive metrics and analysis results from the Doordash Merchant Growth Analytics project.

## Files in This Directory

### Summary Documents

1. **NUMBERS_AT_A_GLANCE.txt** (BEST START HERE)
   - Visual summary with box diagrams
   - All key metrics at a glance
   - Perfect for quick reference and presentations
   - Format: Clean ASCII tables

2. **EXACT_NUMBERS.txt**
   - Complete listing of all metrics
   - Organized by category
   - Includes key statistics for presentations
   - Best for finding specific numbers

3. **metrics_summary.txt**
   - Detailed 7-section breakdown
   - Includes interpretation and insights
   - Comprehensive driver analysis
   - Contains experiment methodology details

### Data Files

4. **quick_reference.csv**
   - Structured CSV format
   - All metrics with categories and units
   - Easily importable to spreadsheets
   - 40+ metrics covered

5. **driver_analysis_results.csv**
   - Logistic regression results
   - Odds ratios with 95% confidence intervals
   - P-values and significance indicators
   - 5 key growth drivers

6. **experiment_spec.json**
   - Full experiment specification
   - Hypothesis and design details
   - Complete results with p-values
   - Treatment and control definitions

## Key Metrics Summary

### Population
- Total Merchants: 5,000
- High Growth: 2,113 (42.3%)
- Active: 4,561 (91.2%)

### Financial (12-month)
- Total GMV: $287.6 million
- Total Orders: 7.55 million
- Avg GMV/Merchant: $57,529

### Segmentation
- ACTIVATE Segment: 1,397 merchants (27.9%)
- ACTIVATE GMV: $103.3 million
- ACTIVATE High Growth Rate: 34.6%

### Promotion Status
- Active: 2,235 (44.7%)
- Inactive: 2,765 (55.3%)

### Driver Analysis (Top 3)
1. Promotion Active: OR=2.233 (p<0.000001) ✓ STRONGEST DRIVER
2. Wide Delivery Radius: OR=1.404 (p<0.000001) ✓
3. Photos Uploaded: OR=1.388 (p<0.000001) ✓

### Experiment Results
- Sample Size: 913 merchants (451 treatment, 462 control)
- Duration: 90 days (14-day burn-in excluded)
- Raw Effect: $644.47/month
- CUPED-Adjusted Effect: $593.54/month ✓
- Variance Reduction: 82.4%
- p-value: <0.0001 ✓ HIGHLY SIGNIFICANT
- **DECISION: SHIP ✓**

## How to Use These Files

### For Presentations
Start with **NUMBERS_AT_A_GLANCE.txt** - it has all the metrics in a clean, visual format ready for slides.

### For Data Analysis
Use **quick_reference.csv** to import metrics into Excel/Python for further analysis.

### For Documentation
Reference **EXACT_NUMBERS.txt** when you need to cite specific numbers in reports.

### For Details
See **metrics_summary.txt** for full context, methodology, and insights.

### For Reproducibility
Check **experiment_spec.json** for complete experimental design details.

## Important Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| High Growth Merchants | 2,113 (42.3%) | Overall rate |
| ACTIVATE Segment | 1,397 (27.9%) | Main intervention target |
| Promotion Effect (OR) | 2.233 | 2.2x improvement in growth probability |
| Monthly GMV Lift | $593.54 | CUPED-adjusted result |
| Statistical Significance | p<0.0001 | Highly significant |
| Variance Reduction | 82.4% | CUPED efficiency |
| Sample Size | 913 | Eligible for testing |
| Active Promotions | 2,235 (44.7%) | Market penetration |
| Menu Complete | 1,094 (21.9%) | Content gap opportunity |

## Analysis Highlights

1. **Promotion is the Strongest Driver**: 2.2x odds ratio - promotion activity has the largest impact on high growth probability

2. **Large ACTIVATE Segment**: 1,397 merchants representing $103.3M annual GMV waiting for intervention

3. **Highly Efficient Experiment**: CUPED methodology achieved 82.4% variance reduction, allowing detection of meaningful effects

4. **Significant Engagement Gaps**: 
   - 55.3% have no active promotions
   - 78.1% have incomplete menus (< 80%)
   - 2.31/5.0 average engagement score

5. **Validated for Scale**: Effect size and statistical significance support rolling out to broader population

## Data Source

All metrics derived from:
- `/data/gold_merchants.parquet` - 5,000 merchants with features
- `/data/gold_orders_monthly.parquet` - 60,000 monthly order records
- `/data/gold_engagement_history.parquet` - 60,000 engagement records

## Methodology

- **Segmentation**: 2x2 grid (GMV tier × Engagement tier)
- **Driver Analysis**: Logistic regression with high-growth binary outcome
- **Experiment Design**: Randomized controlled trial with city-level clustering
- **Variance Reduction**: CUPED (Controlled-experiment Using Pre-Experiment Data)
- **Statistical Tests**: Frequentist hypothesis testing with alpha=0.05

## Questions?

Refer to the specific document based on your need:
- Quick facts? → NUMBERS_AT_A_GLANCE.txt
- Specific number? → EXACT_NUMBERS.txt
- Full analysis? → metrics_summary.txt
- Structured data? → quick_reference.csv
- Experiment details? → experiment_spec.json
