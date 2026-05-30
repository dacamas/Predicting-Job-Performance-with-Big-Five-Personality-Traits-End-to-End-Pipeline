# Predicting-Job-Performance-with-Big-Five-Personality-Traits-End-to-End-Pipeline
https://colab.research.google.com/github/dacamas/Predicting-Job-Performance-with-Big-Five-Personality-Traits-End-to-End-Pipeline/blob/main/Predicting_Job_Performance_with_Big_Five_Personality_Traits.ipynb

An end-to-end PyTorch machine learning pipeline that predicts job performance outcomes from Big Five personality trait scores. Built as a production-ready system designed for real organisational data.

---

## What This Project Demonstrates

### Machine Learning Engineering
- **Multi-task learning** — a single model trained simultaneously on two objectives: regression (continuous performance score) and classification (Low / Mid / High tier), using a normalized combined loss function to balance the two tasks
- **Residual MLP architecture** — custom `ResBlock` layers with BatchNorm and GELU activations, avoiding gradient vanishing at depth without the overhead of a transformer on a small feature set
- **Training hygiene** — OneCycleLR scheduling with warmup, gradient clipping, early stopping with patience, and best-checkpoint saving
- **Proper data splitting** — 70/15/15 train/val/test split with `StandardScaler` fit exclusively on training data to prevent leakage into evaluation

### Feature Engineering
- Reverse-scoring of IPIP personality items per standard psychometric protocol
- Trait-level aggregation from 50 raw item responses to five dimension scores
- Interaction terms grounded in occupational psychology literature (e.g. Conscientiousness × Extraversion for leadership roles)
- Occupational category encoding with job-fit composites derived from meta-analytic weights (Barrick & Mount, 1991)

### Interpretability
- SHAP `DeepExplainer` applied to the regression head, producing feature-level attribution scores for each prediction
- Beeswarm plots showing direction of trait effects on performance score
- XGBoost feature importance comparison as a cross-validation of SHAP findings

### Benchmarking
- Systematic comparison against Ridge regression and XGBoost baselines
- Discussion of why gradient boosted trees tend to outperform MLPs on small tabular feature sets, and where the neural net adds value (calibrated dual output, SHAP compatibility, extensibility)

---

## Important Note on Evaluation Metrics

This pipeline uses semi-synthetic labels generated from the same trait scores used as model inputs, following published meta-analytic weights with added Gaussian noise. As a result, reported R² and F1 scores are **not meaningful as real-world performance estimates** — any model with access to the input traits can partially recover the label generation function by construction.

This is a known and documented limitation of working without real organisational data, not a flaw in the pipeline architecture. The synthetic setup serves to verify that the full pipeline runs correctly end to end. Evaluation metrics become meaningful the moment real labelled data is substituted in Section 5 onward.

---

## Applying This Pipeline to Real Data

This is where the project becomes a genuine research contribution. The pipeline is designed to accept real data with minimal changes.

### What Data You Would Need

**Personality scores** — administer the [IPIP-50 questionnaire](https://ipip.ori.org) to employees. It is free, takes approximately 10 minutes per person, and produces raw item responses in the exact format this pipeline expects. No licensing required.

**Job performance ratings** — one of the following:

| Type | Examples | Notes |
|---|---|---|
| Supervisor ratings | Annual review scores, performance appraisal ratings | Most common in research; subjective but validated |
| Objective KPIs | Sales revenue, tickets resolved, units produced | Stronger validity; not available for all roles |
| 360-degree feedback | Peer + manager composite ratings | Most robust; requires more organisational buy-in |

**Minimum viable dataset** — approximately 200–300 employees with both personality scores and at least one performance measure. This matches the sample size of most published studies in this area.

### How to Plug In Real Data

Replace Sections 3 and 4 of the notebook with:

```python
# Load your real data
df_personality  = pd.read_csv('ipip_responses.csv')   # 50 IPIP item columns + employee_id
df_performance  = pd.read_csv('performance_ratings.csv')  # employee_id + performance_score

# Merge on employee ID
df_merged = df_personality.merge(df_performance, on='employee_id')

# Drop the synthetic label generation entirely — use real ratings
df_feat['perf_score'] = df_merged['performance_score'].values.astype(np.float32)

# Derive tiers from real score distribution
q33, q66 = np.percentile(df_feat['perf_score'], [33, 66])
df_feat['perf_tier'] = np.where(
    df_feat['perf_score'] < q33, 0,
    np.where(df_feat['perf_score'] < q66, 1, 2)
).astype(np.int64)
```

Everything from Section 5 onward — feature scaling, model training, evaluation, SHAP — runs unchanged.

### What Real Results Would Look Like

Based on published meta-analyses, realistic expectations for personality-only prediction:

- **R² 0.15–0.30** — consistent with the literature; Conscientiousness and Emotional Stability are the strongest predictors across occupations
- **Higher R² for specific occupational groups** — sales and managerial roles show stronger personality-performance relationships than skilled/technical roles
- **SHAP plots** will reflect known findings: Conscientiousness should dominate feature importance, with Extraversion elevated for customer-facing roles

Outperforming these benchmarks would likely require adding cognitive ability scores or structured interview ratings alongside Big Five scores.

---

## Project Structure

```
├── big5_job_performance_pipeline.ipynb   # main notebook
├── README.md
└── data/
    └── place_real_data_here.md           # instructions for data placement
```

## Dependencies

```
torch
scikit-learn
xgboost
shap
pandas
numpy
matplotlib
seaborn
```

## References

- Barrick, M. R., & Mount, M. K. (1991). The Big Five personality dimensions and job performance. *Personnel Psychology*, 44(1), 1–26.
- Judge, T. A., et al. (2002). Personality and leadership: A qualitative and quantitative review. *Journal of Applied Psychology*, 87(4), 765–780.
- Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure. *Psychological Assessment*, 4(1), 26–42.
