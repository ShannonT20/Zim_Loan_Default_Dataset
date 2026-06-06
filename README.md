# Zimbabwe Loan Default Prediction Dataset

**A calibrated synthetic microfinance and SME loan dataset modeling Zimbabwean retail lending dynamics under privacy constraints.**

## Overview

This dataset contains **38,932 synthetic loan records** with a calibrated logistic risk function that mechanistically links default outcomes to 11 borrower, loan, and contextual risk factors — modeling realistic retail lending dynamics in Zimbabwe's banking and microfinance sector.

**This dataset is entirely synthetic — no records from any external dataset were copied or sampled.** All features are independently generated using distributions calibrated to official Zimbabwean statistical sources. Research was conducted to find publicly available loan default data from Zimbabwe or the broader African continent, but no suitable datasets could be found. Banking data across Africa is extremely hard to access due to regulation and confidentiality. The [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data) on Kaggle (Nikhil1e9, 2024) was studied to understand the general statistical relationships between borrower characteristics and default outcomes, but **no records were read from, sampled from, or copied from that dataset** during generation. All values are independently generated from Zimbabwe-calibrated distributions.

The dataset includes province-level geographic coverage (all 10 provinces, weighted from the ZIMSTAT 2022 Census), 13 employment sectors (with Informal Sector at 17%, matching ZIMSTAT Q1 2024 QLFS data showing 41.3% informal employment nationally), province-aware mobile money disbursement channels (EcoCash, InnBucks, OneMoney), and realistic USD-denominated amounts calibrated to Zimbabwe's dollarized economy.

## Motivation

### The Problem

Zimbabwe's banking sector faces persistent challenges with loan defaults, which constrain credit availability — particularly for small and medium enterprises (SMEs), agricultural producers, and individuals in underserved communities. The Reserve Bank of Zimbabwe's Quarterly Banking Sector Report (December 2024) shows the formal banking sector maintained an NPL ratio of ~2.09%, but the microfinance sector — which serves the majority of retail and SME borrowers — reported a Portfolio at Risk > 30 days (PAR30) of **8.3% as of September 2024** (ZAMFI), far exceeding the international benchmark of 5%. In 2019, microfinance PAR30 reached 14.58% (RBZ Quarterly Microfinance Report, June 2019), driven by macroeconomic volatility, currency instability, and income deterioration.

### The Data Gap

During the development of this dataset, extensive research was conducted to identify publicly available loan default datasets from Zimbabwe or the broader African continent. The search did not identify a suitable openly accessible dataset for this use case. Banking data across Africa — and particularly in Zimbabwe — is often difficult to access due to:

- **Regulatory restrictions** — financial data is heavily regulated under banking secrecy and data protection laws
- **Confidentiality requirements** — loan-level records contain sensitive personal and financial information that institutions cannot legally share
- **Institutional reluctance** — banks view their data as proprietary competitive assets
- **Lack of data infrastructure** — many African financial institutions lack the data governance frameworks needed to safely anonymize and release data for research

This data scarcity creates a significant barrier for African researchers, students, and startups who want to develop AI solutions for the financial sector but have no representative data to work with. Most publicly available loan datasets (e.g., from Kaggle or UCI) reflect Western financial contexts — different income levels, different regulatory environments, different product structures — and are not transferable to African contexts.

### Practical Use

This dataset is intended for research, education, and prototyping. It was also used in the **Deep Learning IndabaX Zimbabwe 2026 Hackathon: Loan Default Prediction in Zimbabwe's Banking Sector**, where participants built end-to-end machine learning solutions to predict the `defaulted` target.

The public competition page described:

- A Zimbabwe-style tabular loan default prediction challenge hosted under Deep Learning IndabaX Zimbabwe 2026
- ROC-AUC as the primary evaluation metric, with accuracy, precision, recall, and F1 as supporting diagnostics
- Public participation figures showing 238 joined and 147 active at the time captured

This demonstrates that the dataset was usable in a real hackathon setting for model development and leaderboard evaluation. It does not, by itself, establish production validity or real-world deployment readiness.

## Dataset Description

| Property | Value |
|----------|-------|
| **Records** | 38,932 |
| **Features** | 22 (including target) |
| **Target Variable** | `defaulted` (binary: 1 = default, 0 = repaid) |
| **Default Rate** | ~24.1% |
| **Format** | CSV (UTF-8 encoded) |
| **Size** | ~6.7 MB |
| **License** | CC-BY-4.0 |
| **Language** | English |
| **Geographic Coverage** | All 10 provinces of Zimbabwe |
| **Temporal Coverage** | Loans disbursed between October 2024 and February 2026 |

## Features

| Column | Type | Description | Missing % |
|--------|------|-------------|-----------|
| `loan_ref` | ID | Unique loan identifier (format: 2 letters + 5 digits) | 0% |
| `product_code` | Categorical | Loan product type: 0=Personal, 1=SME, 2=Agriculture, 3=Salary_Based, 4=Asset_Finance, 5=Emergency | 0% |
| `date_approved` | Date | Loan approval date (D/M/YYYY) | 0% |
| `date_disbursed` | Date | Loan disbursement date (D/M/YYYY) | 0% |
| `first_payment_due` | Date | First payment due date (D/M/YYYY) | 0% |
| `maturity_date` | Date | Loan maturity date (D/M/YYYY) | 0% |
| `amount_usd` | Numerical | Loan amount in USD | 0% |
| `annual_rate_pct` | Numerical | Annual interest rate (%) | ~1% |
| `term_months` | Numerical | Loan term in months | 0% |
| `payment_frequency` | Categorical | Payment schedule: Monthly, Bi-Weekly, Weekly | 0% |
| `loan_purpose` | Categorical | Purpose of the loan (15 categories) | ~2% |
| `client_gender` | Categorical | Borrower gender: Male, Female | 0% |
| `client_dob` | Date | Borrower date of birth (D-Mon-YY) | 0% |
| `marital_status` | Categorical | Married, Single, Divorced, Widowed | ~1.5% |
| `num_dependents` | Numerical | Number of dependents | ~6% |
| `employment_sector` | Categorical | Borrower's employment sector (13 categories) | ~3% |
| `months_at_employer` | Numerical | Months at current employer | ~5% |
| `monthly_income_usd` | Numerical | Monthly income in USD | ~8% |
| `existing_obligations` | Numerical | Number of existing active loans | 0% |
| `collateral_type` | Categorical | Collateral: None, Vehicle, Property, Livestock, Savings, Guarantor | ~4% |
| `disbursement_channel` | Categorical | EcoCash, Bank_Transfer, Cash, InnBucks, OneMoney | 0% |
| `province` | Categorical | Borrower's province in Zimbabwe | 0% |
| `defaulted` | Binary | Target: 1 = loan defaulted, 0 = loan repaid | 0% |

## Zimbabwe-Specific Contextual Features

### Provinces (all 10 — weighted from ZIMSTAT 2022 Census)

Provincial lending weights are adjusted from raw population shares to reflect that urban centres (Harare, Bulawayo) have disproportionately higher lending activity:

| Province | 2022 Census Pop. Share | Lending Weight |
|---|---|---|
| Harare | 16.0% | 26% |
| Bulawayo | 4.4% | 10% |
| Manicaland | 13.4% | 11% |
| Mashonaland West | 12.5% | 10% |
| Mashonaland East | 9.7% | 8% |
| Midlands | 10.3% | 9% |
| Masvingo | 9.8% | 7% |
| Mashonaland Central | 9.4% | 7% |
| Matabeleland North | 5.5% | 4% |
| Matabeleland South | 5.0% | 4% |

### Employment Sectors (13 — calibrated to ZIMSTAT Q1 2024 QLFS)

Agriculture, Mining, Manufacturing, Retail_Trade, Government, Education, Healthcare, Construction, Transport, Informal_Sector, NGO, Finance, Telecom

ZIMSTAT Q1 2024 reports 41.3% informal employment nationally; the dataset's informal sector weight (17%) is lower because informal workers are underrepresented among formal loan borrowers.

### Disbursement Channels (Province-Aware)

Disbursement channels vary by provincial urbanity, reflecting real mobile money adoption patterns:

| Channel | Urban (Harare, Bulawayo) | Semi-Urban | Rural |
|---|---|---|---|
| EcoCash | 37% | 42% | 36% |
| Bank Transfer | 32% | 25% | 18% |
| Cash | 11% | 16% | 28% |
| InnBucks | 12% | 10% | 10% |
| OneMoney | 8% | 7% | 9% |

EcoCash (~8.35 million subscribers) dominates nationally; InnBucks (~3 million subscribers) is the fastest-growing competitor (TechZim, October 2024). Rural areas have significantly higher cash disbursement.

### Income Distribution (Calibrated to ZIMSTAT Q1 2024 QLFS)

ZIMSTAT reports 83.4% of all employed persons earn < $362/month and 34.4% earn < $90/month. In the dataset, 38.3% of **borrowers** earn < $362/month — lower than the general population because loan borrowers are a wealthier, bankable subset (the poorest do not qualify for formal credit).

| Statistic | Value |
|---|---|
| Min | $31 |
| P25 | $280 |
| Median | $443 |
| Mean | $543 |
| P75 | $696 |
| Max | $2,792 |

Mean ($543) > Median ($443) confirms right-skewed income distribution, consistent with developing economies.

### Loan Products (6)
Personal, SME, Agriculture, Salary-Based, Asset Finance, Emergency — reflecting the diverse product offerings of Zimbabwe's microfinance and banking institutions

## Data Quality and Missing Values

The dataset intentionally contains missing values across several columns, reflecting real-world data quality challenges in financial services. This makes the dataset suitable for research on:
- Missing data imputation techniques
- Robust model training with incomplete features
- Feature engineering under data quality constraints

## Potential Applications

- **Credit scoring and risk assessment** — building predictive models for loan default
- **Financial inclusion research** — understanding lending patterns across provinces and demographics
- **Feature engineering benchmarks** — date features, categorical encoding, imputation strategies
- **Fairness and bias analysis** — examining default prediction across gender, province, and employment sector
- **Explainable AI (XAI)** — interpreting which factors drive default predictions in an African banking context

## Generation Methodology

### Source Data and Research

This dataset is **entirely synthetic** — no real borrower information is included and **no records from any external dataset were copied, sampled, or read during generation**. All 38,932 records are independently generated from scratch using distributions calibrated to official Zimbabwean sources:

1. **ZIMSTAT 2022 Population and Housing Census**: Provincial population shares used as the basis for geographic lending weights. Zimbabwe's total population of 15,178,979 across 10 provinces, with 39% urban population (up from 33% in 2012).

2. **ZIMSTAT Q1 2024 Quarterly Labour Force Survey**: Employment sector shares (41.3% informal, 22.9% agriculture, 23.9% retail), income benchmarks (83.4% earn < $362/mo, 34.4% earn < $90/mo), unemployment rate (20.5%), and labour force participation rate (48%).

3. **Reserve Bank of Zimbabwe (RBZ)**: Monetary policy statements and quarterly banking/microfinance reports. Bank lending rates (USD): personal 13.49%–17.59%, business 10.27%–15.80% per annum (RBZ Aug 2025). RBZ policy rate: 35%. Microfinance rates: 7%–15% per month (84%–180% per annum), confirmed by MicroLoan Foundation (7%/month), ZimLoan (20-30% per term), and ZAMFI sector reports.

4. **RBZ Quarterly Microfinance Reports and ZAMFI**: Microfinance sector Portfolio at Risk > 30 days (PAR30) of 8.3% (September 2024) and 14.58% (June 2019); total microfinance loans of ZiG 1.8 billion (US$72.33 million) as of September 2024.

5. **Mobile money market data**: EcoCash's ~8.35 million subscribers and market leadership (Econet, 2024); InnBucks' ~3 million subscribers and rapid growth through TM Pick n Pay partnerships (TechZim, October 2024); OneMoney's marginal market share.

6. **Kaggle structural reference**: The [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data) (Nikhil1e9, 2024) was studied to understand general statistical relationships between borrower characteristics and default outcomes in consumer lending. **No records were read from, copied from, or sampled from this dataset.** It served only as a conceptual reference for understanding what features matter in credit risk (e.g., that DTI, employment tenure, and interest rate are predictive of default). All generation logic, distributions, and values are independently designed for the Zimbabwe context.

### Default Probability: Calibrated Logistic Risk Function

Default is **not randomly assigned**. Each loan's default probability is computed via a calibrated logistic risk function incorporating **11 risk factors**:

```
P(default) = σ(β₀ + β_income + β_DTI + β_age + β_sector + β_collateral
               + β_obligations + β_rate + β_product + β_province
               + β_dependents + β_tenure + ε)
```

where σ is the sigmoid function and ε ~ N(0, 0.3) provides stochastic noise around the calibrated risk frontier.

The 11 risk factors:
1. **Monthly income** (higher income → lower risk)
2. **Debt-to-income ratio** (higher DTI → higher risk)
3. **Borrower age** (young and pre-retirement → higher risk; prime-age → lower risk)
4. **Employment sector** (Informal → highest risk; Government/Finance/NGO → lowest)
5. **Collateral type** (None → highest risk; Property → lowest)
6. **Existing obligations** (more active loans → higher risk)
7. **Annual interest rate** (higher rate → higher risk, reflecting adverse selection)
8. **Loan product type** (Emergency → highest risk; Salary-Based → lowest)
9. **Province** (urban economies → slightly lower risk)
10. **Number of dependents** (5+ → higher risk)
11. **Employment tenure** (< 6 months → higher risk; 60+ months → lower risk)

### Why 24.1% Default Rate?

The overall default rate of ~24.1% is calibrated to reflect a **mixed retail/microfinance portfolio** that includes personal, SME, emergency, and agricultural loans — many of which are unsecured. This rate is contextually realistic:

- Zimbabwe's formal banking NPL ratio is low (~2.09%) because banks have strict lending criteria
- The microfinance PAR30 ranged from 8.3% (2024) to 14.58% (2019) — but PAR30 measures loans 30+ days overdue at a point in time, not lifetime default
- **Lifetime cumulative default rates** over the full loan term are substantially higher than point-in-time PAR30 snapshots
- IFC data shows Africa-wide corporate default rates of 6.7% (1986–2023), with low-income economies at 8.6%
- A mixed portfolio that includes emergency loans (39.9% default), informal sector borrowers (36.7% default), and unsecured lending (29.3% default) will have an aggregate rate above pure formal banking metrics

### Generation Pipeline

1. **Borrower age** — independently generated via mixture of triangular distributions (peak 28–42), not sourced from any external file
2. **Demographic modeling** — gender, marital status distributions with age-dependent probabilities
3. **Income modeling** — sector-specific triangular distributions calibrated to ZIMSTAT Q1 2024 benchmarks
4. **Loan product logic** — 6 product types with product-specific amount multipliers, interest rate ranges, term structures, and correlated collateral/purpose selections
5. **Geographic weighting** — province lending weights derived from ZIMSTAT 2022 Census population data, adjusted for urban lending concentration
6. **Province-aware disbursement channels** — urban/semi-urban/rural channel distributions reflecting real mobile money adoption patterns
7. **Default probability modeling** — calibrated logistic risk function computing default probability as a function of 11 interdependent risk factors
8. **Date consistency** — logical ordering enforced: approval < disbursement < first payment < maturity
9. **Strategic missing value injection** — varying rates (1%–8%) across 8 different columns to create realistic data quality challenges
10. **Reproducibility** — generation uses a fixed random seed (42) and a retained Python script; no external data files are read

## Validation: Evidence of Mechanistic Realism

The following validation evidence demonstrates that the dataset exhibits the monotonic, interpretable relationships expected in real credit risk data.

### Default Rate by Income Quartile

| Income Quartile | Range (USD/month) | Default Rate | Loans |
|---|---|---|---|
| Q1 (lowest) | $31 – $280 | **34.0%** | 8,954 |
| Q2 | $280 – $442 | 24.6% | 8,954 |
| Q3 | $443 – $696 | 19.8% | 8,954 |
| Q4 (highest) | $696 – $2,792 | **17.9%** | 8,956 |

Clear monotonic decrease: lower income = higher default risk. 1.9x spread.

### Default Rate by Debt-to-Income (DTI) Ratio

| DTI Bucket | Default Rate | Loans |
|---|---|---|
| Low (< 0.1) | **12.3%** | 536 |
| Medium (0.1 – 0.3) | 15.9% | 8,387 |
| High (0.3 – 0.5) | 21.4% | 7,627 |
| Very High (> 0.5) | **29.0%** | 19,268 |

Clear monotonic increase: higher debt burden = higher default risk. 2.4x spread.

### Default Rate by Age Group

| Age Group | Default Rate | Loans |
|---|---|---|
| 18–25 | **28.9%** | 4,408 |
| 26–35 | 23.8% | 10,459 |
| 36–50 | **20.2%** | 13,428 |
| 51–70 | 26.0% | 8,409 |

Young borrowers riskiest; prime-age (36–50) most stable; older borrowers show slight increase (retirement/income risk).

### Default Rate by Employment Sector

| Sector | Default Rate | Loans |
|---|---|---|
| Informal Sector | **36.7%** | 6,423 |
| Agriculture | 29.8% | 6,459 |
| Retail Trade | 21.8% | 4,875 |
| Government | 15.3% | 3,351 |
| Finance | **13.8%** | 1,520 |
| NGO | 13.3% | 1,186 |

2.8x spread between highest-risk (Informal, 36.7%) and lowest-risk (NGO, 13.3%) sectors.

### Default Rate by Collateral Type

| Collateral | Default Rate | Loans |
|---|---|---|
| None | **29.3%** | 10,828 |
| Livestock | 24.2% | 4,026 |
| Savings | 19.9% | 4,987 |
| Guarantor | 21.9% | 8,021 |
| Vehicle | 22.0% | 4,214 |
| Property | **17.7%** | 5,299 |

Unsecured loans default at 1.7x the rate of property-backed loans.

### Default Rate by Loan Product

| Product | Default Rate | Loans |
|---|---|---|
| Emergency | **39.9%** | 2,404 |
| Agriculture | 26.4% | 6,908 |
| SME | 25.9% | 8,565 |
| Personal | 24.9% | 10,940 |
| Asset Finance | 22.3% | 3,871 |
| Salary-Based | **13.0%** | 6,244 |

Emergency loans default at 3.1x the rate of salary-based loans.

### Default Rate by Existing Obligations

| Obligations | Default Rate | Loans |
|---|---|---|
| 0 | **20.2%** | 9,681 |
| 1–2 | 22.7% | 17,432 |
| 3–4 | 25.8% | 7,907 |
| 5+ | **30.5%** | 3,912 |

Monotonic increase with debt accumulation. 1.5x spread.

### Default Rate by Employment Tenure

| Tenure | Default Rate | Loans |
|---|---|---|
| < 6 months | **32.6%** | 863 |
| 6–24 months | 29.8% | 8,151 |
| 24–60 months | 23.8% | 13,135 |
| 60+ months | **19.0%** | 14,837 |

Clear monotonic decrease: employment stability = lower default risk. 1.7x spread.

### Loan Amount–Income Correlation

Pearson correlation coefficient: **r = 0.587** — moderate positive correlation confirming that loan amounts are income-dependent, not randomly assigned.

### Default Rate by Interest Rate Band

| Rate Band | Default Rate | Loans |
|---|---|---|
| Bank rate (< 20%) | **18.2%** | 17,714 |
| Low MFI (20% – 80%) | 27.8% | 1,412 |
| Mid MFI (80% – 130%) | 28.1% | 13,750 |
| High MFI (130%+) | **32.1%** | 5,667 |

Clear monotonic increase: higher interest rates (indicating microfinance sourcing and higher-risk borrower profiles) correlate with higher default rates. 1.8x spread.

### Interest Rate by Product

| Product | Mean Rate | Range |
|---|---|---|
| Salary-Based | 21.8% | 10.1% – 119.3% |
| Asset Finance | 31.2% | 12.1% – 127.8% |
| SME | 72.5% | 10.0% – 155.9% |
| Agriculture | 74.0% | 8.2% – 142.8% |
| Personal | 87.0% | 13.0% – 179.7% |
| Emergency | 145.5% | 15.8% – 202.2% |

Interest rates are product-differentiated and reflect the dual bank/microfinance lending structure in Zimbabwe. Salary-Based and Asset Finance products are predominantly bank-sourced (10%–20% annual, matching RBZ Aug 2025 data: personal 13.49%–17.59%, business 10.27%–15.80%). Emergency loans are almost entirely microfinance-sourced (96–204% annual, reflecting the 8%–17% monthly rates charged by Zimbabwean MFIs). Personal, SME, and Agriculture products show bimodal rate distributions — a cluster at bank rates and a cluster at MFI rates — reflecting the reality that both banks and microfinance institutions serve these segments.

### Disbursement Channel by Province Type

| Province Type | EcoCash | Bank Transfer | Cash | InnBucks | OneMoney |
|---|---|---|---|---|---|
| Urban | 37% | 32% | 11% | 12% | 8% |
| Semi-Urban | 42% | 25% | 17% | 10% | 7% |
| Rural | 36% | 18% | **28%** | 10% | 9% |

Rural areas show 2.5x higher cash usage and lower bank transfer rates, reflecting financial infrastructure differences.

## Limitations

1. **Synthetic, not observed**: While default probabilities are mechanistically linked to 11 risk factors via a calibrated logistic function, the underlying relationships are modeled, not learned from real Zimbabwean loan performance data. Real-world default dynamics may involve non-linearities, interaction effects, and temporal patterns not captured here.

2. **No temporal dynamics**: The dataset covers the ZiG-era period (October 2024 – February 2026) but does not model macroeconomic dynamics, currency instability, policy changes, or seasonal effects (e.g., agricultural cycles) that significantly influence default patterns in Zimbabwe.

3. **No credit history trajectory**: Each record is a single point-in-time snapshot. Real credit risk models benefit from longitudinal repayment behavior, prior default history, and credit bureau scores — none of which are modeled here.

4. **Simplified income model**: Income is modeled as sector-specific triangular distributions. Real income distributions have heavier tails, more complex sector × province interactions, and significant within-sector variance due to firm size, seniority, and education.

5. **No behavioral features**: Real lending data includes behavioral signals — mobile money transaction patterns, savings behavior, repayment regularity — that are strong default predictors. These are absent from this dataset.

6. **Default rate contextualization**: The 24.1% overall rate reflects a mixed portfolio including high-risk segments (emergency, informal). Formal banking NPLs in Zimbabwe are much lower (~2%). Users should understand that this rate represents a specific portfolio mix, not Zimbabwe's banking sector as a whole.

7. **Single-country scope**: The dataset is calibrated to Zimbabwe's specific economic context and should not be generalized to other African markets without recalibration.

## Ethical Considerations

- **Fully synthetic**: All 38,932 records are synthetically generated. **No real individuals' data is included** and no real banking institution's proprietary data was accessed or used. No records from any external dataset were copied.
- **No re-identification risk (from real borrowers)**: Since the data is entirely synthetic (not derived from real borrower records), there is no plausible path to re-identify any real borrower from this dataset.
- **No regulatory conflict**: Because no real banking data was accessed, this dataset does not conflict with banking secrecy laws, data protection regulations, or institutional confidentiality agreements.
- **Bias awareness**: The default probability model incorporates sector and geographic factors that reflect real-world disparities. Researchers should test models for fairness across demographic groups.
- **Not a substitute for real data**: Models trained on this dataset must be validated against real institutional data before deployment in production lending systems.

## References

1. **Zimbabwe National Statistics Agency (ZIMSTAT)**. *2022 Population and Housing Census*. Total population: 15,178,979. Provincial population shares used for geographic weighting. https://zimstat.co.zw/population-census/

2. **Zimbabwe National Statistics Agency (ZIMSTAT)**. *Quarterly Labour Force Survey, Q1 2024*. Employment: 3,289,853 employed; 41.3% informal; 83.4% earn < $362/mo. https://zimstat.co.zw/labour-statistic/

3. **Reserve Bank of Zimbabwe (RBZ)**. *Quarterly Banking Sector Report, December 2024*. NPL ratio: 2.09% (end-2023); capital adequacy: 46.15% (June 2024). https://www.rbz.co.zw/

4. **Reserve Bank of Zimbabwe (RBZ)**. *Quarterly Microfinance Report, June 2024*. Microfinance loans: ZiG 1.8 billion (US$72.33 million). https://www.rbz.co.zw/

5. **Zimbabwe Association of Microfinance Institutions (ZAMFI)**. *Analysis of the Performance of the Microfinance Sector in Zimbabwe, September 2024*. PAR30: 8.3%. https://zamfi.org/

6. **IFC (2024)**. *Portfolio Default Rate Analysis*. Africa default rate: 6.7% (1986–2023); low-income economies: 8.6%.

7. **Nikhil1e9 (2024)**. *Loan Default Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default/data — Studied as a conceptual reference for understanding credit risk feature relationships. No records were copied or sampled.

8. **Money & Moves / Tinashe Mukogo (2025)**. *"Visual: Who Are Zimbabwean Banks Lending To?"* — Bank lending rates: personal 13.49%–17.59%, business 10.27%–15.80% (RBZ Aug 2025 data).

9. **ZAMFI/Newsday Zimbabwe (2024)**. *"High interest rates choke microfinance sector"* — MFI rates driven by 35% policy rate.

10. **Zindi Africa (2024)**. *African Machine Learning Competitions*. https://zindi.africa/ — Platform for African data science challenges, referenced for dataset quality standards.

## Generation Methodology (Technical)

The dataset was generated using a calibrated logistic risk function implemented in Python. The default probability for each loan is computed as:

```
P(default) = σ(β₀ + Σᵢ βᵢxᵢ + ε)

where:
  σ(z) = 1 / (1 + e⁻ᶻ)                    [sigmoid function]
  β₀ = -1.9                                [base log-odds, calibrated for ~24% default rate]
  βᵢ = risk factor coefficients           [11 factors: income, DTI, age, sector, collateral, 
                                            obligations, interest rate, product, province, 
                                            dependents, employment tenure]
  ε ~ N(0, 0.3)                           [Gaussian noise for stochastic variance]
```

**Risk factor weights (β coefficients):**
- β_income: -0.4 to +0.6 (piecewise: low income → higher risk)
- β_DTI: -0.3 to +0.7 (piecewise: high debt burden → higher risk)
- β_age: -0.2 to +0.3 (piecewise: young/pre-retirement → higher risk)
- β_sector: -0.3 to +0.5 (Informal +0.5, Government/Finance/NGO -0.3)
- β_collateral: -0.4 to +0.35 (None +0.35, Property -0.4)
- β_obligations: -0.2 to +0.4 (0 obligations -0.2, >3 obligations +0.4)
- β_rate: -0.25 to +0.35 (piecewise: >120% +0.35, 60-120% +0.15, <20% -0.25)
- β_product: -0.3 to +0.4 (Emergency +0.4, Salary-Based -0.3)
- β_province: -0.1 to +0.15 (urban -0.1, Matabeleland +0.15)
- β_dependents: +0.25 if ≥5 dependents
- β_tenure: -0.15 to +0.2 (<6 months +0.2, >60 months -0.15)

All feature distributions (income, loan amount, interest rate, collateral, etc.) were independently generated using triangular, weighted categorical, and conditional distributions calibrated to ZIMSTAT 2022 Census, ZIMSTAT Q1 2024 QLFS, RBZ banking/microfinance reports, and Zimbabwe mobile money market data.

**Reproducibility:** Generation script uses fixed random seed (42) and reads no external data files, enabling full standalone reproducibility.

## Dataset Creator

**Created by:** Shannon Tafadzwa Sikadi  
**Affiliation:** Deep Learning Indaba X Zimbabwe 2026 Organizing Committee  
**Email:** shannonsikadi@gmail.com  
**GitHub:** [github.com/ShannonT20](https://github.com/ShannonT20)  
**LinkedIn:** [linkedin.com/in/shannon-sikadi-9370b3196](https://www.linkedin.com/in/shannon-sikadi-9370b3196/)  
**Zindi Profile:** [zindi.africa/users/Shannon_Sikadi](https://zindi.africa/users/Shannon%5FSikadi)

**About the Creator:** Shannon is a Business Systems Analyst and Data Scientist with expertise in process automation and machine learning for actuarial, banking, and insurance sectors. Passionate about leveraging AI for digital transformation and financial inclusion in Zimbabwe, Shannon is part of the Deep Learning Indaba X Zimbabwe organizing team working to strengthen AI capacity across Africa.

## Release Information

This dataset was used in the **Deep Learning IndabaX Zimbabwe 2026** hackathon challenge: “Can you predict who is likely to default on their loan?”

**Practical Relevance:** This dataset has already been used in a public IndabaX Zimbabwe 2026 hackathon workflow for model training, submission, and leaderboard evaluation. It is intended for education, prototyping, and research where access to real lending data is limited.

**Why This Matters:**  
- Supports ML education and hackathon use in an Africa-relevant credit-risk setting  
- Provides a synthetic benchmark for experimentation and comparison of modeling approaches  
- Preserves privacy by avoiding the use of real borrower records  
- Requires validation on real institutional data before any production use

**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)  
**Dataset Version:** 1.0

## Citation

If you use this dataset, please cite:

```
Shannon Tafadzwa Sikadi (2026).
Zimbabwe Loan Default Prediction Dataset.
Deep Learning Indaba X Zimbabwe 2026.
License: CC-BY-4.0
DOI: https://doi.org/10.5281/zenodo.20569260
```

## Contact

For questions about this dataset, please contact:

**Shannon Tafadzwa Sikadi**  
Email: shannonsikadi@gmail.com  

Or the Deep Learning Indaba X Zimbabwe organising committee.
