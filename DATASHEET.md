# Datasheet: Zimbabwe Loan Default Prediction Dataset

**A calibrated synthetic microfinance and SME loan dataset modeling Zimbabwean retail lending dynamics under privacy constraints.**

*Following the framework from "Datasheets for Datasets" (Gebru et al., 2021)*

---

## 1. Motivation

**For what purpose was the dataset created?**
This dataset was created to support machine learning research on loan default prediction in the context of Zimbabwe's banking and microfinance sector. Extensive background research was conducted to identify publicly available loan default datasets from Zimbabwe or the broader African continent, but that search did not identify a suitable openly accessible dataset for this use case. Banking data across Africa is often difficult to access due to regulatory restrictions, banking secrecy laws, confidentiality requirements, and institutional reluctance to release proprietary financial records. This data scarcity creates a barrier for researchers, students, and startups who want to develop AI solutions for the financial sector. A synthetic dataset can therefore be useful for safe experimentation, method development, and education.

**Who created the dataset and on behalf of which entity?**
The dataset was created by the Deep Learning Indaba X Zimbabwe community as part of the Indaba X Zimbabwe 2026 hackathon initiative, themed "Preparing Zimbabwe's Workforce for an AI-Driven Future."

**Who funded the creation of the dataset?**
This is a community-driven effort by the Deep Learning Indaba X Zimbabwe chapter.

---

## 2. Composition

**What do the instances represent?**
Each instance represents a single synthetic loan modeled on the types of loans disbursed by banks and microfinance institutions in Zimbabwe, along with borrower demographic information and whether the loan was eventually repaid or defaulted.

**How many instances are there in total?**
38,932 loan records.

**Does the dataset contain all possible instances or is it a sample?**
It is a synthetic dataset designed to capture realistic statistical patterns of Zimbabwe's lending sector. It does not represent a census of actual loans.

**What data does each instance consist of?**
Each instance contains 22 features covering:
- Loan characteristics (amount, rate, term, product type, purpose, collateral, payment frequency)
- Temporal information (approval date, disbursement date, first payment date, maturity date)
- Borrower demographics (gender, date of birth, marital status, dependents)
- Borrower financial profile (monthly income, employment sector, months at employer, existing obligations)
- Geographic information (province)
- Disbursement channel (including mobile money platforms, province-aware)
- Target variable (defaulted: binary)

**Is there a label or target associated with each instance?**
Yes. The `defaulted` column is a binary target variable (1 = loan defaulted, 0 = loan repaid). The overall default rate is approximately 24.1%.

**Is any information missing from individual instances?**
Yes. The dataset intentionally contains missing values across several columns to reflect real-world data quality challenges:
- `monthly_income_usd`: ~8% missing
- `num_dependents`: ~6% missing
- `months_at_employer`: ~5% missing
- `collateral_type`: ~4% missing
- `employment_sector`: ~3% missing
- `loan_purpose`: ~2% missing
- `marital_status`: ~1.5% missing
- `annual_rate_pct`: ~1% missing

Missing values are represented as empty strings in the CSV.

**Are there any errors, sources of noise, or redundancies?**
The dataset is synthetically generated and does not contain data entry errors in the traditional sense. The missing values are deliberately introduced as a data quality challenge. Gaussian noise (σ=0.3) was added to the logistic risk function to prevent deterministic default assignment and introduce realistic variance around the calibrated risk frontier.

**Is the default label mechanistically linked to features or randomly assigned?**
Default probabilities are mechanistically linked via a calibrated logistic risk function incorporating **11 risk factors**. Validation confirms monotonic, interpretable relationships:
- **Income**: Q1 borrowers default at 34.0% vs Q4 at 17.9% (1.9x spread)
- **DTI ratio**: Very High DTI defaults at 29.0% vs Low DTI at 12.3% (2.4x spread)
- **Employment sector**: Informal Sector at 36.7% vs NGO at 13.3% (2.8x spread)
- **Collateral**: Unsecured loans at ~29% vs Property-backed at ~18% (1.6x spread)
- **Product type**: Emergency at 39.9% vs Salary-Based at 13.0% (3.1x spread)
- **Existing obligations**: 5+ obligations at 30.5% vs 0 obligations at 20.2%
- **Age**: Young (18–25) at 28.9% vs Prime (36–50) at 20.2%
- **Employment tenure**: < 6 months at 32.6% vs 60+ months at 19.0% (1.7x spread)
- **Loan amount–income correlation**: Pearson r = 0.587 (amounts are income-dependent)
- **Interest rates**: Product-differentiated with bimodal structure — bank-sourced loans (10–20% annual) and MFI-sourced loans (60–200% annual). Salary-Based mean 21.8% (91% bank-sourced) vs Emergency mean 145.5% (almost entirely MFI-sourced). Range: 8.2%–202.2%, overall mean 69.1%.
- **Income distribution**: Right-skewed (mean $543 > median $443), consistent with developing economies

**Is the dataset self-contained?**
Yes. All data is contained in a single CSV file with accompanying documentation.

**Does the dataset contain data that might be considered confidential?**
No. All data is synthetically generated. No real individuals' information is included.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**
All data was **synthetically generated** — no real banking data was accessed or used, and **no records from any external dataset were copied or sampled**. All 38,932 records are independently generated from scratch. The generation process was informed by:

1. **ZIMSTAT 2022 Population and Housing Census**: Provincial population shares (total: 15,178,979 across 10 provinces; 39% urban) used as the basis for geographic lending weights.

2. **ZIMSTAT Q1 2024 Quarterly Labour Force Survey**: Employment sector distributions (41.3% informal, 22.9% agriculture, 23.9% retail), income benchmarks (83.4% earn < $362/mo, 34.4% earn < $90/mo), unemployment (20.5%), and labour force participation (48%).

3. **Reserve Bank of Zimbabwe (RBZ)**: Quarterly banking sector and microfinance reports used to calibrate interest rates and understand lending structures. Banking NPL ratio: 2.09% (end-2023); microfinance PAR30: 8.3% (September 2024, ZAMFI). Bank lending rates (RBZ Aug 2025): personal 13.49%–17.59%/yr, business 10.27%–15.80%/yr. RBZ policy rate: 35%. Microfinance institutions charge 7%–15% per month (84%–180% annualized), well above the policy rate.

4. **Mobile money market data**: EcoCash's ~8.35 million subscribers, InnBucks' ~3 million subscribers, OneMoney's marginal share — used to calibrate province-aware disbursement channel distributions.

5. **Kaggle conceptual reference**: The [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data) (Nikhil1e9, 2024) was studied to understand general statistical relationships between borrower characteristics and default outcomes. **No records were read from, copied from, or sampled from this dataset during generation.** It served only as a conceptual reference for understanding what features matter in credit risk modeling.

**Why synthetic data instead of real data?**
Banking data is often hard to access due to regulation and confidentiality. Zimbabwe's banking institutions, like many elsewhere, are subject to banking secrecy laws and data protection regulations that limit the release of loan-level data. Even anonymized datasets can be difficult to obtain — the RBZ typically publishes aggregate statistics. A synthetic approach makes it possible to share a privacy-preserving dataset for research, benchmarking, and teaching without relying on real borrower records.

**What mechanisms or procedures were used to collect the data?**
Synthetic generation via Python, using:
- Independently generated borrower ages (mixture of triangular distributions, not sourced from external data)
- Sector-specific triangular income distributions calibrated to ZIMSTAT benchmarks
- Product-conditional collateral, purpose, and interest rate distributions
- Province-aware disbursement channel distributions (urban/semi-urban/rural)
- Calibrated logistic risk function with 11 factors and Gaussian noise for default probability
- Fixed random seed (42) for reproducibility
- No external data files read during generation

**Who was involved in the data collection process?**
The Deep Learning Indaba X Zimbabwe organizing team.

**Over what timeframe was the data collected?**
The synthetic data simulates loans disbursed between October 2024 and February 2026. The generation script was developed in 2026.

**Were any ethical review processes conducted?**
As the dataset is entirely synthetic, no IRB or ethical review was required. No real personal data was collected, accessed, or used. No banking institution's proprietary data was involved. No records from any external dataset were copied.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling applied?**
The data was generated in its final form. Post-generation processing included:
- Shuffling rows to remove ordering artifacts
- Injecting missing values at specified rates across 8 columns
- Validating date consistency (first payment before maturity)
- Validating value ranges (amounts, rates, incomes)

**Was the raw data saved in addition to the processed data?**
The generation script is retained, allowing full reproducibility with a fixed random seed (42). No external data files are required to regenerate the dataset.

---

## 5. Uses

**What tasks has the dataset been used for?**
- Loan default prediction (binary classification)
- Hackathon challenge for Indaba X Zimbabwe 2026
- AI/ML education in the context of African financial data

**What other tasks could the dataset be used for?**
- Credit scoring and risk assessment research
- Financial inclusion analysis across provinces and demographics
- Feature engineering benchmarks (date processing, categorical encoding, imputation)
- Fairness and bias analysis in lending
- Explainable AI (XAI) research
- Missing data imputation technique evaluation

**Is there anything about the composition or collection that might impact future uses?**
- The data is synthetic and should not be used to make actual lending decisions without validation against real portfolio data
- Default probability patterns are generated via a calibrated logistic function with Gaussian noise — real-world dynamics may involve non-linearities, macroeconomic shocks, and temporal patterns not captured here
- The dataset is calibrated to Zimbabwe's specific economic context; cross-country generalization requires recalibration
- No longitudinal or behavioral features (repayment trajectory, transaction patterns, credit bureau history)
- The 24.1% default rate reflects a mixed retail/microfinance portfolio; formal banking NPLs in Zimbabwe are much lower (~2%)

---

## 6. Distribution

**How is the dataset distributed?**
As a CSV file with accompanying documentation (README, datasheet, data card, variable definitions, license).

**When was the dataset first released?**
2026

**What license is the dataset distributed under?**
Creative Commons Attribution 4.0 International (CC-BY-4.0)

---

## 7. Maintenance

**Who is supporting/hosting/maintaining the dataset?**
Shannon Tafadzwa Sikadi (creator) and the Deep Learning Indaba X Zimbabwe community.

**Who created the dataset?**
Shannon Tafadzwa Sikadi — Business Systems Analyst and Data Scientist, Deep Learning Indaba X Zimbabwe 2026 Organizing Committee.

**Contact Information:**
- **Email:** shannonsikadi@gmail.com
- **GitHub:** [github.com/ShannonT20](https://github.com/ShannonT20)
- **LinkedIn:** [linkedin.com/in/shannon-sikadi-9370b3196](https://www.linkedin.com/in/shannon-sikadi-9370b3196/)
- **Zindi Profile:** [zindi.africa/users/Shannon_Sikadi](https://zindi.africa/users/Shannon%5FSikadi)

**Will the dataset be updated?**
Future versions may be released with additional features, larger sample sizes, or refined generation models incorporating feedback from the research community.

**How can the owner/curator/manager of the dataset be contacted?**
Email: shannonsikadi@gmail.com or through the Deep Learning Indaba X Zimbabwe organizing committee.

**Is there an erratum?**
No errata at the time of release.

---

## 8. Generation Methodology (Technical)

The dataset was generated using a calibrated logistic risk function:

```
P(default) = σ(β₀ + Σᵢ βᵢxᵢ + ε)

where:
  σ(z) = 1 / (1 + e⁻ᶻ)                    [sigmoid function]
  β₀ = -1.9                                [base log-odds]
  βᵢ = risk factor coefficients           [11 factors]
  ε ~ N(0, 0.3)                           [Gaussian noise]
```

**11 Risk Factors (β coefficients):**
- Income level (β: -0.4 to +0.6)
- Debt-to-income ratio (β: -0.3 to +0.7)
- Borrower age (β: -0.2 to +0.3)
- Employment sector (β: -0.3 to +0.5)
- Collateral type (β: -0.4 to +0.35)
- Existing obligations (β: -0.2 to +0.4)
- Interest rate tier (β: -0.25 to +0.35)
- Loan product (β: -0.3 to +0.4)
- Province/urbanity (β: -0.1 to +0.15)
- Dependents (β: +0.25 if ≥5)
- Employment tenure (β: -0.15 to +0.2)

**Feature Distributions:** All features (income, loan amount, interest rate, collateral, etc.) were independently generated using triangular and weighted categorical distributions calibrated to ZIMSTAT Census 2022, ZIMSTAT QLFS Q1 2024, RBZ banking reports, and Zimbabwe mobile money data.

**Code:** Generation code (not included in this package) used a fixed seed (42) for reproducibility.

---

## 9. References

1. Zimbabwe National Statistics Agency (ZIMSTAT). *2022 Population and Housing Census*. https://zimstat.co.zw/population-census/
2. Zimbabwe National Statistics Agency (ZIMSTAT). *Quarterly Labour Force Survey, Q1 2024*. https://zimstat.co.zw/labour-statistic/
3. Reserve Bank of Zimbabwe (RBZ). *Quarterly Banking Sector Report, December 2024*. https://www.rbz.co.zw/
4. Reserve Bank of Zimbabwe (RBZ). *Quarterly Microfinance Report, June 2024*. https://www.rbz.co.zw/
5. Zimbabwe Association of Microfinance Institutions (ZAMFI). *Analysis of the Performance of the Microfinance Sector in Zimbabwe, September 2024*. PAR30: 8.3%. https://zamfi.org/
6. IFC (2024). *Portfolio Default Rate Analysis*. Africa default rate: 6.7% (1986–2023).
7. Nikhil1e9 (2024). *Loan Default Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default/data — Conceptual reference only; no records copied.
8. Gebru, T. et al. (2021). *Datasheets for Datasets*. Communications of the ACM, 64(12), pp.86-92.
9. Money & Moves (2025). *Bank lending rates from RBZ August 2025 data*. Personal 13.49%–17.59%/yr, business 10.27%–15.80%/yr.
10. ZAMFI/Newsday (2024). *"High interest rates choke microfinance sector"* — MFI rates above 35% RBZ policy rate.
11. Zindi Africa (2024). *African Machine Learning Competitions*. https://zindi.africa/ — Referenced for dataset quality standards.
