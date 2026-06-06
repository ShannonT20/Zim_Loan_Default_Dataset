# Data Card: Zimbabwe Loan Default Prediction Dataset

**A calibrated synthetic microfinance and SME loan dataset modeling Zimbabwean retail lending dynamics under privacy constraints.**

---

## Identity

| Field | Value |
|---|---|
| **Name** | Zimbabwe Loan Default Prediction Dataset |
| **Version** | 1.0 |
| **Release date** | 2026 |
| **Records** | 38,932 |
| **Features** | 22 (including binary target) |
| **Format** | CSV (UTF-8) |
| **License** | CC-BY-4.0 |
| **Language** | English |
| **Geographic scope** | Zimbabwe (all 10 provinces) |
| **Temporal scope** | Loans disbursed October 2024 – February 2026 |
| **Domain** | Financial services — microfinance and SME lending |
| **Task** | Binary classification (loan default prediction) |

---

## Why This Dataset Exists

Banking data across Africa is often difficult to access due to regulatory restrictions, banking secrecy laws, and institutional confidentiality. The background search for this project did not identify a suitable openly accessible Zimbabwe-focused loan default dataset for this use case. Most widely shared loan datasets (for example on Kaggle or UCI) reflect different financial contexts and may not transfer well.

This dataset was created as a calibrated, privacy-preserving synthetic resource to support ML education, prototyping, and early-stage credit-risk research without requiring institutional data partnerships.

---

## Calibration Sources

All features are independently generated — **no records from any external dataset were copied or sampled**. Distributions were calibrated to:

| Source | Data Used | Reference |
|---|---|---|
| ZIMSTAT 2022 Census | Provincial population (15.18M across 10 provinces, 39% urban) | zimstat.co.zw |
| ZIMSTAT Q1 2024 QLFS | Employment sectors (41.3% informal), income (83.4% earn < $362/mo) | zimstat.co.zw |
| RBZ Banking Reports | NPL ratio (2.09% end-2023), capital adequacy (46.15%); bank lending rates (personal 13.49–17.59%/yr, business 10.27–15.80%/yr, RBZ Aug 2025); policy rate 35% | rbz.co.zw |
| RBZ/ZAMFI Microfinance | PAR30: 8.3% (Sep 2024), 14.58% (Jun 2019); loans: US$72.33M | rbz.co.zw, zamfi.org |
| Mobile money market data | EcoCash ~8.35M users, InnBucks ~3M users | Econet, TechZim |
| Microfinance rate data | MFI lending rates: 7–15% per month (84–180% annualized); rates exceed RBZ 35% policy rate | ZAMFI, MicroLoan Foundation |
| Kaggle (Nikhil1e9, 2024) | Conceptual reference for credit risk feature relationships only | kaggle.com |

The Kaggle dataset was studied to understand general relationships between borrower features and default outcomes. **No records were read from, copied, or sampled from it.** All generation logic, distributions, and values are independently designed for the Zimbabwe context.

---

## Default Probability: Calibrated Logistic Risk Function

Default is **not randomly assigned**. Each loan's default probability is computed via a logistic function incorporating **11 risk factors**:

```
P(default) = σ(β₀ + Σ βᵢxᵢ + ε),  ε ~ N(0, 0.3)
```

| # | Risk Factor | Direction | Mechanism |
|---|---|---|---|
| 1 | Monthly income | ↑ income → ↓ risk | Repayment capacity |
| 2 | DTI ratio | ↑ DTI → ↑ risk | Debt burden relative to income |
| 3 | Borrower age | Prime-age (36–50) → ↓ risk | Employment/income stability |
| 4 | Employment sector | Informal → ↑ risk; Government/Finance → ↓ risk | Income volatility |
| 5 | Collateral type | None → ↑ risk; Property → ↓ risk | Loss-given-default buffer |
| 6 | Existing obligations | ↑ obligations → ↑ risk | Over-indebtedness |
| 7 | Interest rate | ↑ rate → ↑ risk | Adverse selection / repayment burden |
| 8 | Loan product | Emergency → ↑ risk; Salary-Based → ↓ risk | Product risk profile |
| 9 | Province | Urban → slightly ↓ risk | Economic resilience |
| 10 | Dependents | ≥ 5 → ↑ risk | Financial strain |
| 11 | Employment tenure | < 6 months → ↑ risk; 60+ → ↓ risk | Job stability |

---

## Validation Evidence

### Risk Factors — All Monotonic and Interpretable

| Risk Factor | Low-Risk Group | Default Rate | High-Risk Group | Default Rate | Spread |
|---|---|---|---|---|---|
| Income | Q4 ($687–$2,769) | 17.9% | Q1 ($32–$276) | 34.0% | 1.9x |
| DTI ratio | < 0.1 | 12.3% | > 0.5 | 29.0% | 2.4x |
| Sector | NGO | 13.3% | Informal | 36.7% | 2.8x |
| Collateral | Property | ~18% | None | ~29% | 1.6x |
| Product | Salary-Based | 13.0% | Emergency | 39.9% | 3.1x |
| Obligations | 0 | 20.2% | 5+ | 30.5% | 1.5x |
| Age | 36–50 | 20.2% | 18–25 | 28.9% | 1.4x |
| Employment tenure | 60+ months | 19.0% | < 6 months | 32.6% | 1.7x |

### Distributions — Realistic, Not Uniform

- **Income**: Right-skewed (mean $543 > median $443); 38.3% of borrowers earn < $362/mo (lower than general pop's 83.4% because borrowers are a wealthier, bankable subset)
- **Loan amount ↔ Income**: Pearson r = 0.587
- **Interest rates**: Bimodal structure reflecting Zimbabwe's dual lending market — bank-sourced loans (10–20% annual) and MFI-sourced loans (60–200% annual). Range: 8.2%–202.2%, mean 69.1%. Calibrated to RBZ Aug 2025 bank rates (personal 13.49–17.59%/yr, business 10.27–15.80%/yr) and MFI rates (7–15%/month = 84–180%/yr, per ZAMFI).

**Interest Rate by Product:**

| Product | Mean Rate | Range |
|---|---|---|
| Salary-Based | 21.8% | 10.1% – 119.3% |
| Asset Finance | 31.2% | 12.1% – 127.8% |
| SME | 72.5% | 10.0% – 155.9% |
| Agriculture | 74.0% | 8.2% – 142.8% |
| Personal | 87.0% | 13.0% – 179.7% |
| Emergency | 145.5% | 15.8% – 202.2% |

- **Disbursement channels**: Province-aware (rural: 28% cash vs urban: 11% cash)
- **Overall default rate**: 24.1% — consistent with mixed retail/microfinance portfolio benchmarks

### Default Rate Contextualization

| Benchmark | Rate | Source |
|---|---|---|
| Zimbabwe formal banking NPL | 2.09% | RBZ, end-2023 |
| Zimbabwe microfinance PAR30 | 8.3% | ZAMFI, Sep 2024 |
| Zimbabwe microfinance PAR30 (crisis) | 14.58% | RBZ, Jun 2019 |
| IFC Africa corporate default | 6.7% | IFC, 1986–2023 |
| **This dataset (mixed portfolio)** | **24.1%** | **Lifetime cumulative, incl. emergency/informal** |

Note: PAR30 measures loans 30+ days overdue at a point in time. Lifetime cumulative default rates over full loan terms are substantially higher, especially for portfolios that include unsecured emergency loans (39.9% default) and informal sector borrowers (36.7% default).

---

## Missing Values (Intentional Data Quality Challenge)

| Column | Missing Rate | Rationale |
|---|---|---|
| `monthly_income_usd` | ~8% | Forces imputation strategy for a key predictive feature |
| `num_dependents` | ~6% | Demographic missingness |
| `months_at_employer` | ~5% | Employment stability gap |
| `collateral_type` | ~4% | Collateral documentation gap |
| `employment_sector` | ~3% | Informal employment underreporting |
| `loan_purpose` | ~2% | Purpose not always captured |
| `marital_status` | ~1.5% | Demographic missingness |
| `annual_rate_pct` | ~1% | Tricky for feature engineering |

---

## Limitations

1. **Synthetic, not observed** — relationships are modeled via a calibrated logistic function, not learned from real Zimbabwean loan performance data
2. **No temporal dynamics** — does not model macroeconomic shocks, currency instability (ZiG), policy changes, or seasonal effects
3. **No longitudinal features** — single point-in-time snapshots; no repayment trajectory or credit bureau history
4. **No behavioral features** — no mobile money transaction patterns, savings behavior, or repayment regularity
5. **Simplified income model** — sector-specific triangular distributions; real distributions have heavier tails
6. **Single-country scope** — calibrated to Zimbabwe; not generalizable to other African markets without recalibration

---

## Ethical Statement

- **Fully synthetic**: No real individuals' data was accessed, collected, or used. No records from any external dataset were copied. Since the data is not derived from real borrower records, there is no plausible path to re-identify any real borrower from this dataset.
- **No regulatory conflict**: No banking institution's proprietary data was involved.
- **Bias transparency**: The risk function embeds sector and geographic effects reflecting real-world disparities.
- **Not a production substitute**: Models must be validated against real data before deployment.

---

## Reproducibility

The dataset was generated with a fixed random seed (42) and reads no external data files. Generation code is not included in this package.

---

## References

1. ZIMSTAT (2022). *Population and Housing Census*. https://zimstat.co.zw/population-census/
2. ZIMSTAT (2024). *Quarterly Labour Force Survey, Q1 2024*. https://zimstat.co.zw/labour-statistic/
3. RBZ (2024). *Quarterly Banking Sector Report, December 2024*. https://www.rbz.co.zw/
4. ZAMFI (2024). *Microfinance Sector Analysis, September 2024*. https://zamfi.org/
5. IFC (2024). *Portfolio Default Rate Analysis*.
6. Nikhil1e9 (2024). *Loan Default Prediction Dataset*. Kaggle. Conceptual reference only.
7. Gebru, T. et al. (2021). *Datasheets for Datasets*. Communications of the ACM, 64(12).
8. Money & Moves (2025). *Bank lending rates from RBZ August 2025 data*. Personal 13.49%–17.59%/yr, business 10.27%–15.80%/yr.
9. ZAMFI/Newsday (2024). *"High interest rates choke microfinance sector"* — MFI rates above 35% RBZ policy rate.
10. Zindi Africa (2024). *African Machine Learning Competitions*. https://zindi.africa/ — Platform for African data science challenges, referenced for dataset quality standards.

---

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

---

## Dataset Creator & Contact

**Created by:** Shannon Tafadzwa Sikadi  
**Affiliation:** Deep Learning Indaba X Zimbabwe 2026 Organizing Committee  
**Email:** shannonsikadi@gmail.com  
**GitHub:** [github.com/ShannonT20](https://github.com/ShannonT20)  
**LinkedIn:** [linkedin.com/in/shannon-sikadi-9370b3196](https://www.linkedin.com/in/shannon-sikadi-9370b3196/)  
**Zindi Profile:** [zindi.africa/users/Shannon_Sikadi](https://zindi.africa/users/Shannon%5FSikadi)

**About the Creator:** Shannon is a Business Systems Analyst and Data Scientist focused on process automation and machine learning in actuarial, banking, and insurance sectors. Passionate about leveraging AI for digital transformation and financial inclusion in Zimbabwe.

---

## Release Information

This dataset was used in the **Deep Learning IndabaX Zimbabwe 2026** hackathon challenge: “Can you predict who is likely to default on their loan?”

**Key contribution:** Provides an openly shareable Zimbabwe-calibrated synthetic benchmark for education, prototyping, and research in credit-risk modeling where access to real lending data is limited.

**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)  
**Submission Date:** March 2026  
**Dataset Version:** 1.0
