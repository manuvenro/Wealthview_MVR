# Wealthview_MVR
# WealthView: Institutional Quantitative Risk & Portfolio Management Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)
![License](https://img.shields.io/badge/License-Proprietary-red)

**WealthView** is a full-stack, institutional-grade B2B quantitative risk and portfolio management SaaS platform. It is engineered to bridge the gap between complex econometric modeling and actionable portfolio allocation for High-Net-Worth (HNW) asset managers and private equity environments.

![Dashboard Overview](docs/images/dashboard_main.png)

> **Note on Repository Contents:** Certain database schemas and client-side ingestion pipelines have been abstracted or omitted from this public repository to protect institutional data.

## 🧠 Core Quantitative Engines

The platform's backend is implemented from first principles rather than by calling pre-built financial libraries:

### 1. Equity & Attribution Module
*   **Fundamental Valuation:** Dynamic Discounted Cash Flow (DCF) modeling for single-name equities.
*   **Factor Attribution:** Implementation of **Fama-French 3-Factor models** utilizing Ordinary Least Squares (OLS) econometrics to dissect return streams.
*   **Risk-Adjusted Performance:** Automated calculation of Sharpe Ratio, Treynor Ratio, Information Ratio, and Jensen's Alpha against dynamic benchmarks.

![Equity Module DCF](docs/images/equity_dcf.png)

### 2. Tail-Risk & Stress Testing Engine
*   **Downside Risk:** Conditional Value at Risk (CVaR / Expected Shortfall) implementation for extreme market scenarios.
*   **Distribution Analysis:** Jarque-Bera testing for normality in return distributions.
*   **Stress Testing:** Scenario analysis simulating acute macroeconomic shocks and supply-side geopolitical escalations on portfolio exposure.

### 3. Derivatives Pricing
*   **American Options:** Built a **Cox-Ross-Rubinstein (CRR) Binomial Tree** from scratch to price American options (allowing for early exercise).
*   **Greeks Calculation:** Computed delta, gamma, and theta using finite difference methods over the binomial lattice.

## 🤖 Algorithmic Alpha Engine (In Progress)

This module functions as the core of my undergraduate thesis and focuses on systematic quantitative trading.

*   **Objective:** Building a systematic equity strategy to generate alpha in public equities.
*   **Methodological Approach:** Combining advanced factor construction with a robust backtesting infrastructure.
*   **Current Status:** In active development (Not yet yielding production results).
*   **Tech Stack:** Python (NumPy, Pandas).

## ⚙️ Technical Architecture

*   **Frontend:** Built entirely in Python using **Streamlit** for highly reactive, data-heavy financial dashboards.
*   **Backend & Math:** Heavy utilization of `NumPy`, `Pandas`, and `SciPy` for matrix operations and econometric modeling.
*   **Data Pipeline:** Automated, asynchronous real-time financial data ingestion via REST APIs (**Financial Modeling Prep, Polygon.io, FRED**) managed by an `APScheduler` daemon.
*   **Database:** `SQLite` infrastructure for seamless client state management.

## 🧪 Testing & Validation

Numerical accuracy is enforced through rigorous unit testing via `pytest`. The test suite guarantees:
*   **CRR Convergence:** The Binomial Tree pricing algorithm strictly converges to the Black-Scholes formula for European options as the number of steps increases.
*   **OLS Robustness:** The Fama-French regression engine successfully recovers known beta coefficients on synthetic datasets.
*   **Risk Monotonicity:** CVaR strictly bounds VaR ($CVaR \le VaR$) across all confidence intervals.

Run the test suite locally:
```bash
pytest tests/test_quant_validation.py -v

**Local Installation:**





   
