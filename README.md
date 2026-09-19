# Wealthview_MVR
# WealthView: Institutional Quantitative Risk & Portfolio Management Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)
![License](https://img.shields.io/badge/License-Proprietary-red)

**WealthView** is a full-stack, institutional-grade B2B quantitative risk and portfolio management SaaS platform. It is engineered to bridge the gap between complex econometric modeling and actionable portfolio allocation for High-Net-Worth (HNW) asset managers and private equity environments.

> **Note on Repository Contents:** Certain proprietary algorithms, database schemas, and client-side ingestion pipelines have been abstracted or omitted from this public repository to protect institutional data and IP.

## 🧠 Core Quantitative Engines

The platform's backend is driven by several advanced financial models built from scratch without reliance on black-box external libraries:

### 1. Equity & Attribution Module
*   **Fundamental Valuation:** Dynamic Discounted Cash Flow (DCF) modeling for single-name equities.
*   **Factor Attribution:** Implementation of **Fama-French 3-Factor models** utilizing Ordinary Least Squares (OLS) econometrics to dissect return streams.
*   **Risk-Adjusted Performance:** Automated calculation of Sharpe Ratio, Treynor Ratio, Information Ratio, and Jensen's Alpha against dynamic benchmarks.

### 2. Tail-Risk & Stress Testing Engine
*   **Downside Risk:** Conditional Value at Risk (CVaR / Expected Shortfall) implementation for extreme market scenarios.
*   **Distribution Analysis:** Jarque-Bera testing for normality in return distributions.
*   **Stress Testing:** Scenario analysis simulating acute macroeconomic shocks and supply-side geopolitical escalations on portfolio exposure.

### 3. Derivatives Pricing
*   **American Options:** Built a **Cox-Ross-Rubinstein (CRR) Binomial Tree** from scratch to price American options (allowing for early exercise).
*   **Greeks Calculation:** Computed delta, gamma, and theta using finite difference methods over the binomial lattice.

## ⚙️ Technical Architecture

*   **Frontend:** Built entirely in Python using **Streamlit** for highly reactive, data-heavy financial dashboards.
*   **Backend & Math:** Heavy utilization of `NumPy`, `Pandas`, and `SciPy` for matrix operations and econometric modeling.
*   **Data Pipeline:** Automated, asynchronous real-time financial data ingestion via REST APIs (**Financial Modeling Prep, Polygon.io, FRED**) managed by an `APScheduler` daemon.
*   **Database:** `SQLite` infrastructure for seamless client state management.

## 🚀 Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/wealthview.git
   cd wealthview

2. **Set up the virtual enviroment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Enviroment Variables:**
   ```bash
   cp .env.example .env

5. **Run the platform:**
   ```bash
   streamlit run app.py


**Future Roadmap**
### Implementation of the Algorithmic Alpha Engine: A systematic equity strategy combining factor construction with backtesting infrastructure to generate alpha in public equities.

**Developed and maintained by Manuel Ventosa Rodríguez.**





   
