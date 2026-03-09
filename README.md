![MMM Pipeline](https://github.com/Samir-AISS/Mmm-multi-market-bayesian/actions/workflows/mmm_pipeline.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyMC](https://img.shields.io/badge/PyMC-5.x-red?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-live-ff4b4b?logo=streamlit&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Cloud-blue?logo=prefect&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

# Marketing Mix Modeling — Multi-Market Bayesian

A production-grade **Bayesian Marketing Mix Modeling** system analyzing marketing effectiveness across **10 European markets** and **5 channels**, inspired by Google Meridian and Meta Robyn.

**[Live Dashboard](https://mmm-multi-market-bayesian-jccbmodavvizmjzlcddjhc.streamlit.app)** · [Methodology](docs/methodology.md) · [API Reference](docs/api_reference.md)

---

## Results

| Market | R² | MAPE | Best Channel |
|--------|----|------|--------------|
| FR — France | 0.965 | 2.5% | Search |
| DE — Germany | 0.967 | 2.5% | Search |
| UK — United Kingdom | 0.964 | 2.6% | Search |
| IT — Italy | 0.961 | 2.6% | Search |
| ES — Spain | 0.963 | 2.7% | Search |
| NL — Netherlands | 0.966 | 2.5% | Search |
| BE — Belgium | 0.968 | 2.4% | Search |
| PL — Poland | 0.964 | 2.5% | Search |
| SE — Sweden | 0.965 | 2.5% | Search |
| NO — Norway | 0.967 | 2.5% | Search |

**Avg R² : 0.965 · Avg MAPE : 2.5% · 0 divergences**

---

## Architecture

```
mmm-multi-market-bayesian/
├── .github/workflows/      # CI/CD — GitHub Actions (weekly pipeline)
├── app/
│   └── streamlit_app.py    # Interactive dashboard (Streamlit Cloud)
├── config/
│   ├── model_config.yaml   # MCMC hyperparameters, channels config
│   ├── markets_config.yaml # Per-market parameters (GDP, seasonality...)
│   └── priors.yaml         # Bayesian priors (HalfNormal, Beta, Gamma)
├── data/
│   └── synthetic/
│       └── mmm_multi_market.csv   # 2080 rows × 14 columns
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_market_poc.ipynb
│   ├── 03_multi_market_training.ipynb
│   └── 04_model_diagnostics.ipynb
├── pipelines/
│   ├── airflow_dags/mmm_training_dag.py   # Airflow DAG
│   └── orchestration/
│       ├── run_pipeline.py                # Local orchestration
│       └── prefect_flow.py               # Prefect Cloud flow
├── results/
│   ├── precomputed.pkl     # Pre-trained model outputs (dashboard)
│   └── reports/            # Pipeline run reports
├── scripts/
│   └── precompute.py       # Generate precomputed.pkl
├── src/
│   ├── data/               # Generator, loader, validator, features
│   ├── models/             # BayesianMMM, adstock, saturation
│   ├── training/           # Distributed trainer, diagnostics, tuning
│   ├── evaluation/         # Metrics, ROI calculator, validation
│   └── utils/              # Logging, visualization
└── tests/                  # 59 unit tests (pytest)
```

---

## Methodology

```
Revenue(t) = BaseSales
           + Σ [ β_i × Adstock(Spend_i(t)) × Hill(Spend_i(t)) ]
           + γ_season × Seasonality(t)
           + γ_trend  × Trend(t)
           + γ_events × Events(t)
           + ε(t),   ε ~ Normal(0, σ)
```

**Adstock** — geometric carryover : `A[t] = spend[t] + decay × A[t-1]`

**Saturation** — Hill function : `S(x) = x^s / (K^s + x^s)`

**Priors** — weakly informative : `β ~ HalfNormal(1)` · `decay ~ Beta(2,2)` · `K ~ Gamma(3,1)`

Full details → [docs/methodology.md](docs/methodology.md)

---

## Stack

| Layer | Technology |
|-------|-----------|
| Modeling | PyMC 5.x · ArviZ · NumPy |
| Data | Pandas · Scikit-learn |
| Orchestration | Prefect Cloud · Airflow DAG |
| CI/CD | GitHub Actions (weekly) |
| Dashboard | Streamlit Cloud |
| Testing | pytest (59 tests) |
| Tracking | MLflow |

---

## Quick Start

```bash
git clone https://github.com/Samir-AISS/Mmm-multi-market-bayesian.git
cd Mmm-multi-market-bayesian
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data
python src/data/multi_market_generator.py

# Run full pipeline
python pipelines/orchestration/run_pipeline.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Tests

```bash
pytest tests/ -v
# 59 tests — adstock, saturation, metrics, data generator
```

---

## Dataset

- **2 080 rows** — 10 markets × 208 weeks (4 years)
- **14 columns** — revenue, 5 spend channels, seasonality, trend, events, promotions, competitor_price
- **Synthetic** — generated with calibrated parameters (adstock, Hill saturation, market-specific seasonality)
- **Validation** — 33 automated tests, 0 errors

---

## Contact

**Samir EL AISSAOUY** — Data Consultant · Data Engineer / Analyst

[LinkedIn](https://www.linkedin.com/in/samir-el-aissaouy) · Elaissaouy.samir12@gmail.com · Île-de-France

---

*Inspired by [Google Meridian](https://developers.google.com/meridian) and [Meta Robyn](https://github.com/facebookexperimental/Robyn)*