![Pipeline](https://github.com/Samir-AISS/Mmm-multi-market-bayesian/actions/workflows/mmm_pipeline.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyMC](https://img.shields.io/badge/PyMC-5.x-red?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-live-ff4b4b?logo=streamlit&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Cloud-blue?logo=prefect&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

# Marketing Mix Modeling — Hierarchical Bayesian MMM

A **production-grade Hierarchical Bayesian MMM** inspired by Google Meridian and Meta Robyn. Uses PyMC to build a single multi-geo model with shared priors across geographies — exactly like Meridian's geo-level hierarchical approach.

**[Live Dashboard →](https://mmm-multi-market-bayesian-jccbmodavvizmjzlcddjhc.streamlit.app)** · [Model Architecture](docs/model_architecture.md) · [Methodology](docs/methodology.md) · [Data Dictionary](docs/data_dictionary.md)

---

## What makes this Meridian-inspired?

| Feature | Google Meridian | This Project |
|---------|----------------|--------------|
| Hierarchical geo-level modeling | ✅ | ✅ PyMC |
| Hill saturation function | ✅ | ✅ |
| Geometric adstock decay | ✅ | ✅ |
| Shared priors across geos | ✅ | ✅ |
| Non-centered parametrization | ✅ | ✅ |
| Budget optimizer | ✅ | ✅ |
| Bayesian MCMC (NUTS) | ✅ TF Probability | ✅ PyMC |
| Convergence (R-hat) | < 1.05 | **1.01 ✅** |
| Speed | Slow (TFP) | 2-20x faster |

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | Google Meridian simulated (geo_all_channels.csv) |
| Geos modeled | 5 (configurable up to 40) |
| Weeks | 156 (2021–2024) |
| Channels | 5 paid + 1 organic |
| Max R-hat | **1.01** ✅ (converged) |
| Draws | 500 · Tune 1000 · 2 chains |
| Global μ_beta | 0.15 → 0.35 per channel |
| Baseline (geo) | 0.57 → 0.65 |

---

## Model Architecture

```
Revenue(g,t) = baseline(g)
             + Σ_c β(g,c) × Hill(Adstock(spend(g,t,c)))
             + Σ_k γ_k × control(g,t,k)

Hierarchical priors :
  β(g,c)      ~ Normal(μ_β(c), σ_β(c))   ← shared across geos
  μ_β(c)      ~ HalfNormal(1.0)           ← global channel mean
  σ_β(c)      ~ HalfNormal(0.5)           ← cross-geo variance
  baseline(g) ~ Normal(μ_base, σ_base)    ← geo-specific
  decay       = 0.4 (fixed, pre-computed) ← geometric adstock
```

---

## Dataset — Google Meridian

| Property | Value |
|----------|-------|
| Source | [google/meridian](https://github.com/google/meridian/tree/main/meridian/data/simulated_data/csv) |
| File | `geo_all_channels.csv` |
| Geos | 40 (Geo0 → Geo39) |
| Weeks | 156 (2021–2024) |
| Channels | 5 paid + 1 organic |
| Controls | competitor_sales, sentiment_score, Promo |
| KPI | conversions × revenue_per_conversion |

---

## Project Structure

```
mmm-multi-market-bayesian/
├── .github/workflows/
│   └── mmm_pipeline.yml          # CI/CD
├── app/
│   └── streamlit_app.py          # Dashboard
├── data/
│   ├── meridian/                 # Google Meridian datasets ← NEW
│   │   └── geo_all_channels.csv
│   └── synthetic/                # Original synthetic data
├── docs/
│   ├── methodology.md
│   ├── model_architecture.md     ← NEW
│   └── data_dictionary.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_market_poc.ipynb
│   ├── 03_multi_market_training.ipynb
│   ├── 04_model_diagnostics.ipynb
│   └── 05_hierarchical_mmm.ipynb ← NEW
├── results/
│   ├── precomputed.pkl
│   └── hierarchical/             ← NEW
│       └── hierarchical_5geos.pkl
├── scripts/
│   ├── precompute.py
│   ├── download_meridian_data.py ← NEW
│   └── train_hierarchical.py     ← NEW
└── src/
    └── models/
        ├── bayesian_mmm.py
        └── hierarchical_mmm.py   ← NEW
```

---

## Quick Start

```bash
git clone https://github.com/Samir-AISS/Mmm-multi-market-bayesian.git
cd Mmm-multi-market-bayesian
pip install -r requirements.txt

# Download Meridian data
python scripts/download_meridian_data.py

# Train hierarchical model (interactive geo selection)
python scripts/train_hierarchical.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Two MMM Approaches

| | Original MMM | Hierarchical MMM |
|--|-------------|-----------------|
| Data | Synthetic (10 EU markets) | Meridian real-like (40 geos) |
| Model | 10 independent models | 1 hierarchical model |
| Priors | Market-specific | Shared across geos |
| R² | 0.965 | R-hat 1.01 ✅ |
| Use case | Portfolio demo | Production-grade |

---

## Contact

**Samir EL AISSAOUY** — Data Scientist / ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-samir--el--aissaouy-blue?logo=linkedin)](https://www.linkedin.com/in/samir-el-aissaouy)
[![Email](https://img.shields.io/badge/Email-elaissaouy.samir12%40gmail.com-red?logo=gmail)](mailto:elaissaouy.samir12@gmail.com)

---

*Inspired by [Google Meridian](https://github.com/google/meridian) and [Meta Robyn](https://github.com/facebookexperimental/Robyn)*