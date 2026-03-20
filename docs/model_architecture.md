# Model Architecture

## Overview

This project implements a **Hierarchical Bayesian MMM** inspired by Google Meridian. Unlike traditional MMMs that train one model per market, this model trains a **single model across all geos simultaneously**, sharing statistical strength between geographies.

---

## Why Hierarchical?

**Traditional approach (our original model):**
```
FR → Model_FR (208 obs)
DE → Model_DE (208 obs)
UK → Model_UK (208 obs)
...
10 separate models
```

**Hierarchical approach (Meridian-inspired):**
```
All geos → Single Hierarchical Model (6240 obs)
           ↓
    Global priors (shared)
    ├── Geo0 params
    ├── Geo1 params
    └── ... Geo39 params
```

**Benefits:**
- Small geos "borrow strength" from larger geos
- Tighter credible intervals on ROI estimates
- More robust estimates with limited data per geo
- Single model easier to maintain and deploy

---

## Mathematical Specification

### Likelihood

```
Revenue(g,t) ~ Normal(μ(g,t), σ)

μ(g,t) = baseline(g)
        + Σ_c β(g,c) × Hill(Adstock(spend(g,t,c)))
        + Σ_k γ_k × control(g,t,k)
```

Where:
- `g` = geo index
- `t` = time index
- `c` = channel index
- `k` = control variable index

---

### Adstock (Geometric Decay)

```
Adstock(x, decay)[t] = x[t] + decay × Adstock(x, decay)[t-1]
```

Hierarchical decay prior:
```
decay(g,c) = clip(μ_decay(c) + σ_decay(c) × ε, 0.01, 0.99)
ε          ~ Normal(0, 1)
μ_decay(c) ~ Beta(2, 2)        ← centered around 0.5
σ_decay(c) ~ HalfNormal(0.1)   ← small variance
```

---

### Hill Saturation

```
Hill(x; ec50, slope) = x^slope / (ec50^slope + x^slope)
```

Global priors (shared across geos):
```
ec50(c)  ~ HalfNormal(0.5)
slope(c) ~ HalfNormal(2.0)
```

---

### Channel Effects (Hierarchical)

```
β(g,c) = μ_β(c) + σ_β(c) × β_raw(g,c)    ← non-centered
β_raw(g,c) ~ Normal(0, 1)
μ_β(c)     ~ HalfNormal(1.0)               ← global mean
σ_β(c)     ~ HalfNormal(0.5)               ← cross-geo variance
```

This **non-centered parametrization** improves MCMC sampling efficiency (as recommended by Meridian and Stan guidelines).

---

### Baseline (Geo-specific)

```
baseline(g) ~ Normal(μ_base, σ_base)
μ_base      ~ Normal(1.0, 0.5)
σ_base      ~ HalfNormal(0.3)
```

---

### Controls

```
control_contribution(g,t) = Σ_k γ_k × control(g,t,k)
γ_k ~ Normal(0, 0.5)
```

Controls included:
- `competitor_sales_control` — competitor pressure
- `sentiment_score_control` — market sentiment
- `Promo` — promotional activity

---

## Comparison with Meridian

| Component | Meridian | This Model |
|-----------|----------|------------|
| Backend | TensorFlow Probability | PyMC 5.x |
| Sampler | HMC/NUTS | NUTS |
| Hierarchical structure | Geo-level | Geo-level |
| Adstock | Geometric/Binomial | Geometric |
| Saturation | Hill | Hill |
| Non-centered parametrization | ✅ | ✅ |
| ROI priors | ✅ | ✅ (via μ_β) |
| Population scaling | ✅ | ✅ |

---

## Inference

**Sampler**: NUTS (No-U-Turn Sampler)
**Target accept**: 0.9
**Convergence**: R-hat < 1.01, ESS > 400

```python
with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
    )
```

---

## ROI Calculation

```
ROI(g,c) = Σ_t [β(g,c) × Hill(Adstock(spend(g,t,c)))]
           / Σ_t spend(g,t,c)
```

Global ROI across geos:
```
ROI_global(c) = Σ_g Σ_t contrib(g,t,c) / Σ_g Σ_t spend(g,t,c)
```