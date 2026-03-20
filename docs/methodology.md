# Methodology

## Marketing Mix Modeling

Marketing Mix Modeling (MMM) is a statistical technique that measures the incremental impact of marketing activities on business KPIs (revenue, conversions). It uses aggregate data — no user-level tracking required — making it privacy-safe.

### Key Questions MMM Answers
- Which channels drive the most revenue?
- What is the ROI of each channel?
- How should I allocate my budget?
- What would happen if I increased spend on Channel X?

---

## Bayesian Approach

Unlike frequentist MMM, our Bayesian approach:

1. **Quantifies uncertainty** — every estimate comes with credible intervals
2. **Incorporates prior knowledge** — ROI priors from industry benchmarks
3. **Handles small samples** — hierarchical priors share information across geos
4. **Propagates uncertainty** — from parameters to ROI to budget recommendations

---

## Media Transformations

### 1. Adstock (Carryover Effect)

Marketing spend has a lagged effect — TV ads seen today affect purchases next week.

```
Adstock[t] = spend[t] + decay × Adstock[t-1]
```

- `decay ∈ [0, 1]` — higher = longer carryover
- Estimated per channel per geo (hierarchical)

### 2. Hill Saturation (Diminishing Returns)

More spend = less incremental impact per dollar.

```
Hill(x) = x^slope / (ec50^slope + x^slope)
```

- `ec50` — spend level at 50% saturation
- `slope` — steepness of the curve
- Estimated globally (shared across geos)

---

## Hierarchical Structure

The key innovation vs. traditional MMM:

```
Traditional : 40 models × 156 obs = fragile estimates
Hierarchical: 1 model  × 6240 obs = robust estimates
```

Small geos with limited data borrow statistical strength from larger geos through shared priors — exactly as implemented in Google Meridian.

---

## Budget Optimization

Given a total budget B, find the allocation {s_c} that maximizes revenue:

```
max  Σ_c f_c(s_c)
s.t. Σ_c s_c = B
     s_c ≥ 0
```

Where `f_c(s)` is the estimated revenue response curve for channel c.

Solved using `scipy.optimize.minimize` with posterior mean parameters.