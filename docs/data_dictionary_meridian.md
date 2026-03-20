# Data Dictionary — Meridian Dataset

## Source

Google Meridian simulated dataset — available at:
```
https://github.com/google/meridian/tree/main/meridian/data/simulated_data/csv
```

---

## Dataset: `geo_all_channels.csv`

**Principal dataset used for hierarchical modeling.**

| Column | Type | Description |
|--------|------|-------------|
| `geo` | STRING | Geography identifier (Geo0 → Geo39) |
| `time` | DATE | Week start date (weekly granularity) |
| `Channel0_impression` | FLOAT | Channel 0 media impressions |
| `Channel1_impression` | FLOAT | Channel 1 media impressions |
| `Channel2_impression` | FLOAT | Channel 2 media impressions |
| `Channel3_impression` | FLOAT | Channel 3 media impressions |
| `Channel4_impression` | FLOAT | Channel 4 media impressions |
| `Channel0_spend` | FLOAT | Channel 0 media spend ($) |
| `Channel1_spend` | FLOAT | Channel 1 media spend ($) |
| `Channel2_spend` | FLOAT | Channel 2 media spend ($) |
| `Channel3_spend` | FLOAT | Channel 3 media spend ($) |
| `Channel4_spend` | FLOAT | Channel 4 media spend ($) |
| `Organic_channel0_impression` | FLOAT | Organic channel impressions (no spend) |
| `competitor_sales_control` | FLOAT | Competitor sales index (standardized) |
| `sentiment_score_control` | FLOAT | Market sentiment score (standardized) |
| `Promo` | FLOAT | Promotional activity indicator |
| `conversions` | FLOAT | KPI — number of conversions |
| `revenue_per_conversion` | FLOAT | Revenue per conversion ($) |
| `population` | FLOAT | Geo population (for scaling) |

**Derived columns (computed in pipeline):**

| Column | Formula | Description |
|--------|---------|-------------|
| `revenue` | `conversions × revenue_per_conversion` | Total revenue ($) |
| `geo_idx` | label encoding | Integer index for geo |

---

## Dataset: `hypothetical_geo_all_channels.csv`

Similar structure but without `conversions` column — uses `revenue_per_conversion` directly as KPI proxy.

---

## Data Statistics

| Metric | Value |
|--------|-------|
| Total rows | 6,240 |
| Geos | 40 |
| Time period | 2021-01-25 → 2024-01-15 |
| Weeks per geo | 156 |
| Channels | 5 paid + 1 organic |
| Revenue range | ~$83K → ~$740K per week per geo |

---

## Data Flow

```
geo_all_channels.csv
        ↓
scripts/train_hierarchical.py
  → Filter selected geos
  → Compute revenue = conversions × revenue_per_conversion
  → Normalize spend per geo
  → Normalize revenue by mean
        ↓
PyMC Hierarchical Model
        ↓
results/hierarchical/hierarchical_Ngeos.pkl
```