# API Reference — MMM Multi-Market Bayesian

## `src.models.bayesian_mmm`

### `BayesianMMM`

Modèle MMM bayésien principal avec PyMC.
```python
from src.models.bayesian_mmm import BayesianMMM

model = BayesianMMM(market="FR", config=None, channels=None)
```

| Paramètre  | Type         | Description                              |
|------------|--------------|------------------------------------------|
| `market`   | str          | Code marché (ex: "FR")                   |
| `config`   | dict         | Hyperparamètres (draws, tune, chains)    |
| `channels` | list[str]    | Canaux à modéliser (défaut : les 5)      |

**Méthodes :**

| Méthode                  | Retour           | Description                        |
|--------------------------|------------------|------------------------------------|
| `.fit(df, draws, tune)`  | self             | Entraînement MCMC (NUTS)           |
| `.predict(df)`           | np.ndarray       | Prédictions en €                   |
| `.get_contributions(df)` | pd.DataFrame     | Décomposition par composante       |
| `.get_roi(df)`           | pd.DataFrame     | ROI par canal                      |
| `.diagnostics()`         | dict             | R-hat, ESS, convergence            |
| `.save(path)`            | None             | Sauvegarde idata NetCDF            |
| `.load(path, market)`    | BayesianMMM      | Chargement modèle sauvegardé       |

---

## `src.models.adstock`

### `GeometricAdstock`
```python
from src.models.adstock import GeometricAdstock

adstock = GeometricAdstock(decay=0.5)
result  = adstock.transform(spend_array)
```

| Méthode           | Description                        |
|-------------------|------------------------------------|
| `.transform(x)`   | Applique l'adstock géométrique     |
| `.half_life()`    | Retourne la demi-vie en semaines   |
| `.weights(n)`     | Poids de carryover sur n périodes  |

### `DelayedAdstock`
```python
adstock = DelayedAdstock(decay=0.5, peak=2)
```

---

## `src.models.saturation`

### `HillSaturation`
```python
from src.models.saturation import HillSaturation

hill   = HillSaturation(K=100_000, S=2.0)
result = hill.transform(spend_array)
```

| Méthode               | Description                          |
|-----------------------|--------------------------------------|
| `.transform(x)`       | Applique la transformation Hill      |
| `.response_curve()`   | Retourne (x, y) pour visualisation   |
| `.marginal_return(x)` | Dérivée — ROI marginal               |

---

## `src.evaluation.metrics`
```python
from src.evaluation.metrics import compute_all_metrics, print_metrics_report

metrics = compute_all_metrics(y_true, y_pred)
print_metrics_report(metrics, label="Train [FR]")
```

| Fonction                 | Retour      | Description             |
|--------------------------|-------------|-------------------------|
| `r_squared(y, ŷ)`        | float       | R² ∈ (-∞, 1]            |
| `mape(y, ŷ)`             | float (%)   | Mean Absolute % Error   |
| `smape(y, ŷ)`            | float (%)   | Symmetric MAPE          |
| `rmse(y, ŷ)`             | float (€)   | Root Mean Squared Error |
| `nrmse(y, ŷ)`            | float       | RMSE normalisé [0, 1]   |
| `compute_all_metrics()`  | dict        | Toutes les métriques    |
| `print_metrics_report()` | None        | Affichage formaté       |

---

## `src.evaluation.roi_calculator`
```python
from src.evaluation.roi_calculator import compute_roi, budget_recommendation

roi_df = compute_roi(contributions_df, spends_df)
rec_df = budget_recommendation(roi_df, total_budget=1_000_000)
```

---

## `src.data.data_loader`
```python
from src.data.data_loader import load_all_markets, load_market_data, split_train_test

df      = load_all_markets()                    # 2080 lignes
df_fr   = load_market_data("FR")               # 208 lignes
train, test = split_train_test(df_fr, test_ratio=0.2)
```

---

## `src.training.distributed_trainer`
```python
from src.training.distributed_trainer import train_all_markets

results = train_all_markets(
    markets=["FR", "DE", "UK"],
    n_jobs=4,
    save_model=True,
    track_mlflow=False,
)
```

---

## `pipelines.orchestration.run_pipeline`
```bash
# Pipeline complet
python pipelines/orchestration/run_pipeline.py

# Marchés spécifiques
python pipelines/orchestration/run_pipeline.py --markets FR,DE,UK

# Sans réentraînement
python pipelines/orchestration/run_pipeline.py --skip-train

# Forcer régénération données
python pipelines/orchestration/run_pipeline.py --force
```

---

## `app.streamlit_app`
```bash
streamlit run app/streamlit_app.py
```

**Pages disponibles :**

| Page               | Description                              |
|--------------------|------------------------------------------|
| Vue d'ensemble  | KPIs globaux, série temporelle revenue   |
| Performance     | R², MAPE, actuel vs prédit               |
| ROI par canal   | ROI avec intervalles de crédibilité      |
| Budget Optimizer| Simulation d'allocation budgétaire       |
| Cross-Market    | Heatmap et comparaison entre marchés     |
