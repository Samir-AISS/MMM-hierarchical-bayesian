"""
prefect_flow.py
---------------
Orchestration du pipeline MMM avec Prefect Cloud.

Usage:
    prefect cloud login
    python pipelines/prefect_flow.py
"""

import sys
import time
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.data.multi_market_generator import generate_full_dataset
from src.data.data_validator          import validate
from src.data.data_loader             import load_market_data, split_train_test
from src.models.bayesian_mmm          import BayesianMMM
from src.evaluation.metrics           import compute_all_metrics

ALL_MARKETS = ["FR", "DE", "UK", "IT", "ES", "NL", "BE", "PL", "SE", "NO"]
DATA_PATH   = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "mmm_multi_market.csv"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ── TASKS ─────────────────────────────────────────────────────────────────────

@task(name="Generate Data", retries=1)
def task_generate_data(force: bool = False):
    logger = get_run_logger()
    if DATA_PATH.exists() and not force:
        logger.info(f"Données existantes — {DATA_PATH.name}")
        import pandas as pd
        return pd.read_csv(DATA_PATH, parse_dates=["date"])
    logger.info("Génération des données synthétiques...")
    df = generate_full_dataset(output_path=str(DATA_PATH))
    logger.info(f"Données générées — {len(df)} lignes")
    return df


@task(name="Validate Data", retries=0)
def task_validate_data(df):
    logger = get_run_logger()
    report = validate(df)
    report.print_summary()
    if report.n_errors > 0:
        raise ValueError(f"Validation échouée — {report.n_errors} erreur(s)")
    logger.info(f"Validation OK — {report.n_tests} tests")
    return True


@task(name="Train Market", retries=1, retry_delay_seconds=10)
def task_train_market(market: str, draws: int = 300, tune: int = 300):
    logger = get_run_logger()
    logger.info(f"[{market}] Entraînement...")
    t0 = time.time()

    df = load_market_data(market)
    df_train, df_test = split_train_test(df, test_ratio=0.2)

    model = BayesianMMM(market=market)
    model.fit(df_train, draws=draws, tune=tune, chains=2, random_seed=42)

    y_pred  = model.predict(df_test)
    metrics = compute_all_metrics(df_test["revenue"].values, y_pred)
    roi_df  = model.get_roi(df_train)

    duration = time.time() - t0
    logger.info(
        f"[{market}] R²={metrics['r2']:.3f} | "
        f"MAPE={metrics['mape']:.1f}% | {duration:.0f}s"
    )

    return {
        "market":   market,
        "metrics":  metrics,
        "roi":      roi_df.to_dict("records"),
        "duration": duration,
    }


@task(name="Save Precomputed Results")
def task_save_results(all_results: list):
    logger = get_run_logger()

    TRAIN_COLS = [
        "date", "revenue",
        "tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend",
        "seasonality", "trend",
    ]

    precomputed = {}
    for r in all_results:
        market = r["market"]
        df     = load_market_data(market)
        df_train, df_test = split_train_test(df, test_ratio=0.2)
        model  = BayesianMMM(market=market)
        model.fit(df_train, draws=300, tune=300, chains=2, random_seed=42)

        y_pred_train = model.predict(df_train)
        y_pred_test  = model.predict(df_test)

        precomputed[market] = {
            "metrics_train": compute_all_metrics(df_train["revenue"].values, y_pred_train),
            "metrics_test":  compute_all_metrics(df_test["revenue"].values,  y_pred_test),
            "roi":           model.get_roi(df_train).to_dict("records"),
            "contributions": model.get_contributions(df_train).to_dict("records"),
            "y_pred_test":   y_pred_test.tolist(),
            "df_test":       df_test[["date", "revenue"]].to_dict("records"),
            "df_train":      df_train[TRAIN_COLS].to_dict("records"),
        }

    out = RESULTS_DIR / "precomputed.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(precomputed, f)

    logger.info(f"precomputed.pkl sauvegardé — {out.stat().st_size/1e6:.1f} MB")
    return str(out)


@task(name="Generate Report")
def task_generate_report(all_results: list):
    logger = get_run_logger()

    rows = []
    for r in all_results:
        m  = r["metrics"]
        ok = "✅" if m["r2"] >= 0.70 else "⚠️"
        rows.append(f"| {ok} | {r['market']} | {m['r2']:.3f} | {m['mape']:.1f}% | {r['duration']:.0f}s |")

    avg_r2   = sum(r["metrics"]["r2"]   for r in all_results) / len(all_results)
    avg_mape = sum(r["metrics"]["mape"] for r in all_results) / len(all_results)

    table = "\n".join(rows)
    markdown = f"""# MMM Pipeline Report

## Results

| Status | Market | R² | MAPE | Duration |
|--------|--------|----|------|----------|
{table}

## Summary
- **Markets trained** : {len(all_results)}
- **Avg R²** : {avg_r2:.3f}
- **Avg MAPE** : {avg_mape:.1f}%
"""

    create_markdown_artifact(
        key="mmm-pipeline-report",
        markdown=markdown,
        description="MMM Pipeline Results",
    )

    # Sauvegarde CSV
    import pandas as pd
    report_df = pd.DataFrame([{
        "market": r["market"],
        "r2":     r["metrics"]["r2"],
        "mape":   r["metrics"]["mape"],
        "duration": r["duration"],
    } for r in all_results])

    report_path = RESULTS_DIR / "reports" / "prefect_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    logger.info(f"Rapport sauvegardé → {report_path}")
    return markdown


# ── FLOW ──────────────────────────────────────────────────────────────────────

@flow(
    name="MMM Multi-Market Pipeline",
    description="Bayesian MMM — 10 European markets",
    log_prints=True,
)
def mmm_pipeline(
    markets: list = ALL_MARKETS,
    draws:   int  = 300,
    tune:    int  = 300,
    force:   bool = False,
):
    logger = get_run_logger()
    logger.info(f"Pipeline MMM — {len(markets)} marchés")

    # 1. Données
    df = task_generate_data(force=force)

    # 2. Validation
    task_validate_data(df)

    # 3. Entraînement séquentiel (Prefect gère les retries)
    all_results = []
    for market in markets:
        result = task_train_market(market, draws=draws, tune=tune)
        all_results.append(result)

    # 4. Rapport
    task_generate_report(all_results)

    # 5. Sauvegarde precomputed.pkl
    task_save_results(all_results)

    logger.info("Pipeline terminé")
    return all_results


if __name__ == "__main__":
    mmm_pipeline()