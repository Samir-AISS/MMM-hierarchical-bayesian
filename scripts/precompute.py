"""
precompute.py
-------------
Génère results/precomputed.pkl avec tous les résultats pré-calculés.
À lancer une fois avant de déployer sur Streamlit Cloud.

Usage:
    python scripts/precompute.py
"""

import sys
import os
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import load_market_data, split_train_test
from src.models.bayesian_mmm import BayesianMMM
from src.evaluation.metrics import compute_all_metrics

ALL_MARKETS = ['FR', 'DE', 'UK', 'IT', 'ES', 'NL', 'BE', 'PL', 'SE', 'NO']
TRAIN_COLS  = [
    'date', 'revenue',
    'tv_spend', 'facebook_spend', 'search_spend', 'ooh_spend', 'print_spend',
    'seasonality', 'trend'
]

def precompute():
    results = {}
    t_total = time.time()

    print("=" * 50)
    print("  MMM Precompute — 10 marchés")
    print("=" * 50)

    for i, market in enumerate(ALL_MARKETS, 1):
        t0 = time.time()
        print(f"\n[{i}/10] {market}...")

        df = load_market_data(market)
        df_train, df_test = split_train_test(df, test_ratio=0.2)

        model = BayesianMMM(market=market)
        model.fit(df_train, draws=300, tune=300, chains=2, random_seed=42)

        y_pred_train = model.predict(df_train)
        y_pred_test  = model.predict(df_test)

        results[market] = {
            'metrics_train': compute_all_metrics(df_train['revenue'].values, y_pred_train),
            'metrics_test':  compute_all_metrics(df_test['revenue'].values,  y_pred_test),
            'roi':           model.get_roi(df_train).to_dict('records'),
            'contributions': model.get_contributions(df_train).to_dict('records'),
            'y_pred_test':   y_pred_test.tolist(),
            'df_test':       df_test[['date', 'revenue']].to_dict('records'),
            'df_train':      df_train[TRAIN_COLS].to_dict('records'),
        }

        m = results[market]['metrics_test']
        print(f"  R²={m['r2']:.3f} | MAPE={m['mape']:.1f}% | {time.time()-t0:.0f}s")

    # Sauvegarde
    out_path = Path(__file__).resolve().parents[1] / "results" / "precomputed.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print("\n" + "=" * 50)
    print(f"  Sauvegardé → {out_path}")
    print(f"  Taille     : {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Durée      : {time.time()-t_total:.0f}s")
    print("=" * 50)
    print("\nProchaine étape :")
    print("  git add results/precomputed.pkl")
    print("  git commit -m 'data: add precomputed results'")
    print("  git push")


if __name__ == "__main__":
    precompute()