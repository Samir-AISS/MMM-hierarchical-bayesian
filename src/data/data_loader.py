"""
data_loader.py
--------------
Chargement et préparation des données MMM multi-marchés.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "synthetic" / "mmm_multi_market.csv"


def load_all_markets(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Charge le dataset complet (tous les marchés, 2080 lignes).

    Paramètres
    ----------
    path : chemin vers le CSV (défaut : data/synthetic/mmm_multi_market.csv)

    Retourne
    --------
    DataFrame avec 2080 lignes × 14 colonnes
    """
    path = path or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Générez-le avec : python src/data/multi_market_generator.py"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["market", "week"]).reset_index(drop=True)
    return df


def load_market_data(
    market: str,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Charge les données d'un seul marché (208 lignes).

    Paramètres
    ----------
    market : code marché (ex: "FR", "DE", "UK")
    path   : chemin vers le CSV complet

    Retourne
    --------
    DataFrame filtré pour le marché demandé
    """
    df = load_all_markets(path)
    df_market = df[df["market"] == market.upper()].reset_index(drop=True)

    if len(df_market) == 0:
        available = df["market"].unique().tolist()
        raise ValueError(
            f"Marché '{market}' introuvable. "
            f"Marchés disponibles : {available}"
        )

    return df_market


def split_train_test(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel train/test (pas de shuffle — données chronologiques).

    Paramètres
    ----------
    df         : DataFrame d'un seul marché, trié par semaine
    test_ratio : proportion des données pour le test (défaut : 20%)

    Retourne
    --------
    (df_train, df_test)
    """
    df = df.sort_values("week").reset_index(drop=True)
    n_test  = max(1, int(len(df) * test_ratio))
    n_train = len(df) - n_test

    df_train = df.iloc[:n_train].copy()
    df_test  = df.iloc[n_train:].copy()

    return df_train, df_test


def get_available_markets(path: Optional[Path] = None) -> List[str]:
    """Retourne la liste des marchés disponibles dans le dataset."""
    df = load_all_markets(path)
    return sorted(df["market"].unique().tolist())


def get_spend_columns() -> List[str]:
    """Retourne les noms des colonnes de dépenses marketing."""
    return ["tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend"]


def get_dataset_info(path: Optional[Path] = None) -> dict:
    """
    Retourne un résumé du dataset.

    Retourne
    --------
    dict avec : n_rows, n_markets, n_weeks, date_range, columns
    """
    df = load_all_markets(path)
    return {
        "n_rows":     len(df),
        "n_markets":  df["market"].nunique(),
        "n_weeks":    df["week"].max(),
        "date_min":   str(df["date"].min().date()),
        "date_max":   str(df["date"].max().date()),
        "columns":    df.columns.tolist(),
        "markets":    sorted(df["market"].unique().tolist()),
    }