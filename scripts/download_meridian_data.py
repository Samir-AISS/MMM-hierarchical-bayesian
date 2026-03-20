"""
download_meridian_data.py
-------------------------
Télécharge les données Meridian (Google) pour le projet MMM.
Données multi-geo simulées — inspirées du framework Google Meridian.

Usage:
    python scripts/download_meridian_data.py
"""

import os
import pandas as pd
from pathlib import Path

# ── URLs des datasets Meridian ────────────────────────────────────────────────
DATASETS = {
    "hypothetical_geo_all_channels": {
        "url": "https://raw.githubusercontent.com/google/meridian/main/meridian/data/simulated_data/csv/hypothetical_geo_all_channels.csv",
        "description": "Multi-geo · 5 channels · revenue KPI (principal dataset)",
        "recommended": True,
    },
    "geo_all_channels": {
        "url": "https://raw.githubusercontent.com/google/meridian/main/meridian/data/simulated_data/csv/geo_all_channels.csv",
        "description": "Multi-geo · all channels · non-revenue KPI",
        "recommended": False,
    },
}

def download():
    out_dir = Path(__file__).parents[1] / "data" / "meridian"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Downloading Meridian datasets")
    print("=" * 60)

    for name, info in DATASETS.items():
        print(f"\n  {'⭐' if info['recommended'] else '  '} {name}")
        print(f"     {info['description']}")
        print(f"     Downloading...")

        try:
            df = pd.read_csv(info["url"])
            out_path = out_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)

            print(f"     {len(df):,} rows × {len(df.columns)} columns")
            print(f"     Saved → {out_path}")

            if info["recommended"]:
                print(f"\n     Columns  : {list(df.columns)}")
                if "geo" in df.columns:
                    print(f"     Geos     : {sorted(df['geo'].unique())}")
                if "time" in df.columns:
                    print(f"     Period   : {df['time'].min()} → {df['time'].max()}")

        except Exception as e:
            print(f"      Failed: {e}")

    print("\n" + "=" * 60)
    print("  Download complete")
    print("  Next step:")
    print("    python scripts/train_hierarchical.py")
    print("=" * 60)

if __name__ == "__main__":
    download()