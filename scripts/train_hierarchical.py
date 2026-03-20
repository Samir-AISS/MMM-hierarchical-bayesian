"""
train_hierarchical v3 — modèle simplifié pour convergence
Adstock pré-calculé en numpy, seuls beta/baseline sont hiérarchiques.
"""

import os
os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_compile"

import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

DATA_PATH   = Path(__file__).parents[1] / "data" / "meridian" / "geo_all_channels.csv"
RESULTS_DIR = Path(__file__).parents[1] / "results" / "hierarchical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS   = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4"]
SPEND_COLS = [f"{c}_spend" for c in CHANNELS]
CONTROLS   = ["competitor_sales_control", "sentiment_score_control", "Promo"]

# ── Geo Selection ─────────────────────────────────────────────────────────────
def select_geos(df):
    all_geos    = sorted(df["geo"].unique())
    pop_ranking = (df.groupby("geo")["population"].mean()
                     .sort_values(ascending=False).index.tolist())

    print("\n" + "=" * 60)
    print("  MMM Hierarchical Model — Geo Selection")
    print("=" * 60)
    print(f"\n  Available geos ({len(all_geos)}) :")
    print("  " + "  ".join(all_geos[:20]))
    if len(all_geos) > 20:
        print("  " + "  ".join(all_geos[20:]))
    print(f"\n  Options :")
    print(f"  [1] All geos ({len(all_geos)})")
    print(f"  [2] Top 10 by population")
    print(f"  [3] Top 5 by population")
    print(f"  [4] Custom selection")

    choice = input("\n  Choice [1/2/3/4] : ").strip()
    if choice == "1":   selected = all_geos
    elif choice == "2": selected = pop_ranking[:10]
    elif choice == "3": selected = pop_ranking[:5]
    elif choice == "4":
        raw = input("  Geos (comma-separated) : ").strip()
        selected = [g.strip() for g in raw.split(",") if g.strip() in all_geos]
        if not selected: selected = pop_ranking[:5]
    else: selected = pop_ranking[:5]

    print(f"\n  Selected {len(selected)} geos : {selected}")
    return selected

def select_mcmc():
    print("\n  MCMC Configuration :")
    print("  [1] Fast     — 500 draws, 1000 tune, 2 chains")
    print("  [2] Standard — 1000 draws, 1000 tune, 2 chains")
    print("  [3] Full     — 1000 draws, 2000 tune, 4 chains")
    choice = input("  Choice [1/2/3] : ").strip()
    configs = {
        "1": {"draws": 500,  "tune": 1000, "chains": 2},
        "2": {"draws": 1000, "tune": 1000, "chains": 2},
        "3": {"draws": 1000, "tune": 2000, "chains": 4},
    }
    cfg = configs.get(choice, configs["1"])
    print(f"  → {cfg['draws']} draws · {cfg['tune']} tune · {cfg['chains']} chains")
    return cfg

# ── Adstock numpy (pré-calculé hors modèle) ───────────────────────────────────
def compute_adstock_numpy(spend_norm, decay_values):
    """
    Calcule l'adstock en numpy avec des valeurs de decay fixes.
    spend_norm   : (n_geos, n_times, n_ch)
    decay_values : (n_ch,) — valeurs moyennes de decay
    returns      : (n_geos, n_times, n_ch)
    """
    n_geos, n_times, n_ch = spend_norm.shape
    result = np.zeros_like(spend_norm)
    result[:, 0, :] = spend_norm[:, 0, :]
    for t in range(1, n_times):
        result[:, t, :] = spend_norm[:, t, :] + decay_values[None, :] * result[:, t-1, :]
    return result

# ── Data Prep ─────────────────────────────────────────────────────────────────
def prepare(df, geos):
    df = df[df["geo"].isin(geos)].copy()
    df["time"]    = pd.to_datetime(df["time"])
    df["revenue"] = df["conversions"] * df["revenue_per_conversion"]
    df            = df.sort_values(["geo", "time"]).reset_index(drop=True)

    n_geos  = len(geos)
    n_times = df["time"].nunique()
    n_ch    = len(CHANNELS)
    n_ctrl  = len(CONTROLS)

    spend   = np.zeros((n_geos, n_times, n_ch))
    revenue = np.zeros((n_geos, n_times))
    ctrl    = np.zeros((n_geos, n_times, n_ctrl))

    for i, geo in enumerate(geos):
        g = df[df["geo"] == geo].sort_values("time")
        spend[i]   = g[SPEND_COLS].values
        revenue[i] = g["revenue"].values
        ctrl[i]    = g[CONTROLS].values

    spend_max  = spend.max(axis=(0,1), keepdims=True) + 1e-8
    spend_norm = (spend / spend_max).astype("float64")
    rev_scale  = revenue.mean()
    rev_norm   = (revenue / rev_scale).astype("float64")
    ctrl_norm  = ctrl.astype("float64")

    # Pré-calculer adstock avec decay=0.4 (prior moyen)
    decay_prior = np.full(n_ch, 0.4)
    adstock_norm = compute_adstock_numpy(spend_norm, decay_prior)

    print(f"  {n_geos} geos × {n_times} weeks · adstock pre-computed")
    return adstock_norm, rev_norm, ctrl_norm, rev_scale, spend_max, spend_norm, n_geos, n_times

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(adstock_norm, rev_norm, ctrl_norm, n_geos, n_times):
    """
    Modèle hiérarchique simplifié :
    - Adstock pré-calculé (numpy) — plus de boucle dans le graphe
    - Hill saturation + beta hiérarchique
    - Convergence rapide et fiable
    """
    n_ch   = len(CHANNELS)
    n_ctrl = len(CONTROLS)

    # Flatten pour le modèle : (n_geos × n_times, n_ch)
    adstock_flat = adstock_norm.reshape((-1, n_ch))
    ctrl_flat    = ctrl_norm.reshape((-1, n_ctrl))
    rev_flat     = rev_norm.reshape(-1)

    # Index geo pour les effets hiérarchiques
    geo_idx = np.repeat(np.arange(n_geos), n_times)

    with pm.Model() as model:

        # ── Global priors ────────────────────────────────────────────────
        mu_beta  = pm.HalfNormal("mu_beta",  sigma=1.0, shape=n_ch)
        sig_beta = pm.HalfNormal("sig_beta", sigma=0.5, shape=n_ch)

        # ── Geo-level beta (non-centered) ────────────────────────────────
        beta_raw = pm.Normal("beta_raw", 0, 1, shape=(n_geos, n_ch))
        beta     = pm.Deterministic("beta",
                       mu_beta[None,:] + sig_beta[None,:] * beta_raw)

        # ── Hill saturation (global) ──────────────────────────────────────
        ec50  = pm.HalfNormal("ec50",  sigma=0.5, shape=n_ch)
        slope = pm.HalfNormal("slope", sigma=2.0, shape=n_ch)

        # ── Geo baseline (hierarchical) ───────────────────────────────────
        mu_base  = pm.Normal("mu_base",  mu=1.0, sigma=0.5)
        sig_base = pm.HalfNormal("sig_base", sigma=0.3)
        baseline = pm.Normal("baseline", mu=mu_base, sigma=sig_base, shape=n_geos)

        # ── Controls ──────────────────────────────────────────────────────
        gamma = pm.Normal("gamma", mu=0, sigma=0.5, shape=n_ctrl)
        sigma = pm.HalfNormal("sigma", sigma=0.2)

        # ── Hill saturation sur adstock pré-calculé ───────────────────────
        # adstock_flat : (n_obs, n_ch)
        # beta[geo_idx]: (n_obs, n_ch)
        sat = (adstock_flat ** slope[None,:] /
               (ec50[None,:]**slope[None,:] + adstock_flat**slope[None,:]))

        media = pm.math.sum(beta[geo_idx] * sat, axis=1)  # (n_obs,)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl_effect = pm.math.dot(ctrl_flat, gamma)  # (n_obs,)

        # ── Likelihood ────────────────────────────────────────────────────
        mu  = pm.math.maximum(baseline[geo_idx] + media + ctrl_effect, 0)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=rev_flat)

    print(f"  Model built — {len(model.free_RVs)} free parameters")
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows · {df['geo'].nunique()} geos · {df['time'].nunique()} weeks")

    geos = select_geos(df)
    cfg  = select_mcmc()

    adstock_norm, rev_norm, ctrl_norm, rev_scale, spend_max, spend_norm, n_geos, n_times = prepare(df, geos)
    model = build_model(adstock_norm, rev_norm, ctrl_norm, n_geos, n_times)

    print(f"\n  Sampling...")
    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=cfg["draws"],
            tune=cfg["tune"],
            chains=cfg["chains"],
            target_accept=0.9,
            random_seed=42,
            progressbar=True,
        )

    duration = time.time() - t0
    print(f"\n  Done in {duration:.0f}s")

    summary = az.summary(idata, var_names=["mu_beta", "baseline", "sigma"])
    print(summary[["mean", "sd", "r_hat"]].to_string())

    max_rhat = summary["r_hat"].max()
    print(f"\n  Max R-hat: {max_rhat:.3f} {'✅ Converged' if max_rhat < 1.05 else '⚠️ Not converged'}")

    out = {
        "idata":         idata,
        "selected_geos": geos,
        "rev_scale":     rev_scale,
        "spend_max":     spend_max,
        "spend_norm":    spend_norm,
        "adstock_norm":  adstock_norm,
        "channels":      CHANNELS,
        "mcmc_config":   cfg,
        "n_geos":        n_geos,
        "n_times":       n_times,
    }
    path = RESULTS_DIR / f"hierarchical_{n_geos}geos.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)

    print(f"\n{'='*60}")
    print(f"  ✅ Saved → {path}")
    print(f"  Duration : {duration:.0f}s · Geos : {n_geos}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()