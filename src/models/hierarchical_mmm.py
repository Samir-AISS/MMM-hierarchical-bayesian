"""
hierarchical_mmm.py
-------------------
Modèle MMM Hiérarchique Bayésien — inspiré de Google Meridian.

Architecture :
    Revenue(g,t) = baseline(g)
                 + Σ_c β(g,c) × Hill(Adstock(spend(g,t,c)))
                 + Σ_k γ_k × control(g,t,k)

Priors hiérarchiques :
    β(g,c)      ~ Normal(μ_β(c), σ_β(c))  — partagés entre geos
    baseline(g) ~ Normal(μ_base, σ_base)   — baseline par geo

Note : adstock pré-calculé en numpy (decay fixé à 0.4 par défaut)
       pour éviter les graphes PyTensor trop profonds.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

CHANNELS   = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4"]
SPEND_COLS = [f"{c}_spend" for c in CHANNELS]
CONTROLS   = ["competitor_sales_control", "sentiment_score_control", "Promo"]


class HierarchicalMMM:
    """
    Hierarchical Bayesian MMM — Meridian-inspired.

    Parameters
    ----------
    selected_geos : list
        List of geo names to include in the model.
    adstock_decay : float or list
        Fixed adstock decay values (default 0.4 per channel).
        Pre-computed in numpy to ensure model convergence.

    Example
    -------
    >>> model = HierarchicalMMM(selected_geos=["Geo0", "Geo1", "Geo2"])
    >>> model.fit(df, draws=500, tune=1000, chains=2)
    >>> roi = model.get_roi()
    >>> print(roi)
    """

    def __init__(self, selected_geos: list, adstock_decay=0.4):
        self.selected_geos = selected_geos
        self.n_geos        = len(selected_geos)
        self.geo_idx_map   = {g: i for i, g in enumerate(selected_geos)}
        self.adstock_decay = adstock_decay
        self.model         = None
        self.idata         = None
        self.rev_scale     = None
        self.spend_max     = None
        self._adstock_norm = None
        self._spend_norm   = None
        self._rev_norm     = None
        self._ctrl_norm    = None
        self._n_times      = None

    # ── Data Preparation ──────────────────────────────────────────────────────

    def _compute_adstock(self, spend_norm: np.ndarray) -> np.ndarray:
        """Adstock géométrique numpy — (n_geos, n_times, n_ch)."""
        n_geos, n_times, n_ch = spend_norm.shape
        decay = np.full(n_ch, self.adstock_decay) \
            if isinstance(self.adstock_decay, float) \
            else np.array(self.adstock_decay)

        result = np.zeros_like(spend_norm)
        result[:, 0, :] = spend_norm[:, 0, :]
        for t in range(1, n_times):
            result[:, t, :] = (spend_norm[:, t, :] +
                               decay[None, :] * result[:, t-1, :])
        return result

    def prepare_data(self, df: pd.DataFrame):
        """Prépare les arrays numpy pour le modèle."""
        df = df[df["geo"].isin(self.selected_geos)].copy()
        df["time"]    = pd.to_datetime(df["time"])
        df["revenue"] = df["conversions"] * df["revenue_per_conversion"]
        df            = df.sort_values(["geo", "time"]).reset_index(drop=True)

        self._n_times = df["time"].nunique()
        n_ch    = len(CHANNELS)
        n_ctrl  = len(CONTROLS)

        spend   = np.zeros((self.n_geos, self._n_times, n_ch))
        revenue = np.zeros((self.n_geos, self._n_times))
        ctrl    = np.zeros((self.n_geos, self._n_times, n_ctrl))

        for i, geo in enumerate(self.selected_geos):
            g = df[df["geo"] == geo].sort_values("time")
            spend[i]   = g[SPEND_COLS].values
            revenue[i] = g["revenue"].values
            ctrl[i]    = g[CONTROLS].values

        self.spend_max     = spend.max(axis=(0,1), keepdims=True) + 1e-8
        self._spend_norm   = (spend / self.spend_max).astype("float64")
        self.rev_scale     = revenue.mean()
        self._rev_norm     = (revenue / self.rev_scale).astype("float64")
        self._ctrl_norm    = ctrl.astype("float64")
        self._adstock_norm = self._compute_adstock(self._spend_norm)

        print(f"  Data ready — {self.n_geos} geos × {self._n_times} weeks")
        return self

    # ── Model Building ────────────────────────────────────────────────────────

    def build(self):
        """Construit le modèle PyMC hiérarchique."""
        n_geos  = self.n_geos
        n_times = self._n_times
        n_ch    = len(CHANNELS)
        n_ctrl  = len(CONTROLS)

        # Flatten pour le modèle
        adstock_flat = self._adstock_norm.reshape((-1, n_ch))
        ctrl_flat    = self._ctrl_norm.reshape((-1, n_ctrl))
        rev_flat     = self._rev_norm.reshape(-1)
        geo_idx      = np.repeat(np.arange(n_geos), n_times)

        with pm.Model() as model:

            # ── Global channel priors ─────────────────────────────────────
            mu_beta  = pm.HalfNormal("mu_beta",  sigma=1.0, shape=n_ch)
            sig_beta = pm.HalfNormal("sig_beta", sigma=0.5, shape=n_ch)

            # ── Geo-level effects (non-centered) ──────────────────────────
            beta_raw = pm.Normal("beta_raw", 0, 1, shape=(n_geos, n_ch))
            beta     = pm.Deterministic("beta",
                           mu_beta[None,:] + sig_beta[None,:] * beta_raw)

            # ── Hill saturation (global) ──────────────────────────────────
            ec50  = pm.HalfNormal("ec50",  sigma=0.5, shape=n_ch)
            slope = pm.HalfNormal("slope", sigma=2.0, shape=n_ch)

            # ── Geo baseline (hierarchical) ───────────────────────────────
            mu_base  = pm.Normal("mu_base",  mu=1.0, sigma=0.5)
            sig_base = pm.HalfNormal("sig_base", sigma=0.3)
            baseline = pm.Normal("baseline", mu=mu_base, sigma=sig_base,
                                 shape=n_geos)

            # ── Controls ──────────────────────────────────────────────────
            gamma = pm.Normal("gamma", mu=0, sigma=0.5, shape=n_ctrl)
            sigma = pm.HalfNormal("sigma", sigma=0.2)

            # ── Hill saturation ───────────────────────────────────────────
            sat = (adstock_flat ** slope[None,:] /
                   (ec50[None,:]**slope[None,:] +
                    adstock_flat**slope[None,:]))

            media       = pm.math.sum(beta[geo_idx] * sat, axis=1)
            ctrl_effect = pm.math.dot(ctrl_flat, gamma)

            mu  = pm.math.maximum(baseline[geo_idx] + media + ctrl_effect, 0)
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=rev_flat)

        self.model = model
        print(f"  Model built — {len(model.free_RVs)} free parameters")
        return self

    # ── Sampling ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, draws=500, tune=1000,
            chains=2, target_accept=0.9, random_seed=42):
        """Prépare, construit et échantillonne."""
        self.prepare_data(df)
        self.build()
        print(f"\n  Sampling — {chains} chains × {draws} draws + {tune} tune")
        with self.model:
            self.idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
            )
        print("  Sampling complete")
        return self

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> pd.DataFrame:
        """Résumé de convergence ArviZ."""
        if self.idata is None:
            raise ValueError("Call fit() first.")
        return az.summary(self.idata,
                          var_names=["mu_beta", "baseline", "sigma"])

    def r_hat_ok(self, threshold=1.05) -> bool:
        """Vérifie la convergence (R-hat < threshold)."""
        summary  = self.diagnostics()
        max_rhat = summary["r_hat"].max()
        status   = " Converged" if max_rhat < threshold else "⚠️ Not converged"
        print(f"  Max R-hat: {max_rhat:.3f} — {status}")
        return max_rhat < threshold

    # ── ROI ───────────────────────────────────────────────────────────────────

    def get_roi(self) -> pd.DataFrame:
        """ROI par geo et canal (moyenne postérieure)."""
        if self.idata is None:
            raise ValueError("Call fit() first.")

        beta_mean  = self.idata.posterior["beta"].values.mean(axis=(0,1))
        ec50_mean  = self.idata.posterior["ec50"].values.mean(axis=(0,1))
        slope_mean = self.idata.posterior["slope"].values.mean(axis=(0,1))

        results = []
        for g_i, geo in enumerate(self.selected_geos):
            total_spend = (self._adstock_norm[g_i] * self.spend_max[0,0]).sum(axis=0)
            for c_i, ch in enumerate(CHANNELS):
                if total_spend[c_i] < 1e-6:
                    continue
                ads = self._adstock_norm[g_i, :, c_i]
                sat = (ads**slope_mean[c_i] /
                       (ec50_mean[c_i]**slope_mean[c_i] + ads**slope_mean[c_i]))
                contrib = (beta_mean[g_i, c_i] * sat * self.rev_scale).sum()
                results.append({
                    "geo":     geo,
                    "channel": ch,
                    "roi":     contrib / total_spend[c_i],
                })
        return pd.DataFrame(results)

    def get_global_roi(self) -> pd.DataFrame:
        """ROI global par canal (tous geos confondus)."""
        roi_df = self.get_roi()
        return (roi_df.groupby("channel")["roi"]
                      .mean()
                      .reset_index()
                      .sort_values("roi", ascending=False))

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "idata":         self.idata,
                "selected_geos": self.selected_geos,
                "geo_idx_map":   self.geo_idx_map,
                "rev_scale":     self.rev_scale,
                "spend_max":     self.spend_max,
                "adstock_norm":  self._adstock_norm,
                "spend_norm":    self._spend_norm,
                "n_times":       self._n_times,
                "adstock_decay": self.adstock_decay,
            }, f)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(selected_geos=d["selected_geos"],
                  adstock_decay=d.get("adstock_decay", 0.4))
        obj.idata          = d["idata"]
        obj.geo_idx_map    = d["geo_idx_map"]
        obj.rev_scale      = d["rev_scale"]
        obj.spend_max      = d["spend_max"]
        obj._adstock_norm  = d["adstock_norm"]
        obj._spend_norm    = d["spend_norm"]
        obj._n_times       = d["n_times"]
        print(f"  Loaded from {path}")
        return obj