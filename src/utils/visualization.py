"""
visualization.py
----------------
Visualisations pour le projet MMM Multi-Market.

Graphiques disponibles :
  - plot_channel_contributions()   : waterfall des contributions par canal
  - plot_saturation_curves()       : courbes Hill par canal
  - plot_actual_vs_predicted()     : série temporelle réel vs prédit
  - plot_roi_comparison()          : comparaison ROI entre marchés
  - plot_posterior_distributions() : distributions postérieures PyMC
  - plot_budget_optimizer()        : courbes ROI marginal
  - plot_market_heatmap()          : heatmap cross-market
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional, List


# ── Palette couleurs cohérente pour tous les graphiques ──────────────────────
CHANNEL_COLORS = {
    "tv":       "#2196F3",   # bleu
    "facebook": "#4CAF50",   # vert
    "search":   "#FF9800",   # orange
    "ooh":      "#9C27B0",   # violet
    "print":    "#F44336",   # rouge
    "base":     "#607D8B",   # gris bleu
    "seasonality": "#00BCD4",
    "trend":    "#795548",
    "events":   "#FF5722",
}

FIGURE_SIZE  = (12, 6)
TITLE_STYLE  = {"fontsize": 14, "fontweight": "bold"}
LABEL_STYLE  = {"fontsize": 11}
GRID_STYLE   = {"alpha": 0.3, "linestyle": "--"}


# ── 1. Waterfall contributions ────────────────────────────────────────────────

def plot_channel_contributions(
    contributions_df: pd.DataFrame,
    market: str = "ALL",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart empilé des contributions moyennes hebdomadaires par canal.

    Paramètres
    ----------
    contributions_df : DataFrame avec colonnes [base, tv, facebook, search, ooh, print, ...]
    market           : label affiché dans le titre
    output_path      : si fourni, sauvegarde la figure
    """
    channels = [c for c in ["base", "tv", "facebook", "search", "ooh", "print",
                             "seasonality", "trend", "events"]
                if c in contributions_df.columns]

    means = contributions_df[channels].mean()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors  = [CHANNEL_COLORS.get(c, "#BDBDBD") for c in channels]
    bars    = ax.bar(channels, means.values, color=colors, edgecolor="white", linewidth=0.8)

    # Valeurs au-dessus des barres
    for bar, val in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + means.max() * 0.01,
            f"€{val:,.0f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_title(f"Contributions moyennes par canal — Marché {market}", **TITLE_STYLE)
    ax.set_ylabel("Contribution hebdomadaire moyenne (€)", **LABEL_STYLE)
    ax.set_xlabel("Canal", **LABEL_STYLE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(axis="y", **GRID_STYLE)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── 2. Courbes de saturation ──────────────────────────────────────────────────

def plot_saturation_curves(
    df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Courbes de saturation Hill pour chaque canal.
    Utilise les dépenses réelles du dataset pour calibrer l'axe X.

    Paramètres
    ----------
    df       : DataFrame avec les colonnes *_spend
    channels : liste de canaux à afficher (défaut : tous)
    """
    from src.models.saturation import HillSaturation

    if channels is None:
        channels = ["tv", "facebook", "search", "ooh", "print"]

    n     = len(channels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, channel in enumerate(channels):
        spend_col = f"{channel}_spend"
        if spend_col not in df.columns:
            continue

        spend_data = df[spend_col].values
        x_max      = spend_data.max() * 1.2
        x          = np.linspace(0, x_max, 300)

        # Paramètre K = médiane des dépenses observées
        K = np.median(spend_data[spend_data > 0])
        hill = HillSaturation(K=K, S=2.0)
        y    = hill.transform(x)

        ax = axes[i]
        ax.plot(x / 1000, y, color=CHANNEL_COLORS.get(channel, "#333"),
                linewidth=2.5, label="Courbe Hill")

        # Zone des dépenses observées
        ax.axvspan(spend_data.min() / 1000, spend_data.max() / 1000,
                   alpha=0.1, color=CHANNEL_COLORS.get(channel, "#333"),
                   label="Plage observée")
        ax.axvline(K / 1000, color="gray", linestyle="--", linewidth=1,
                   label=f"K = €{K/1000:.0f}k")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

        ax.set_title(f"{channel.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Spend hebdomadaire (k€)", fontsize=9)
        ax.set_ylabel("Saturation [0-1]", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(**GRID_STYLE)
        ax.set_facecolor("#FAFAFA")

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Courbes de saturation par canal (Hill)", **TITLE_STYLE)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── 3. Actuel vs Prédit ───────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    market: str = "ALL",
    y_pred_lower: Optional[np.ndarray] = None,
    y_pred_upper: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Série temporelle : revenus réels vs prédits.
    Affiche optionnellement l'intervalle de crédibilité bayésien.

    Paramètres
    ----------
    df            : DataFrame avec colonnes 'week' et 'revenue'
    y_pred        : prédictions centrales (moyenne postérieure)
    y_pred_lower  : borne inférieure HDI 94%
    y_pred_upper  : borne supérieure HDI 94%
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    weeks = df["week"].values

    # Intervalle de crédibilité
    if y_pred_lower is not None and y_pred_upper is not None:
        ax.fill_between(weeks, y_pred_lower, y_pred_upper,
                        alpha=0.2, color="#2196F3", label="HDI 94%")

    ax.plot(weeks, df["revenue"].values, color="#333333",
            linewidth=1.5, label="Réel", zorder=3)
    ax.plot(weeks, y_pred, color="#2196F3",
            linewidth=2, linestyle="--", label="Prédit", zorder=4)

    # Métriques rapides
    residuals = df["revenue"].values - y_pred
    mape_val  = np.mean(np.abs(residuals / np.where(df["revenue"].values != 0,
                                                    df["revenue"].values, 1))) * 100
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((df["revenue"].values - df["revenue"].mean()) ** 2)
    r2_val    = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.set_title(
        f"Réel vs Prédit — Marché {market}  |  R²={r2_val:.3f}  MAPE={mape_val:.1f}%",
        **TITLE_STYLE
    )
    ax.set_xlabel("Semaine", **LABEL_STYLE)
    ax.set_ylabel("Revenue (€)", **LABEL_STYLE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1e6:.1f}M"))
    ax.legend(fontsize=10)
    ax.grid(**GRID_STYLE)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── 4. ROI par canal (comparaison marchés) ────────────────────────────────────

def plot_roi_comparison(
    roi_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart horizontal du ROI par canal, avec barres d'erreur HDI si disponibles.

    Paramètres
    ----------
    roi_df : DataFrame avec colonnes [channel, roi, roi_lower*, roi_upper*]
             (* optionnel — intervalles de crédibilité)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    channels = roi_df["channel"].tolist()
    rois     = roi_df["roi"].tolist()
    colors   = [CHANNEL_COLORS.get(c, "#BDBDBD") for c in channels]

    # Barres d'erreur si disponibles
    xerr = None
    if "roi_lower" in roi_df.columns and "roi_upper" in roi_df.columns:
        xerr = [
            roi_df["roi"].values - roi_df["roi_lower"].values,
            roi_df["roi_upper"].values - roi_df["roi"].values,
        ]

    bars = ax.barh(channels, rois, color=colors, xerr=xerr,
                   capsize=4, edgecolor="white", linewidth=0.8)

    # Ligne ROI = 1 (seuil de rentabilité)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="ROI = 1 (seuil rentabilité)")

    # Valeurs à droite des barres
    for bar, val in zip(bars, rois):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}x", va="center", fontsize=10, fontweight="bold")

    ax.set_title("ROI par canal marketing", **TITLE_STYLE)
    ax.set_xlabel("ROI (€ généré / € dépensé)", **LABEL_STYLE)
    ax.legend(fontsize=9)
    ax.grid(axis="x", **GRID_STYLE)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── 5. Distributions postérieures ────────────────────────────────────────────

def plot_posterior_distributions(
    idata,
    params: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Histogrammes des distributions postérieures des paramètres clés.
    Nécessite arviz.

    Paramètres
    ----------
    idata  : az.InferenceData retourné par PyMC
    params : liste de noms de variables à afficher
    """
    import arviz as az

    if params is None:
        params = ["beta_tv", "beta_facebook", "beta_search",
                  "beta_ooh", "beta_print"]

    # Filtrer les params existants dans idata
    available = [p for p in params if p in idata.posterior]

    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Aucun paramètre disponible dans idata",
                ha="center", transform=ax.transAxes)
        return fig

    n     = len(available)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, param in enumerate(available):
        samples = idata.posterior[param].values.flatten()
        ax = axes[i]

        ax.hist(samples, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(samples), color="red", linewidth=2, label=f"Moyenne: {np.mean(samples):.3f}")
        ax.axvline(np.percentile(samples, 3),  color="orange", linestyle="--", linewidth=1)
        ax.axvline(np.percentile(samples, 97), color="orange", linestyle="--", linewidth=1,
                   label="HDI 94%")

        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_xlabel("Valeur", fontsize=9)
        ax.set_ylabel("Fréquence", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(**GRID_STYLE)
        ax.set_facecolor("#FAFAFA")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributions postérieures des paramètres", **TITLE_STYLE)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── 6. Optimiseur budgétaire ──────────────────────────────────────────────────

def plot_budget_optimizer(
    spend_range: np.ndarray,
    marginal_roi_by_channel: dict,
    current_allocation: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Courbes de ROI marginal par canal — aide à l'optimisation budgétaire.

    Paramètres
    ----------
    spend_range             : array de niveaux de spend (axe X commun)
    marginal_roi_by_channel : {channel: array de ROI marginal}
    current_allocation      : {channel: spend actuel} — affiche le point actuel
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for channel, marginal_roi in marginal_roi_by_channel.items():
        color = CHANNEL_COLORS.get(channel, "#BDBDBD")
        ax.plot(spend_range / 1000, marginal_roi, color=color,
                linewidth=2, label=channel.upper())

        if current_allocation and channel in current_allocation:
            curr_spend = current_allocation[channel]
            # Interpolation du ROI marginal au spend actuel
            idx = np.searchsorted(spend_range, curr_spend)
            if idx < len(marginal_roi):
                ax.scatter(curr_spend / 1000, marginal_roi[idx],
                           color=color, s=80, zorder=5)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="ROI marginal = 1 (seuil)")
    ax.set_title("Courbes de ROI marginal par canal", **TITLE_STYLE)
    ax.set_xlabel("Spend hebdomadaire (k€)", **LABEL_STYLE)
    ax.set_ylabel("ROI marginal (€/€)", **LABEL_STYLE)
    ax.legend(fontsize=10)
    ax.grid(**GRID_STYLE)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ── 7. Heatmap cross-market ───────────────────────────────────────────────────

def plot_market_heatmap(
    roi_all_markets: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmap ROI par canal × marché.

    Paramètres
    ----------
    roi_all_markets : DataFrame avec colonnes [market, channel, roi]
    """
    pivot = roi_all_markets.pivot(index="channel", columns="market", values="roi")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Valeurs dans les cellules
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}x", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val < pivot.values.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="ROI (€/€)")
    ax.set_title("Heatmap ROI par canal et par marché", **TITLE_STYLE)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig