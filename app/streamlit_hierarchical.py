"""
streamlit_hierarchical.py
--------------------------
Dashboard Hierarchical MMM — Meridian-inspired
Light / Dark mode · st.tabs() navigation
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Hierarchical MMM", layout="wide",
                   initial_sidebar_state="collapsed")

# ── THEMES ────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#0D0F14", "surface": "#13161E", "surface2": "#1A1E2A",
        "border": "#252A38", "text": "#E8EAF0", "text_muted": "#6B7280",
        "accent": "#4F7EFF", "accent2": "#00D4AA", "accent3": "#FF6B6B",
        "channels": ["#4F7EFF","#00D4AA","#FFB547","#FF6B6B","#A78BFA"],
        "icon": "☀️", "label": "Light mode",
    },
    "light": {
        "bg": "#F5F6FA", "surface": "#FFFFFF", "surface2": "#EEF0F7",
        "border": "#DDE1EF", "text": "#1A1D2E", "text_muted": "#6B7280",
        "accent": "#3B65E8", "accent2": "#00A887", "accent3": "#E04040",
        "channels": ["#3B65E8","#00A887","#E08A00","#E04040","#7C5BD4"],
        "icon": "🌙", "label": "Dark mode",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

C = THEMES[st.session_state.theme]

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'Syne',sans-serif;background:{C['bg']};color:{C['text']};}}
.stApp{{background:{C['bg']};}}
.block-container{{padding-top:2rem;max-width:1400px;}}
.stTabs [data-baseweb="tab-list"]{{
    background:{C['surface']};
    border-radius:10px;
    padding:4px;
    border:1px solid {C['border']};
    gap:2px;
}}
.stTabs [data-baseweb="tab"]{{
    background:transparent;
    color:{C['text_muted']};
    border-radius:8px;
    font-family:'Syne',sans-serif;
    font-weight:600;
    font-size:13px;
    padding:8px 18px;
    border:none;
}}
.stTabs [aria-selected="true"]{{
    background:{C['accent']} !important;
    color:white !important;
}}
.stTabs [data-baseweb="tab-border"]{{display:none;}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:24px;}}
.kpi{{background:{C['surface']};border:1px solid {C['border']};border-radius:10px;padding:16px 20px;}}
.kpi-label{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{C['text_muted']};margin-bottom:8px;}}
.kpi-value{{font-size:24px;font-weight:700;font-family:'DM Mono',monospace;color:{C['text']};}}
.kpi-sub{{font-size:11px;margin-top:5px;font-family:'DM Mono',monospace;}}
.pos{{color:{C['accent2']};}} .neg{{color:{C['accent3']};}} .neu{{color:{C['text_muted']};}}
.sec{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{C['text_muted']};border-bottom:1px solid {C['border']};padding-bottom:8px;margin-bottom:16px;margin-top:8px;}}
.page-title{{font-size:24px;font-weight:800;margin-bottom:4px;color:{C['text']};}}
.page-sub{{font-size:13px;color:{C['text_muted']};margin-bottom:24px;}}
.info-box{{background:{C['surface2']};border:1px solid {C['border']};border-left:3px solid {C['accent']};border-radius:8px;padding:14px 18px;font-size:12px;font-family:'DM Mono',monospace;line-height:2;color:{C['text_muted']};}}
#MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}header{{visibility:hidden;}}
</style>""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def kpi(label, value, sub=None, cls="neu"):
    sub_html = f'<div class="kpi-sub {cls}">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub_html}</div>'

def sec(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def apply_layout(fig, title="", height=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne, sans-serif", color=C["text"], size=12),
        margin=dict(l=16,r=16,t=36,b=16), height=height,
        title=dict(text=title, font_size=12, font_color=C["text_muted"], x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"], borderwidth=1),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["border"], font_color=C["text"]),
    )
    fig.update_xaxes(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"])
    return fig

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    results_dir = Path(__file__).parents[1] / "results" / "hierarchical"
    pkls = sorted(results_dir.glob("hierarchical_*.pkl"))
    if not pkls:
        return None
    with open(pkls[-1], "rb") as f:
        return pickle.load(f)

def compute_roi(_data):
    idata        = _data["idata"]
    adstock_norm = _data["adstock_norm"]
    spend_max    = _data["spend_max"]
    rev_scale    = _data["rev_scale"]
    geos         = _data["selected_geos"]
    channels     = _data["channels"]
    beta_mean    = idata.posterior["beta"].values.mean(axis=(0,1))
    ec50_mean    = idata.posterior["ec50"].values.mean(axis=(0,1))
    slope_mean   = idata.posterior["slope"].values.mean(axis=(0,1))
    rows = []
    for g_i, geo in enumerate(geos):
        total_spend = (adstock_norm[g_i] * spend_max[0,0]).sum(axis=0)
        for c_i, ch in enumerate(channels):
            if total_spend[c_i] < 1e-6: continue
            ads     = adstock_norm[g_i, :, c_i]
            sat     = ads**slope_mean[c_i] / (ec50_mean[c_i]**slope_mean[c_i] + ads**slope_mean[c_i])
            contrib = (beta_mean[g_i, c_i] * sat * rev_scale).sum()
            rows.append({"geo":geo,"channel":ch,"roi":contrib/total_spend[c_i],
                         "contrib":contrib,"spend":float(total_spend[c_i])})
    return pd.DataFrame(rows)

def compute_convergence(_data):
    import arviz as az
    return az.summary(_data["idata"], var_names=["mu_beta","baseline","sigma"])

# ── HEADER ────────────────────────────────────────────────────────────────────
data = load_results()

hc1, hc2 = st.columns([8,1])
with hc1:
    st.markdown('<div class="page-title">Hierarchical MMM</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Bayesian · Meridian-inspired · PyMC · Shared priors across geos</div>', unsafe_allow_html=True)
with hc2:
    if st.button(C['label'], use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

if not data:
    st.error("No results found. Run `python scripts/train_hierarchical.py` first.")
    st.stop()

roi_df   = compute_roi(data)
conv_df  = compute_convergence(data)
idata    = data["idata"]
geos     = data["selected_geos"]
channels = data["channels"]
n_geos   = len(geos)
n_ch     = len(channels)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "ROI Analysis",
    "Geo Comparison",
    "Hierarchical Structure",
    "Diagnostics",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    max_rhat = conv_df["r_hat"].max()
    avg_roi  = roi_df["roi"].mean()
    best_ch  = roi_df.groupby("channel")["roi"].mean().idxmax()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Geos Modeled", str(n_geos), f"{n_ch} channels","pos"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Max R-hat", f"{max_rhat:.3f}",
                             "✅ converged" if max_rhat<1.05 else "⚠️ not converged",
                             "pos" if max_rhat<1.05 else "neg"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg ROI", f"{avg_roi:.2f}x","all channels · all geos","pos"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Best Channel", best_ch,"by avg ROI","pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        sec("Model Architecture — Meridian-inspired")
        st.markdown(f"""<div class="info-box">
Revenue(g,t) = <span style="color:{C['accent2']}">baseline(g)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Σ <span style="color:{C['accent']}">β(g,c)</span> × Hill(Adstock(spend))<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Σ γ_k × control(g,t,k)<br><br>
<span style="color:{C['accent2']}">Hierarchical priors :</span><br>
β(g,c) ~ Normal(<span style="color:{C['accent']}">μ_β(c)</span>, σ_β(c))<br>
μ_β(c) ~ HalfNormal(1.0) ← global<br>
baseline(g) ~ Normal(μ_base, σ_base)
        </div>""", unsafe_allow_html=True)

    with cr:
        sec("Global ROI by Channel")
        roi_global = roi_df.groupby("channel")["roi"].mean().reset_index().sort_values("roi",ascending=True)
        fig = go.Figure(go.Bar(
            x=roi_global["roi"], y=roi_global["channel"], orientation="h",
            marker_color=C["channels"][:len(roi_global)],
            text=[f"{v:.2f}x" for v in roi_global["roi"]],
            textposition="outside",
            textfont=dict(color=C["text"],size=11,family="DM Mono"),
            hovertemplate="<b>%{y}</b> — ROI: %{x:.3f}x<extra></extra>",
        ))
        fig.add_vline(x=1, line_color=C["accent3"], line_dash="dash",
                      annotation_text="break-even", annotation_font_color=C["text_muted"])
        apply_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec(f"Global Channel Effects — μ_beta Posterior ({n_ch} channels)")
    mu_beta = idata.posterior["mu_beta"].values.reshape(-1, n_ch)
    fig = go.Figure()
    for c_i, ch in enumerate(channels):
        col = C["channels"][c_i % len(C["channels"])]
        r,g_,b_ = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
        fig.add_trace(go.Violin(
            x=[ch]*len(mu_beta), y=mu_beta[:,c_i], name=ch,
            fillcolor=f"rgba({r},{g_},{b_},0.35)", line_color=col,
            box_visible=True, meanline_visible=True,
            hovertemplate=f"<b>{ch}</b><br>μ_β=%{{y:.3f}}<extra></extra>",
        ))
    apply_layout(fig, height=300)
    fig.update_layout(violinmode="overlay", showlegend=False)
    fig.update_yaxes(title="μ_beta")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROI ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    roi_global = roi_df.groupby("channel")["roi"].mean()
    cols = st.columns(n_ch)
    for i, ch in enumerate(channels):
        v = roi_global.get(ch, 0)
        cls = "pos" if v>=1.5 else "neu" if v>=1 else "neg"
        with cols[i]: st.markdown(kpi(ch, f"{v:.2f}x","avg ROI",cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        sec("ROI Distribution by Channel")
        fig = go.Figure()
        for c_i, ch in enumerate(channels):
            col = C["channels"][c_i % len(C["channels"])]
            fig.add_trace(go.Box(
                y=roi_df[roi_df["channel"]==ch]["roi"], name=ch,
                marker_color=col, line_color=col,
                hovertemplate=f"<b>{ch}</b><br>ROI: %{{y:.3f}}x<extra></extra>",
            ))
        fig.add_hline(y=1, line_color=C["accent3"], line_dash="dash",
                      annotation_text="break-even", annotation_font_color=C["accent3"])
        apply_layout(fig, height=320)
        fig.update_yaxes(title="ROI")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with cr:
        sec("Revenue Contribution by Channel")
        contrib = roi_df.groupby("channel")["contrib"].sum()
        fig = go.Figure(go.Pie(
            labels=contrib.index, values=contrib.values, hole=0.55,
            marker_colors=C["channels"][:len(contrib)],
            textinfo="percent+label", textfont_size=11,
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=320)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec(f"ROI Heatmap — {n_geos} Geos × {n_ch} Channels")
    roi_pivot = roi_df.pivot(index="geo", columns="channel", values="roi")
    fig = go.Figure(go.Heatmap(
        z=roi_pivot.values, x=roi_pivot.columns.tolist(), y=roi_pivot.index.tolist(),
        colorscale=[[0,C["accent3"]],[0.5,C["surface2"]],[1,C["accent2"]]],
        zmid=1,
        text=[[f"{v:.2f}x" for v in row] for row in roi_pivot.values],
        texttemplate="%{text}", textfont_size=11,
        hovertemplate="<b>%{y} — %{x}</b><br>ROI: %{z:.3f}x<extra></extra>",
    ))
    apply_layout(fig, height=max(280, n_geos*55))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEO COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    geo_summary = roi_df.groupby("geo").agg(
        avg_roi=("roi","mean"), total_contrib=("contrib","sum"), total_spend=("spend","sum"),
    ).reset_index()
    geo_summary["blended_roi"] = geo_summary["total_contrib"] / geo_summary["total_spend"]
    avg_blended = geo_summary["blended_roi"].mean()
    best_geo    = geo_summary.loc[geo_summary["blended_roi"].idxmax(),"geo"]
    worst_geo   = geo_summary.loc[geo_summary["blended_roi"].idxmin(),"geo"]

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Geos",str(n_geos),"modeled"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Best Geo",best_geo,f"{geo_summary['blended_roi'].max():.2f}x ROI","pos"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Weakest Geo",worst_geo,f"{geo_summary['blended_roi'].min():.2f}x ROI","neg"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Avg ROI",f"{avg_blended:.2f}x","blended all geos","neu"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        sec("Blended ROI by Geo")
        gs = geo_summary.sort_values("blended_roi", ascending=True)
        fig = go.Figure(go.Bar(
            x=gs["blended_roi"], y=gs["geo"], orientation="h",
            marker_color=[C["accent2"] if r>=avg_blended else C["accent3"] for r in gs["blended_roi"]],
            text=[f"{v:.2f}x" for v in gs["blended_roi"]],
            textposition="outside",
            textfont=dict(color=C["text"],size=11,family="DM Mono"),
            hovertemplate="<b>%{y}</b><br>ROI: %{x:.3f}x<extra></extra>",
        ))
        fig.add_vline(x=avg_blended, line_color=C["accent"], line_dash="dash",
                      annotation_text=f"avg {avg_blended:.2f}x",
                      annotation_font_color=C["accent"])
        apply_layout(fig, height=max(280, n_geos*50))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with cr:
        sec("Revenue vs Spend — Bubble Chart")
        fig = go.Figure(go.Scatter(
            x=geo_summary["total_spend"], y=geo_summary["total_contrib"],
            mode="markers+text", text=geo_summary["geo"],
            textposition="top center",
            textfont=dict(size=11,color=C["text_muted"]),
            marker=dict(
                size=geo_summary["blended_roi"]*18,
                color=[C["accent2"] if r>=avg_blended else C["accent3"] for r in geo_summary["blended_roi"]],
                opacity=0.85, line=dict(color=C["border"],width=1),
            ),
            hovertemplate="<b>%{text}</b><br>Spend: $%{x:,.0f}<br>Contrib: $%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=max(280,n_geos*50))
        fig.update_xaxes(title="Total Spend ($)")
        fig.update_yaxes(title="Revenue Contribution ($)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec("Revenue Contribution by Channel & Geo")
    fig = go.Figure()
    for c_i, ch in enumerate(channels):
        ch_data = roi_df[roi_df["channel"]==ch]
        ch_dict = dict(zip(ch_data["geo"], ch_data["contrib"]))
        fig.add_trace(go.Bar(
            name=ch, x=geos,
            y=[ch_dict.get(g,0) for g in geos],
            marker_color=C["channels"][c_i % len(C["channels"])],
            hovertemplate=f"<b>%{{x}}</b> — {ch}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    apply_layout(fig, height=320)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Revenue Contribution ($)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — HIERARCHICAL STRUCTURE
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    mu_beta_post  = idata.posterior["mu_beta"].values.reshape(-1, n_ch)
    beta_post     = idata.posterior["beta"].values.reshape(-1, n_geos, n_ch)
    sig_beta_post = idata.posterior["sig_beta"].values.reshape(-1, n_ch)
    baseline_post = idata.posterior["baseline"].values.reshape(-1, n_geos)
    mu_base_post  = idata.posterior["mu_base"].values.reshape(-1)

    avg_mu  = mu_beta_post.mean(axis=0).mean()
    avg_sig = sig_beta_post.mean(axis=0).mean()

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(kpi("Global μ_beta",f"{avg_mu:.3f}","avg across channels"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Global σ_beta",f"{avg_sig:.3f}","cross-geo variance"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Shrinkage",f"{max(0,1-avg_sig):.1%}","toward global mean","pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec(f"Geo-level β by Channel — Shrinkage Effect ({n_geos} geos × {n_ch} channels)")

    fig = make_subplots(rows=1, cols=n_ch, subplot_titles=channels)
    for c_i, ch in enumerate(channels):
        col = C["channels"][c_i % len(C["channels"])]
        global_mean = mu_beta_post[:,c_i].mean()
        for g_i, geo in enumerate(geos):
            geo_mean = beta_post[:,g_i,c_i].mean()
            geo_std  = beta_post[:,g_i,c_i].std()
            fig.add_trace(go.Scatter(
                x=[geo], y=[geo_mean],
                error_y=dict(type="data",array=[geo_std*2],visible=True,color=col),
                mode="markers", marker=dict(color=col,size=12),
                name=geo, showlegend=(c_i==0),
                hovertemplate=f"<b>{geo}</b> — {ch}<br>β={geo_mean:.3f}±{geo_std:.3f}<extra></extra>",
            ), row=1, col=c_i+1)
        fig.add_hline(y=global_mean, line_color=C["text_muted"], line_dash="dash",
                      line_width=1, row=1, col=c_i+1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne",color=C["text"],size=11),
        height=380, margin=dict(l=16,r=16,t=40,b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=C["border"]),
        hoverlabel=dict(bgcolor=C["surface2"],font_color=C["text"]),
    )
    fig.update_xaxes(showticklabels=False, gridcolor=C["border"], linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], linecolor=C["border"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec(f"Baseline Distribution by Geo ({n_geos} geos)")
    global_base = mu_base_post.mean()
    fig = go.Figure()
    for g_i, geo in enumerate(geos):
        col = C["channels"][g_i % len(C["channels"])]
        r,g_,b_ = int(col[1:3],16), int(col[3:5],16), int(col[5:7],16)
        fig.add_trace(go.Violin(
            x=[geo]*len(baseline_post[:,g_i]), y=baseline_post[:,g_i],
            name=geo, fillcolor=f"rgba({r},{g_},{b_},0.35)", line_color=col,
            box_visible=True, meanline_visible=True,
            hovertemplate=f"<b>{geo}</b><br>baseline=%{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=global_base, line_color=C["accent2"], line_dash="dash",
                  annotation_text=f"global μ_base={global_base:.2f}",
                  annotation_font_color=C["accent2"])
    apply_layout(fig, height=320)
    fig.update_layout(violinmode="overlay", showlegend=False)
    fig.update_yaxes(title="Baseline value")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    max_rhat = conv_df["r_hat"].max()
    n_params = len(conv_df)
    pct_ok   = (conv_df["r_hat"] < 1.05).mean()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Max R-hat",f"{max_rhat:.3f}",
                             "✅ converged" if max_rhat<1.05 else "⚠️ not converged",
                             "pos" if max_rhat<1.05 else "neg"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Parameters",str(n_params),"monitored"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Converged",f"{pct_ok:.0%}","R-hat < 1.05",
                             "pos" if pct_ok==1 else "neg"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Target","< 1.05","R-hat threshold"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("R-hat by Parameter")

    rhat_sorted = conv_df["r_hat"].sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=rhat_sorted.values, y=rhat_sorted.index, orientation="h",
        marker_color=[C["accent2"] if r<1.05 else C["accent3"] for r in rhat_sorted.values],
        text=[f"{v:.3f}" for v in rhat_sorted.values],
        textposition="outside",
        textfont=dict(color=C["text"],size=10,family="DM Mono"),
        hovertemplate="<b>%{y}</b><br>R-hat: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=1.05, line_color=C["accent3"], line_dash="dash",
                  annotation_text="threshold 1.05", annotation_font_color=C["accent3"])
    fig.add_vline(x=1.0, line_color=C["accent2"], line_dash="dot",
                  annotation_text="ideal", annotation_font_color=C["accent2"])
    apply_layout(fig, height=max(300, n_params*22))
    fig.update_xaxes(title="R-hat", range=[0.95, max(1.1, max_rhat+0.05)])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec("Posterior Summary Table")
    disp = conv_df[["mean","sd","hdi_3%","hdi_97%","r_hat"]].round(3)
    st.dataframe(disp, use_container_width=True)