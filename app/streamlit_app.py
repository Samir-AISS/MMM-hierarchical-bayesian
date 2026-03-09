"""
streamlit_app.py — MMM Dashboard (Streamlit Cloud ready)
Charge les résultats pré-calculés depuis results/precomputed.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MMM Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

THEMES = {
    "Dark": {
        "bg": "#0D0F14", "surface": "#13161E", "surface2": "#1A1E2A",
        "border": "#252A38", "accent": "#4F7EFF", "accent2": "#00D4AA",
        "accent3": "#FF6B6B", "text": "#E8EAF0", "text_muted": "#6B7280",
    },
    "Light": {
        "bg": "#F5F7FA", "surface": "#FFFFFF", "surface2": "#EEF1F6",
        "border": "#D1D9E0", "accent": "#3B6FEF", "accent2": "#00A884",
        "accent3": "#E05555", "text": "#1A1D23", "text_muted": "#6B7280",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

_t = THEMES[st.session_state.theme]
COLORS = {
    **_t,
    "channels": {
        "tv": "#4F7EFF", "facebook": "#00D4AA",
        "search": "#FFB547", "ooh": "#FF6B6B", "print": "#A78BFA",
    }
}
CH_LABELS  = {"tv":"TV","facebook":"Facebook","search":"Search","ooh":"OOH","print":"Print"}
ALL_MARKETS = ["FR","DE","UK","IT","ES","NL","BE","PL","SE","NO"]
MARKET_NAMES = {
    "FR":"France","DE":"Germany","UK":"United Kingdom","IT":"Italy","ES":"Spain",
    "NL":"Netherlands","BE":"Belgium","PL":"Poland","SE":"Sweden","NO":"Norway"
}
CHANNELS   = ["tv","facebook","search","ooh","print"]
SPEND_COLS = [f"{c}_spend" for c in CHANNELS]

def hex_to_rgba(h, a=1.0):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=16,r=16,t=36,b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"], borderwidth=1),
    hoverlabel=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["border"],
                    font_color=COLORS["text"]),
)

def apply_layout(fig, title="", height=320):
    fig.update_layout(**PLOTLY_BASE, height=height,
        title=dict(text=title, font_size=12, font_color=COLORS["text_muted"],
                   x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=COLORS["border"], linecolor=COLORS["border"],
                     zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], linecolor=COLORS["border"],
                     zerolinecolor=COLORS["border"])
    return fig

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background:{COLORS['bg']};color:{COLORS['text']};}}
.stApp{{background:{COLORS['bg']};}}
[data-testid="stSidebar"]{{background:{COLORS['surface']};border-right:1px solid {COLORS['border']};}}
[data-testid="stSidebar"] *{{color:{COLORS['text']} !important;}}
.stSelectbox>div>div,.stMultiSelect>div>div{{background:{COLORS['surface2']};border:1px solid {COLORS['border']};border-radius:6px;}}
.block-container{{padding-top:1.5rem;padding-bottom:2rem;max-width:1400px;}}
.kpi{{background:{COLORS['surface']};border:1px solid {COLORS['border']};border-radius:10px;padding:18px 22px;}}
.kpi-label{{font-size:10px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:{COLORS['text_muted']};margin-bottom:7px;}}
.kpi-value{{font-size:26px;font-weight:600;font-family:'DM Mono',monospace;color:{COLORS['text']};line-height:1;}}
.kpi-sub{{font-size:11px;margin-top:5px;font-family:'DM Mono',monospace;}}
.pos{{color:{COLORS['accent2']};}} .neg{{color:{COLORS['accent3']};}} .neu{{color:{COLORS['text_muted']};}}
.sec{{font-size:11px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:{COLORS['text_muted']};border-bottom:1px solid {COLORS['border']};padding-bottom:9px;margin-bottom:18px;}}
.page-title{{font-size:22px;font-weight:600;color:{COLORS['text']};margin-bottom:4px;}}
.page-sub{{font-size:13px;color:{COLORS['text_muted']};margin-bottom:26px;}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="collapsedControl"]{{display:none !important;}}
[data-testid="stSidebarCollapseButton"]{{display:none !important;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-thumb{{background:{COLORS['border']};border-radius:3px;}}
</style>""", unsafe_allow_html=True)

# ── LOAD PRECOMPUTED ──────────────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    pkl_path = Path(__file__).parents[1] / "results" / "precomputed.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_raw():
    from src.data.data_loader import load_all_markets
    return load_all_markets()

def kpi(label, value, sub=None, cls="neu"):
    sub_html = f'<div class="kpi-sub {cls}">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub_html}</div>'

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)
def fmt(v):
    if v >= 1e6: return f"€{v/1e6:.2f}M"
    if v >= 1e3: return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

precomputed = load_precomputed()
data_raw    = load_raw()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:6px 0 22px">
      <div style="font-size:17px;font-weight:700;color:{COLORS['text']};letter-spacing:-.3px">MMM Analytics</div>
      <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:3px">Multi-Market · Bayesian MMM</div>
    </div>""", unsafe_allow_html=True)

    page = st.selectbox("", ["Overview","Model Performance","ROI Analysis",
                              "Budget Optimizer","Market Comparison"],
                        label_visibility="collapsed")
    st.markdown(f"<hr style='border-color:{COLORS['border']};margin:14px 0'>", unsafe_allow_html=True)
    market = st.selectbox("", ALL_MARKETS,
                          format_func=lambda x: f"{x} — {MARKET_NAMES[x]}",
                          label_visibility="collapsed")
    st.markdown(f"<hr style='border-color:{COLORS['border']};margin:14px 0'>", unsafe_allow_html=True)
    mode = "precomputed" if precomputed else "live"
    st.markdown(f"""
    <div style="font-size:11px;color:{COLORS['text_muted']};line-height:1.8">
      <span style="color:{COLORS['accent2'] if mode=='precomputed' else COLORS['accent3']}">
        {'● Precomputed' if mode=='precomputed' else '● Live mode'}</span><br>
      10 markets · 208 weeks<br>5 channels · Bayesian NUTS
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{COLORS['border']};margin:14px 0'>", unsafe_allow_html=True)
    current_theme = st.session_state.theme
    toggle_label = "☀️  Light mode" if current_theme == "Dark" else "🌙  Dark mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.theme = "Light" if current_theme == "Dark" else "Dark"
        st.rerun()

# ── DATA HELPERS ──────────────────────────────────────────────────────────────
def get_market_data(mkt):
    if isinstance(data_raw, dict):
        return data_raw[mkt]
    return data_raw[data_raw["market"] == mkt].copy()

def get_results(mkt):
    """Retourne les résultats pré-calculés ou entraîne à la volée."""
    if precomputed and mkt in precomputed:
        r = precomputed[mkt]
        df_train = pd.DataFrame(r["df_train"])
        df_test  = pd.DataFrame(r["df_test"])
        if "date" in df_train.columns:
            df_train["date"] = pd.to_datetime(df_train["date"])
            df_test["date"]  = pd.to_datetime(df_test["date"])
        contrib  = pd.DataFrame(r["contributions"])
        if "date" in contrib.columns:
            contrib["date"] = pd.to_datetime(contrib["date"])
        roi_df   = pd.DataFrame(r["roi"])
        y_pred   = np.array(r["y_pred_test"])
        m_train  = r["metrics_train"]
        m_test   = r["metrics_test"]
        return df_train, df_test, y_pred, m_train, m_test, roi_df, contrib
    else:
        from src.models.bayesian_mmm import BayesianMMM
        from src.data.data_loader import split_train_test
        from src.evaluation.metrics import compute_all_metrics
        df = get_market_data(mkt)
        df_train, df_test = split_train_test(df, test_ratio=0.2)
        with st.spinner(f"Training {mkt}…"):
            model = BayesianMMM(market=mkt)
            model.fit(df_train, draws=300, tune=300, chains=2, random_seed=42)
        y_pred  = model.predict(df_test)
        m_train = compute_all_metrics(df_train["revenue"].values, model.predict(df_train))
        m_test  = compute_all_metrics(df_test["revenue"].values, y_pred)
        roi_df  = model.get_roi(df_train)
        contrib = model.get_contributions(df_train)
        return df_train, df_test, y_pred, m_train, m_test, roi_df, contrib


# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════
if page == "Overview":
    df = get_market_data(market)
    st.markdown(f'<div class="page-title">{MARKET_NAMES[market]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Market Overview · 208 weeks · 5 channels</div>', unsafe_allow_html=True)

    total_rev   = df["revenue"].sum()
    avg_rev     = df["revenue"].mean()
    total_spend = df[SPEND_COLS].sum().sum()
    roi_g       = total_rev / total_spend
    top_ch      = df[SPEND_COLS].sum().idxmax().replace("_spend","")

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Total Revenue", fmt(total_rev), f"avg {fmt(avg_rev)}/wk", "pos"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Total Spend",   fmt(total_spend), f"avg {fmt(total_spend/208)}/wk"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Global ROI",    f"{roi_g:.2f}x", "revenue / spend", "pos"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Top Channel",   CH_LABELS[top_ch], "by total spend"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([2,1])

    with cl:
        sec("Revenue Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["revenue"], fill="tozeroy",
            fillcolor=hex_to_rgba(COLORS["accent"], 0.07),
            line=dict(color=COLORS["accent"], width=2), name="Revenue",
            hovertemplate="<b>%{x|%b %Y}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=270)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with cr:
        sec("Spend Mix")
        fig = go.Figure(go.Pie(
            labels=[CH_LABELS[c] for c in CHANNELS],
            values=[df[f"{c}_spend"].sum() for c in CHANNELS],
            hole=0.6, marker_colors=[COLORS["channels"][c] for c in CHANNELS],
            textinfo="percent", textfont_size=11,
            hovertemplate="<b>%{label}</b><br>€%{value:,.0f} · %{percent}<extra></extra>",
        ))
        apply_layout(fig, height=270)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec("Weekly Spend by Channel")
    fig = go.Figure()
    for ch in CHANNELS:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[f"{ch}_spend"], stackgroup="one",
            name=CH_LABELS[ch], line=dict(width=0),
            fillcolor=hex_to_rgba(COLORS["channels"][ch], 0.8),
            hovertemplate=f"<b>{CH_LABELS[ch]}</b><br>€%{{y:,.0f}}<extra></extra>",
        ))
    apply_layout(fig, height=230)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f'<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{MARKET_NAMES[market]} · Bayesian MMM · OLS fallback</div>', unsafe_allow_html=True)

    df_train, df_test, y_pred, m_train, m_test, _, _ = get_results(market)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("R² Test",    f"{m_test['r2']:.3f}",   "target ≥ 0.85", "pos" if m_test["r2"]>=0.85 else "neg"),   unsafe_allow_html=True)
    with c2: st.markdown(kpi("MAPE Test",  f"{m_test['mape']:.1f}%","target ≤ 15%",  "pos" if m_test["mape"]<=15  else "neg"),   unsafe_allow_html=True)
    with c3: st.markdown(kpi("RMSE Test",  fmt(m_test["rmse"]),     "root mean sq error"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("R² Train",   f"{m_train['r2']:.3f}",  "in-sample fit"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3,1])

    with cl:
        sec("Actual vs Predicted — Test Set")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_test["date"], y=df_test["revenue"],
            name="Actual", line=dict(color=COLORS["text"], width=1.5),
            hovertemplate="<b>Actual</b> €%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_test["date"], y=y_pred,
            name="Predicted", line=dict(color=COLORS["accent"], width=2, dash="dot"),
            hovertemplate="<b>Predicted</b> €%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=290)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with cr:
        sec("Predicted vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_test["revenue"].values, y=y_pred, mode="markers",
            marker=dict(color=COLORS["accent"], size=5, opacity=0.7),
            hovertemplate="Actual €%{x:,.0f}<br>Pred €%{y:,.0f}<extra></extra>",
        ))
        lim = [min(df_test["revenue"].min(), y_pred.min())*0.97,
               max(df_test["revenue"].max(), y_pred.max())*1.03]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
            line=dict(color=COLORS["accent3"], width=1, dash="dash"), showlegend=False))
        apply_layout(fig, height=290)
        fig.update_xaxes(title="Actual (€)")
        fig.update_yaxes(title="Predicted (€)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec("Residuals")
    residuals = df_test["revenue"].values - y_pred
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Over time","Distribution"])
    fig.add_trace(go.Scatter(x=df_test["date"], y=residuals, mode="lines",
        line=dict(color=COLORS["accent2"], width=1.5),
        hovertemplate="€%{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line_color=COLORS["accent3"], line_dash="dash", row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=18,
        marker_color=COLORS["accent"], opacity=0.75), row=1, col=2)
    fig.update_layout(**PLOTLY_BASE, height=240, showlegend=False)
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROI ANALYSIS
# ════════════════════════════════════════════════════════════════════════
elif page == "ROI Analysis":
    st.markdown(f'<div class="page-title">ROI Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{MARKET_NAMES[market]} · Channel Efficiency</div>', unsafe_allow_html=True)

    df_train, _, _, _, _, roi_df, contrib = get_results(market)
    roi_map = {r["channel"]: r["roi"] for r in roi_df.to_dict("records")}

    cols = st.columns(5)
    for i, ch in enumerate(CHANNELS):
        v = roi_map.get(ch, 0)
        cls = "pos" if v >= 2 else "neu" if v >= 1 else "neg"
        with cols[i]: st.markdown(kpi(CH_LABELS[ch], f"{v:.2f}x", "ROI ratio", cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([1,1])

    with cl:
        sec("ROI by Channel")
        rs = roi_df.sort_values("roi", ascending=True)
        fig = go.Figure(go.Bar(
            x=rs["roi"], y=[CH_LABELS[c] for c in rs["channel"]],
            orientation="h",
            marker=dict(color=[COLORS["channels"][c] for c in rs["channel"]], line_width=0),
            text=[f"{v:.2f}x" for v in rs["roi"]], textposition="outside",
            textfont=dict(color=COLORS["text"], size=11, family="DM Mono"),
            hovertemplate="<b>%{y}</b> ROI: %{x:.3f}x<extra></extra>",
        ))
        fig.add_vline(x=1, line_color=COLORS["accent3"], line_dash="dash",
                      annotation_text="break-even",
                      annotation_font_color=COLORS["text_muted"], annotation_font_size=10)
        apply_layout(fig, height=290)
        fig.update_xaxes(title="ROI (revenue / spend)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with cr:
        sec("Revenue Contribution")
        contrib_mean = {}
        for ch in CHANNELS:
            if ch in contrib.columns:
                contrib_mean[CH_LABELS[ch]] = float(contrib[ch].mean())
        if "base" in contrib.columns:
            contrib_mean["Base"] = float(contrib["base"].mean())

        ch_colors = [COLORS["channels"].get(ch, COLORS["border"]) for ch in CHANNELS] + [COLORS["border"]]
        fig = go.Figure(go.Pie(
            labels=list(contrib_mean.keys()), values=list(contrib_mean.values()),
            hole=0.55, marker_colors=ch_colors[:len(contrib_mean)],
            textinfo="percent+label", textfont_size=11,
            hovertemplate="<b>%{label}</b><br>€%{value:,.0f} · %{percent}<extra></extra>",
        ))
        apply_layout(fig, height=290)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec("Channel Contributions Over Time")
    date_col = contrib["date"] if "date" in contrib.columns else df_train["date"]
    fig = go.Figure()
    for ch in CHANNELS:
        if ch in contrib.columns:
            fig.add_trace(go.Scatter(
                x=date_col, y=contrib[ch], stackgroup="one",
                name=CH_LABELS[ch], line=dict(width=0),
                fillcolor=hex_to_rgba(COLORS["channels"][ch], 0.75),
                hovertemplate=f"<b>{CH_LABELS[ch]}</b><br>€%{{y:,.0f}}<extra></extra>",
            ))
    apply_layout(fig, height=250)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — BUDGET OPTIMIZER
# ════════════════════════════════════════════════════════════════════════
elif page == "Budget Optimizer":
    st.markdown(f'<div class="page-title">Budget Optimizer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{MARKET_NAMES[market]} · Allocation Simulation</div>', unsafe_allow_html=True)

    df_train, _, _, _, _, roi_df, _ = get_results(market)
    roi_map = {r["channel"]: r["roi"] for r in roi_df.to_dict("records")}
    current = {ch: float(df_train[f"{ch}_spend"].mean()) for ch in CHANNELS}
    current_total = sum(current.values())

    ctrl, res = st.columns([1,2])
    with ctrl:
        sec("Parameters")
        budget = st.slider("Weekly Budget (€)",
                           int(current_total*0.5), int(current_total*2),
                           int(current_total), step=1000, format="€%d")
        st.markdown(f"<div style='font-size:11px;color:{COLORS['text_muted']};margin:14px 0 8px;text-transform:uppercase;letter-spacing:.06em'>Min allocation %</div>", unsafe_allow_html=True)
        defaults = {"tv":20,"facebook":15,"search":15,"ooh":10,"print":5}
        mins = {ch: st.slider(CH_LABELS[ch], 0, 40, defaults[ch], 5, key=f"m_{ch}") for ch in CHANNELS}

    with res:
        min_spend = {ch: budget * mins[ch]/100 for ch in CHANNELS}
        remaining = budget - sum(min_spend.values())
        if remaining < 0:
            st.warning("Minimum constraints exceed total budget.")
            alloc = {ch: budget/5 for ch in CHANNELS}
        else:
            roi_sum = sum(roi_map.get(ch,1) for ch in CHANNELS)
            alloc = {ch: min_spend[ch] + remaining*(roi_map.get(ch,1)/roi_sum) for ch in CHANNELS}

        cur_rev = sum(current[ch]*roi_map.get(ch,1) for ch in CHANNELS)
        opt_rev = sum(alloc[ch]*roi_map.get(ch,1) for ch in CHANNELS)
        delta   = (opt_rev - cur_rev)/cur_rev*100

        sec("Optimal Allocation")
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi("Budget",     fmt(budget),   "weekly"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Est. Revenue",fmt(opt_rev), f"{delta:+.1f}% vs current", "pos" if delta>0 else "neg"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Blended ROI", f"{opt_rev/budget:.2f}x", "revenue / spend", "pos"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current", x=[CH_LABELS[c] for c in CHANNELS],
            y=[current[c] for c in CHANNELS],
            marker=dict(color=COLORS["surface2"], line_color=COLORS["border"], line_width=1),
            hovertemplate="<b>Current %{x}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Optimal", x=[CH_LABELS[c] for c in CHANNELS],
            y=[alloc[c] for c in CHANNELS],
            marker=dict(color=[COLORS["channels"][c] for c in CHANNELS], line_width=0),
            hovertemplate="<b>Optimal %{x}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=260)
        fig.update_layout(barmode="group", bargap=0.25)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        rows = [{"Channel":CH_LABELS[ch],
                 "Current":f"€{current[ch]:,.0f}",
                 "Optimal":f"€{alloc[ch]:,.0f}",
                 "Δ":f"{(alloc[ch]-current[ch])/current[ch]*100:+.1f}%",
                 "ROI":f"{roi_map.get(ch,0):.2f}x"} for ch in CHANNELS]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 5 — MARKET COMPARISON
# ════════════════════════════════════════════════════════════════════════
elif page == "Market Comparison":
    st.markdown('<div class="page-title">Market Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">All Markets · Cross-Market Analysis</div>', unsafe_allow_html=True)

    rows = []
    for mkt in ALL_MARKETS:
        df_m = get_market_data(mkt)
        rev  = df_m["revenue"].sum()
        spd  = df_m[SPEND_COLS].sum().sum()
        rows.append({"market":mkt,"name":MARKET_NAMES[mkt],
                     "revenue":rev,"spend":spd,"roi":rev/spd,"avg_rev":df_m["revenue"].mean()})
    summary = pd.DataFrame(rows).sort_values("revenue", ascending=False)

    best_roi = summary.loc[summary["roi"].idxmax()]
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Markets", "10", "EU panel"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Total Revenue", fmt(summary["revenue"].sum()), "all markets","pos"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Best ROI", best_roi["market"], f"{best_roi['roi']:.2f}x","pos"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Top Revenue", summary.iloc[0]["market"], fmt(summary.iloc[0]["revenue"]),"pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3,2])

    with cl:
        sec("Revenue vs Spend")
        fig = go.Figure(go.Scatter(
            x=summary["spend"], y=summary["revenue"],
            mode="markers+text", text=summary["market"],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["text"]),
            marker=dict(
                size=summary["avg_rev"]/summary["avg_rev"].max()*28+8,
                color=[COLORS["accent"] if r>=summary["roi"].mean() else COLORS["text_muted"]
                       for r in summary["roi"]],
                line=dict(color=COLORS["border"],width=1)),
            hovertemplate="<b>%{text}</b><br>Revenue €%{y:,.0f}<br>Spend €%{x:,.0f}<extra></extra>",
            showlegend=False,
        ))
        apply_layout(fig, height=320)
        fig.update_xaxes(title="Total Spend (€)")
        fig.update_yaxes(title="Total Revenue (€)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with cr:
        sec("ROI Ranking")
        rs = summary.sort_values("roi", ascending=True)
        avg = summary["roi"].mean()
        fig = go.Figure(go.Bar(
            x=rs["roi"], y=rs["market"], orientation="h",
            marker=dict(color=[COLORS["accent"] if r>=avg else COLORS["text_muted"] for r in rs["roi"]], line_width=0),
            text=[f"{v:.2f}x" for v in rs["roi"]], textposition="outside",
            textfont=dict(color=COLORS["text"], size=10, family="DM Mono"),
            hovertemplate="<b>%{y}</b> ROI: %{x:.3f}x<extra></extra>",
        ))
        fig.add_vline(x=avg, line_color=COLORS["accent2"], line_dash="dash",
                      annotation_text=f"avg {avg:.2f}x",
                      annotation_font_color=COLORS["accent2"], annotation_font_size=10)
        apply_layout(fig, height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    sec("Channel Spend Heatmap")
    hmap = []
    for mkt in ALL_MARKETS:
        df_m = get_market_data(mkt)
        hmap.append([df_m[f"{ch}_spend"].mean() for ch in CHANNELS])
    hmap_arr = np.array(hmap)
    hmap_norm = hmap_arr / hmap_arr.max(axis=0)

    fig = go.Figure(go.Heatmap(
        z=hmap_norm, x=[CH_LABELS[c] for c in CHANNELS], y=ALL_MARKETS,
        colorscale=[[0,COLORS["surface"]],[0.5,hex_to_rgba(COLORS["accent"],0.5)],[1,COLORS["accent"]]],
        text=[[fmt(hmap_arr[i,j]) for j in range(len(CHANNELS))] for i in range(len(ALL_MARKETS))],
        texttemplate="%{text}", textfont_size=10, showscale=True,
        hovertemplate="<b>%{y} — %{x}</b><br>%{text}<extra></extra>",
    ))
    apply_layout(fig, height=310)
    fig.update_layout(margin=dict(l=40,r=60,t=30,b=16))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})