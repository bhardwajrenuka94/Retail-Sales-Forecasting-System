"""
app.py — Retail Sales Forecasting Dashboard
Built by Renuka Bhardwaj | Data Scientist
Models: Holt-Winters + SARIMA + Prophet | MAPE ~8% | 100K+ records | 52 stores
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Forecast | Renuka Bhardwaj",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
# CSS — Dark premium (same family as churn app)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
 
  :root {
    --bg      : #0a0d14;
    --surface : #111520;
    --card    : #161c2d;
    --accent  : #00e5ff;
    --green   : #00d68f;
    --amber   : #ffd60a;
    --red     : #ff6b6b;
    --text    : #e8eaf0;
    --muted   : #6b7280;
    --border  : #1e2640;
  }
 
  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
  }
 
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
 
  h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
  }
 
  /* Tabs */
  [data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }
 
  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
  }
 
  /* Metric cards */
  .kpi {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .kpi-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }
  .kpi-lbl {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
  }
  .kpi-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 3px;
  }
  .kpi.green { border-top-color: var(--green); }
  .kpi.green .kpi-val { color: var(--green); }
  .kpi.amber { border-top-color: var(--amber); }
  .kpi.amber .kpi-val { color: var(--amber); }
  .kpi.red   { border-top-color: var(--red); }
  .kpi.red   .kpi-val { color: var(--red); }
 
  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-right: 4px;
  }
  .b-hw     { background:#ffd60a22; color:#ffd60a; border:1px solid #ffd60a44; }
  .b-sarima { background:#00e5ff22; color:#00e5ff; border:1px solid #00e5ff44; }
  .b-prop   { background:#00d68f22; color:#00d68f; border:1px solid #00d68f44; }
  .b-mape   { background:#a78bfa22; color:#a78bfa; border:1px solid #a78bfa44; }
 
  /* Insight cards */
  .insight {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 14px 16px;
    margin: 6px 0;
  }
  .insight b { color: var(--accent); }
 
  /* Store row */
  .store-row {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.85rem;
  }
 
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
@st.cache_data
def load_all():
    sales     = pd.read_csv("sales_data.csv",    parse_dates=["date"])
    monthly   = pd.read_csv("monthly_sales.csv", parse_dates=["date"])
    store_m   = pd.read_csv("store_monthly.csv", parse_dates=["date"])
    results   = pd.read_csv("model_results.csv")
    forecasts = pd.read_csv("forecasts.csv",     parse_dates=["ds"])
    return sales, monthly, store_m, results, forecasts
 
try:
    sales_df, monthly_df, store_monthly, results_df, forecasts_df = load_all()
    loaded = True
except FileNotFoundError as e:
    loaded = False
    err_msg = str(e)
 
# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Retail Forecast")
    st.markdown("---")
 
    if loaded:
        stores     = sorted(results_df["store_id"].unique())
        categories = sorted(monthly_df["category"].unique())
 
        sel_store = st.selectbox("Select Store", stores)
        sel_cat   = st.selectbox("Category View", ["All"] + list(categories))
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Model Wins**")
 
        hw_n  = (results_df["best_model"] == "Holt-Winters").sum()
        sa_n  = (results_df["best_model"] == "SARIMA").sum()
        pr_n  = (results_df["best_model"] == "Prophet").sum()
 
        st.markdown(f"""
        <div style='margin:8px 0'>
          <span class='badge b-hw'>Holt-Winters</span>
          <span style='color:#e8eaf0; font-size:0.85rem'>{hw_n} stores</span>
        </div>
        <div style='margin:8px 0'>
          <span class='badge b-sarima'>SARIMA</span>
          <span style='color:#e8eaf0; font-size:0.85rem'>{sa_n} stores</span>
        </div>
        <div style='margin:8px 0'>
          <span class='badge b-prop'>Prophet</span>
          <span style='color:#e8eaf0; font-size:0.85rem'>{pr_n} stores</span>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("---")
        overall_mape = results_df["mape"].mean()
        st.markdown(f"""
        <div style='text-align:center; padding:16px; background:var(--card);
             border:1px solid var(--border); border-radius:10px'>
          <div style='font-family:Space Mono; font-size:2.2rem;
               color:#00d68f; font-weight:700'>{overall_mape:.1f}%</div>
          <div style='color:var(--muted); font-size:0.68rem;
               text-transform:uppercase; letter-spacing:0.1em; margin-top:4px'>Overall MAPE</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("""
    <div style='color:#6b7280; font-size:0.72rem; line-height:1.7'>
      Built by <b style='color:#00e5ff'>Renuka Bhardwaj</b><br>
      Data Scientist<br>AnalytixLabs Certified<br>Karnal, Haryana
    </div>
    """, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# Error state
# ─────────────────────────────────────────────
if not loaded:
    st.error(f"Files not found: {err_msg}")
    st.code("""python generate_data.py
python train_models.py
streamlit run app.py""", language="bash")
    st.stop()
 
# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2rem; margin-bottom:4px'>Retail Sales Forecasting System</h1>
<p style='color:#6b7280; font-size:0.92rem; margin-bottom:28px'>
  Holt-Winters &nbsp;·&nbsp; SARIMA &nbsp;·&nbsp; Facebook Prophet &nbsp;|&nbsp;
  100,000+ records &nbsp;·&nbsp; 52 stores &nbsp;·&nbsp; 5 categories &nbsp;·&nbsp; MAPE ~8%
</p>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# KPI Strip
# ─────────────────────────────────────────────
total_rec   = len(sales_df)
total_rev   = sales_df["sales"].sum()
n_stores    = results_df["store_id"].nunique()
overall_mape = results_df["mape"].mean()
date_range  = f"{sales_df['date'].min().strftime('%b %y')} – {sales_df['date'].max().strftime('%b %y')}"
best_store  = results_df.loc[results_df["mape"].idxmin()]
 
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class='kpi'>
      <div class='kpi-val'>{overall_mape:.1f}%</div>
      <div class='kpi-lbl'>Overall MAPE</div>
      <div class='kpi-sub'>Across 52 stores</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='kpi green'>
      <div class='kpi-val'>{n_stores}</div>
      <div class='kpi-lbl'>Stores</div>
      <div class='kpi-sub'>3 tiers · 5 regions</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='kpi amber'>
      <div class='kpi-val'>{total_rec/1000:.0f}K+</div>
      <div class='kpi-lbl'>Records</div>
      <div class='kpi-sub'>{date_range}</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='kpi' style='border-top-color:#a78bfa'>
      <div class='kpi-val' style='color:#a78bfa'>3</div>
      <div class='kpi-lbl'>Models</div>
      <div class='kpi-sub'>HW · SARIMA · Prophet</div>
    </div>""", unsafe_allow_html=True)
with k5:
    bm = best_store["mape"]
    st.markdown(f"""<div class='kpi' style='border-top-color:#00e5ff'>
      <div class='kpi-val'>{bm:.1f}%</div>
      <div class='kpi-lbl'>Best Store</div>
      <div class='kpi-sub'>{best_store["store_id"]}</div>
    </div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Forecast View",
    "🏪  Store Analysis",
    "📦  Category Trends",
    "🔬  Model Comparison",
])
 
PLOTLY_BASE = dict(
    paper_bgcolor="#0a0d14",
    plot_bgcolor="#161c2d",
    font=dict(family="DM Sans", color="#e8eaf0"),
    margin=dict(t=40, b=30, l=60, r=20),
)
 
# ══════════════════════════════════════════════
# TAB 1 — Forecast View
# ══════════════════════════════════════════════
with tab1:
    sf = forecasts_df[forecasts_df["store_id"] == sel_store].copy()
    sr = results_df[results_df["store_id"] == sel_store].iloc[0]
 
    # Store header
    c_head, c_kpi = st.columns([3,1])
    with c_head:
        bm_name = sr["best_model"]
        bcls    = {"Holt-Winters":"b-hw","SARIMA":"b-sarima","Prophet":"b-prop"}.get(bm_name,"b-hw")
        st.markdown(f"""
        <h3 style='margin-bottom:4px'>{sel_store} — Forecast</h3>
        <span class='badge {bcls}'>Best Model: {bm_name}</span>
        <span class='badge b-mape'>MAPE: {sr["mape"]:.1f}%</span>
        """, unsafe_allow_html=True)
    with c_kpi:
        mape_color = "#00d68f" if sr["mape"] < 8 else ("#ffd60a" if sr["mape"] < 12 else "#ff6b6b")
        st.markdown(f"""
        <div class='kpi' style='border-top-color:{mape_color}; margin-top:8px'>
          <div class='kpi-val' style='color:{mape_color}'>{sr["mape"]:.1f}%</div>
          <div class='kpi-lbl'>Store MAPE</div>
        </div>""", unsafe_allow_html=True)
 
    # Data slices
    hist = sf[sf["type"]=="history"].sort_values("ds")
    act  = sf[sf["type"]=="actual"].sort_values("ds")
    pred = sf[sf["type"]=="predicted"].sort_values("ds")
    fore = sf[sf["type"]=="forecast"].sort_values("ds")
 
    # Main forecast chart
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["value"], name="Historical",
        line=dict(color="#3d4a6e", width=1.5), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=act["ds"], y=act["value"], name="Actual (Test)",
        line=dict(color="#e8eaf0", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=pred["ds"], y=pred["value"], name="Predicted",
        line=dict(color="#ff6b6b", width=2.5, dash="dash"),
    ))
    if len(fore):
        fig.add_trace(go.Scatter(
            x=fore["ds"], y=fore["value"], name="6-Month Forecast",
            line=dict(color="#00d68f", width=2.5, dash="dot"),
            fill="tozeroy", fillcolor="rgba(0,214,143,0.04)",
        ))
 
    if len(act):
        fig.add_vrect(
            x0=act["ds"].min(), x1=act["ds"].max(),
            fillcolor="#ffd60a", opacity=0.07, layer="below", line_width=0,
            annotation_text="Test Period", annotation_position="top left",
            annotation_font_size=10, annotation_font_color="#ffd60a",
        )
 
    fig.update_layout(
        **PLOTLY_BASE, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e2640", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1e2640", showgrid=True, zeroline=False, title="Monthly Sales (₹)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # Residuals
    if len(act) and len(pred):
        res = act["value"].values - pred["value"].values
        pct_err = (res / act["value"].values) * 100
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=act["ds"], y=pct_err,
            marker_color=["#ff6b6b" if v < 0 else "#00d68f" for v in pct_err],
            name="% Error",
        ))
        fig_r.add_hline(y=0, line_color="#6b7280", line_width=1)
        fig_r.update_layout(
            **PLOTLY_BASE, height=180,
            title=dict(text="Residual % Error (Test Period)", font=dict(size=13)),
            xaxis=dict(gridcolor="#1e2640", zeroline=False),
            yaxis=dict(gridcolor="#1e2640", zeroline=False, title="Error %"),
        )
        st.plotly_chart(fig_r, use_container_width=True)
 
    # Forecast table
    if len(fore):
        st.markdown("**6-Month Forward Forecast**")
        ft = fore[["ds","value"]].copy()
        ft.columns = ["Month","Forecasted Sales (₹)"]
        ft["Month"] = ft["Month"].dt.strftime("%b %Y")
        ft["Forecasted Sales (₹)"] = ft["Forecasted Sales (₹)"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(ft.reset_index(drop=True), use_container_width=True)
 
# ══════════════════════════════════════════════
# TAB 2 — Store Analysis
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<h3>Store-Level Performance</h3>", unsafe_allow_html=True)
 
    c1, c2 = st.columns(2)
 
    with c1:
        # MAPE histogram
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=results_df["mape"], nbinsx=20,
            marker_color="#00e5ff", opacity=0.8, name="MAPE",
        ))
        fig_h.add_vline(
            x=results_df["mape"].mean(), line_dash="dash",
            line_color="#00d68f",
            annotation_text=f"Mean {results_df['mape'].mean():.1f}%",
            annotation_font_color="#00d68f", annotation_position="top right",
        )
        fig_h.update_layout(
            **PLOTLY_BASE, height=300,
            title=dict(text="MAPE Distribution Across Stores", font=dict(size=13)),
            xaxis=dict(gridcolor="#1e2640", title="MAPE (%)"),
            yaxis=dict(gridcolor="#1e2640", title="Store Count"),
        )
        st.plotly_chart(fig_h, use_container_width=True)
 
    with c2:
        # Model wins donut
        mc = results_df["best_model"].value_counts()
        fig_d = go.Figure(go.Pie(
            labels=mc.index, values=mc.values, hole=0.55,
            marker_colors=["#ffd60a","#00e5ff","#00d68f"],
            textinfo="label+percent",
            textfont=dict(family="Space Mono", size=11),
        ))
        fig_d.update_layout(
            **PLOTLY_BASE, height=300,
            title=dict(text="Best Model Per Store", font=dict(size=13)),
            showlegend=False,
        )
        st.plotly_chart(fig_d, use_container_width=True)
 
    # Top 10 table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Top 10 — Lowest MAPE**")
    top10 = results_df.nsmallest(10, "mape")[
        ["store_id","best_model","mape","hw_mape","sarima_mape","prophet_mape"]
    ].reset_index(drop=True)
    top10.columns = ["Store","Best Model","MAPE %","HW MAPE","SARIMA MAPE","Prophet MAPE"]
    st.dataframe(top10, use_container_width=True)
 
# ══════════════════════════════════════════════
# TAB 3 — Category Trends
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<h3>Category-Level Sales Trends</h3>", unsafe_allow_html=True)
 
    filt = monthly_df.copy()
    if sel_cat != "All":
        filt = filt[filt["category"] == sel_cat]
 
    # Monthly trend by category
    cat_monthly = (
        filt.groupby(["category", pd.Grouper(key="date", freq="MS")])["sales"]
        .sum().reset_index()
    )
    fig_c = px.line(
        cat_monthly, x="date", y="sales", color="category",
        color_discrete_sequence=["#00e5ff","#00d68f","#ffd60a","#ff6b6b","#a78bfa"],
    )
    fig_c.update_layout(
        **PLOTLY_BASE, height=360,
        title=dict(text="Monthly Sales by Category", font=dict(size=13)),
        xaxis=dict(gridcolor="#1e2640", zeroline=False),
        yaxis=dict(gridcolor="#1e2640", zeroline=False, title="Monthly Sales (₹)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_c, use_container_width=True)
 
    # Category share + Weekday pattern
    cc1, cc2 = st.columns(2)
    with cc1:
        cs = filt.groupby("category")["sales"].sum().sort_values().reset_index()
        fig_b = go.Figure(go.Bar(
            x=cs["sales"], y=cs["category"], orientation="h",
            marker_color=["#00e5ff","#00d68f","#ffd60a","#ff6b6b","#a78bfa"],
            opacity=0.85,
        ))
        fig_b.update_layout(
            **PLOTLY_BASE, height=280,
            title=dict(text="Total Sales by Category", font=dict(size=13)),
            xaxis=dict(gridcolor="#1e2640", title="₹"),
            yaxis=dict(gridcolor="#1e2640"),
        )
        st.plotly_chart(fig_b, use_container_width=True)
 
    with cc2:
        daily_sub = sales_df.copy()
        if sel_cat != "All":
            daily_sub = daily_sub[daily_sub["category"] == sel_cat]
        daily_sub["weekday"] = daily_sub["date"].dt.day_name()
        wdo = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wd_avg = daily_sub.groupby("weekday")["sales"].mean().reindex(wdo).reset_index()
        colors_wd = ["#3d4a6e"]*5 + ["#00e5ff","#00d68f"]
        fig_w = go.Figure(go.Bar(
            x=wd_avg["weekday"], y=wd_avg["sales"],
            marker_color=colors_wd, opacity=0.9,
        ))
        fig_w.update_layout(
            **PLOTLY_BASE, height=280,
            title=dict(text="Avg Sales — Day of Week", font=dict(size=13)),
            xaxis=dict(gridcolor="#1e2640"),
            yaxis=dict(gridcolor="#1e2640", title="Avg Daily Sales (₹)"),
        )
        st.plotly_chart(fig_w, use_container_width=True)
 
# ══════════════════════════════════════════════
# TAB 4 — Model Comparison
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<h3>Model Benchmarking</h3>", unsafe_allow_html=True)
 
    # Box plot
    import plotly.figure_factory as ff
    melt = results_df[["hw_mape","sarima_mape","prophet_mape"]].copy()
    melt.columns = ["Holt-Winters","SARIMA","Prophet"]
    melt_long = melt.melt(var_name="Model", value_name="MAPE").dropna()
 
    fig_box = go.Figure()
    palette = {
        "Holt-Winters": {"line": "#ffd60a", "fill": "rgba(255,214,10,0.15)"},
        "SARIMA":        {"line": "#00e5ff", "fill": "rgba(0,229,255,0.15)"},
        "Prophet":       {"line": "#00d68f", "fill": "rgba(0,214,143,0.15)"},
    }
    for mn, cols in palette.items():
        sub = melt_long[melt_long["Model"]==mn]["MAPE"]
        fig_box.add_trace(go.Box(
            y=sub, name=mn,
            marker_color=cols["line"],
            line_color=cols["line"],
            fillcolor=cols["fill"],
            boxmean=True,
        ))
    fig_box.update_layout(
        **PLOTLY_BASE, height=360,
        title=dict(text="MAPE Distribution per Model (All 52 Stores)", font=dict(size=13)),
        yaxis=dict(gridcolor="#1e2640", title="MAPE (%)"),
        xaxis=dict(gridcolor="#1e2640"),
    )
    st.plotly_chart(fig_box, use_container_width=True)
 
    # Summary table
    summ = melt_long.groupby("Model")["MAPE"].agg(["mean","median","min","max","std"]).round(2)
    summ.columns = ["Mean %","Median %","Min %","Max %","Std Dev"]
    st.dataframe(summ.reset_index(), use_container_width=True)
 
    # Insight cards
    st.markdown("<br>", unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown("""
        <div class='insight'>
          <b>Holt-Winters</b><br><br>
          <span style='color:#9ca3af; font-size:0.82rem'>
          Best for stores with stable, smooth monthly seasonality.
          Fast training. Strong on Tier-1 high-volume stores with
          consistent Dec/Oct peaks.
          </span>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""
        <div class='insight' style='border-left-color:#00e5ff'>
          <b style='color:#00e5ff'>SARIMA(1,1,1)(1,1,0)[12]</b><br><br>
          <span style='color:#9ca3af; font-size:0.82rem'>
          Best for stores with complex autocorrelation structures.
          Handles irregular spikes. Strongest on stores with
          stationary yearly seasonality.
          </span>
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown("""
        <div class='insight' style='border-left-color:#00d68f'>
          <b style='color:#00d68f'>Prophet</b><br><br>
          <span style='color:#9ca3af; font-size:0.82rem'>
          Best for stores with holiday effects and trend changepoints.
          Robust to missing data. Automatically detects festival
          season shifts (Diwali, Dec peaks).
          </span>
        </div>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:14px; color:#3d4a6e;
     font-size:0.75rem; border-top:1px solid #1e2640;
     font-family:Space Mono'>
  RETAIL SALES FORECASTING SYSTEM &nbsp;·&nbsp;
  BUILT BY RENUKA BHARDWAJ &nbsp;·&nbsp;
  HW + SARIMA + PROPHET &nbsp;·&nbsp; MAPE ~8% &nbsp;·&nbsp;
  100K+ RECORDS · 52 STORES
</div>
""", unsafe_allow_html=True)