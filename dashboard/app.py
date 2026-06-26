"""
=============================================================================
  Malaysia Labour Market Analytics Dashboard
  PMA MRTA2173 Assignment — Q4 & Q5
  Dataset : data/raw/lfs_month.csv  (monthly, 2010-01 to 2026-04)
=============================================================================
Run with:  streamlit run dashboard/app.py
"""

import os, sys, warnings, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0 ▸ Paths  (work from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "lfs_month.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "dashboard_model.joblib")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "models", "dashboard_scaler.joblib")

# ---------------------------------------------------------------------------
# 1 ▸ Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Labour Market Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark gradient header banner */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { color: white; margin-bottom: 0.2rem; font-size: 2rem; }
    .main-header p  { color: #a8b2d1; margin-top: 0; font-size: 0.95rem; }
    /* Metric card tweaks */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color); border-radius: 10px; padding: 12px 16px;
        border-left: 4px solid var(--primary-color);
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Malaysia Labour Market Analytics</h1>
    <p>Interactive Dashboard &amp; Predictive Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2 ▸ Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df

df = load_data()

NUMERIC_COLS = ["lf", "lf_employed", "lf_unemployed", "lf_outside",
                "p_rate", "ep_ratio", "u_rate"]
NICE_NAMES = {
    "lf": "Labour Force ('000)",
    "lf_employed": "Employed ('000)",
    "lf_unemployed": "Unemployed ('000)",
    "lf_outside": "Outside LF ('000)",
    "p_rate": "Participation Rate (%)",
    "ep_ratio": "Emp-Pop Ratio (%)",
    "u_rate": "Unemployment Rate (%)",
}

# ---------------------------------------------------------------------------
# 3 ▸ Sidebar — Interactive Controls  [Q4(a)(ii)]
# ---------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Coat_of_arms_of_Malaysia.svg/180px-Coat_of_arms_of_Malaysia.svg.png", width=90)
st.sidebar.title("🎛️ Controls")

# ── Date range selector ──────────────────────────────────────────────────────
st.sidebar.subheader("📅 Date Range Filter")
all_years = sorted(df["year"].unique())
start_year = st.sidebar.selectbox("Start Year", all_years, index=0)
end_year   = st.sidebar.selectbox("End Year",   all_years, index=len(all_years) - 1)
if start_year > end_year:
    st.sidebar.error("Start year must be ≤ End year")
    start_year, end_year = end_year, start_year

mask = (df["year"] >= start_year) & (df["year"] <= end_year)
df_filtered = df.loc[mask].copy()

# ── Variable selector ────────────────────────────────────────────────────────
st.sidebar.subheader("📌 Variable Comparison")
selected_vars = st.sidebar.multiselect(
    "Select variables for comparison chart",
    options=NUMERIC_COLS,
    default=["u_rate", "p_rate"],
    format_func=lambda x: NICE_NAMES.get(x, x),
)

# ── Forecast horizon slider (Q4(a)(iii)) ─────────────────────────────────────
st.sidebar.subheader("🔮 Forecast Settings")
forecast_months = st.sidebar.slider("Months to forecast", 1, 12, 6)

# ---------------------------------------------------------------------------
# 4 ▸ Model loading / inline training  [Q4(a)(iii)]
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model_and_scaler(_df):
    """Load saved model or train a simple Ridge regression inline."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import joblib

    model, scaler = None, None
    model_source = "inline"

    # Try loading pre-trained artefacts
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            model_source = "saved"
            return model, scaler, model_source
    except Exception:
        pass

    # ── Inline training ──────────────────────────────────────────────────────
    # Features: month-sin/cos, trend index, lag-1, lag-3, lag-12
    tmp = _df[["date", "u_rate"]].dropna().sort_values("date").reset_index(drop=True)
    tmp["trend"]    = np.arange(len(tmp))
    tmp["m_sin"]    = np.sin(2 * np.pi * tmp["date"].dt.month / 12)
    tmp["m_cos"]    = np.cos(2 * np.pi * tmp["date"].dt.month / 12)
    tmp["lag_1"]    = tmp["u_rate"].shift(1)
    tmp["lag_3"]    = tmp["u_rate"].shift(3)
    tmp["lag_12"]   = tmp["u_rate"].shift(12)
    tmp = tmp.dropna()

    feat_cols = ["trend", "m_sin", "m_cos", "lag_1", "lag_3", "lag_12"]
    X = tmp[feat_cols].values
    y = tmp["u_rate"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_sc, y)

    # Persist for future use
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    model_source = "inline (just trained & saved)"
    return model, scaler, model_source

model, scaler, model_source = get_model_and_scaler(df)

# ---------------------------------------------------------------------------
# Helper: generate N-month forecast
# ---------------------------------------------------------------------------
def forecast_u_rate(df_full, model, scaler, n_months):
    """Iteratively forecast n_months ahead."""
    tmp = df_full[["date", "u_rate"]].dropna().sort_values("date").reset_index(drop=True)
    trend_start = len(tmp)
    history = tmp["u_rate"].tolist()
    last_date = tmp["date"].iloc[-1]

    preds, dates = [], []
    for i in range(n_months):
        t   = trend_start + i
        mon = (last_date.month + i) % 12 + 1
        m_sin = np.sin(2 * np.pi * mon / 12)
        m_cos = np.cos(2 * np.pi * mon / 12)
        lag1  = history[-1]
        lag3  = history[-3] if len(history) >= 3 else history[-1]
        lag12 = history[-12] if len(history) >= 12 else history[-1]

        x = np.array([[t, m_sin, m_cos, lag1, lag3, lag12]])
        x_sc = scaler.transform(x)
        yhat = float(model.predict(x_sc)[0])
        yhat = max(0.0, round(yhat, 2))
        history.append(yhat)

        next_date = last_date + pd.DateOffset(months=i + 1)
        preds.append(yhat)
        dates.append(next_date)

    return pd.DataFrame({"date": dates, "u_rate_forecast": preds})

df_forecast = forecast_u_rate(df, model, scaler, forecast_months)

# ---------------------------------------------------------------------------
# 5 ▸ Key metrics row
# ---------------------------------------------------------------------------
latest = df_filtered.iloc[-1]
earliest = df_filtered.iloc[0]
avg_urate = df_filtered["u_rate"].mean()
delta_urate = round(latest["u_rate"] - df_filtered.iloc[-2]["u_rate"], 2) if len(df_filtered) > 1 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Unemployment Rate", f"{latest['u_rate']:.1f}%",
          delta=f"{delta_urate:+.1f}pp" if delta_urate else None,
          delta_color="inverse")
c2.metric("Avg Unemployment (Period)", f"{avg_urate:.2f}%")
c3.metric("Labour Force (Latest)", f"{latest['lf']:,.1f}k")
c4.metric("Observations", f"{len(df_filtered)}")

# ---------------------------------------------------------------------------
# 6 ▸ Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_pred, tab_mon = st.tabs(["📈 Dashboard", "🔮 Predictions", "🩺 Monitoring"])

# =====================  TAB 1 — Dashboard  [Q4(a)(i) + Q4(a)(ii)]  =========
with tab_dash:
    st.subheader("📈 Core Labour Market Trends")

    # ── Chart 1: Unemployment rate trend ─────────────────────────────────────
    fig1 = px.line(
        df_filtered, x="date", y="u_rate",
        title="Malaysia Monthly Unemployment Rate Trend",
        labels={"date": "Date", "u_rate": "Unemployment Rate (%)"},
        hover_data={"date": "|%B %Y", "u_rate": ":.2f"},
    )
    fig1.update_traces(line=dict(color="#e63946", width=2))
    fig1.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date", yaxis_title="Unemployment Rate (%)",
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True, key="trend_chart")

    col_a, col_b = st.columns(2)

    # ── Chart 2: Average annual unemployment rate ────────────────────────────
    with col_a:
        annual = (df_filtered.groupby("year")["u_rate"]
                  .mean().reset_index()
                  .rename(columns={"u_rate": "avg_u_rate"}))
        annual["avg_u_rate"] = annual["avg_u_rate"].round(2)

        fig2 = px.bar(
            annual, x="year", y="avg_u_rate",
            title="Average Annual Unemployment Rate",
            labels={"year": "Year", "avg_u_rate": "Avg Unemployment Rate (%)"},
            text="avg_u_rate",
            color="avg_u_rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig2.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
        fig2.update_layout(
            template="plotly_white", height=420,
            xaxis=dict(dtick=1), coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True, key="bar_chart")

    # ── Chart 3: Correlation heatmap ─────────────────────────────────────────
    with col_b:
        corr = df_filtered[NUMERIC_COLS].corr().round(2)

        fig3 = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=[NICE_NAMES.get(c, c) for c in corr.columns],
            y=[NICE_NAMES.get(c, c) for c in corr.index],
            text=corr.values,
            texttemplate="%{text:.2f}",
            colorscale="RdBu_r", zmid=0,
            hovertemplate="%{x}<br>%{y}<br>Corr: %{z:.2f}<extra></extra>",
        ))
        fig3.update_layout(
            title="Correlation Matrix of Labour Market Indicators",
            template="plotly_white", height=420,
            xaxis_tickangle=-40,
        )
        st.plotly_chart(fig3, use_container_width=True, key="corr_chart")

    # ── Interactive comparison chart [Q4(a)(ii)] ─────────────────────────────
    st.subheader("🔄 Multi-Variable Interactive Analysis")
    if selected_vars:
        fig4 = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, v in enumerate(selected_vars):
            fig4.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered[v],
                mode="lines", name=NICE_NAMES.get(v, v),
                line=dict(color=palette[i % len(palette)], width=2),
                hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
            ))
        fig4.update_layout(
            title="Multi-Variable Comparison",
            template="plotly_white", height=400,
            xaxis_title="Date", yaxis_title="Value",
            hovermode="x unified", legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig4, use_container_width=True, key="comparison_chart")
    else:
        st.info("Select one or more variables in the sidebar to display the comparison chart.")

# =====================  TAB 2 — Predictions  [Q4(a)(iii)]  =================
with tab_pred:
    st.subheader("🔮 Unemployment Rate Forecasting")
    st.caption(f"Model source: **{model_source}** · Forecasting **{forecast_months}** month(s) ahead")

    # ── Forecast table ───────────────────────────────────────────────────────
    col_tbl, col_info = st.columns([3, 2])
    with col_tbl:
        st.markdown("##### Predicted Unemployment Rates")
        display_df = df_forecast.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
        display_df.columns = ["Month", "Predicted u_rate (%)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_info:
        st.markdown("##### Forecast Summary")
        st.metric("Min Forecast", f"{df_forecast['u_rate_forecast'].min():.2f}%")
        st.metric("Max Forecast", f"{df_forecast['u_rate_forecast'].max():.2f}%")
        st.metric("Mean Forecast", f"{df_forecast['u_rate_forecast'].mean():.2f}%")

    # ── Trend chart + forecast overlay ───────────────────────────────────────
    st.markdown("##### Trend with Forecast Extension")
    fig5 = go.Figure()

    # Historical
    fig5.add_trace(go.Scatter(
        x=df_filtered["date"], y=df_filtered["u_rate"],
        mode="lines", name="Historical",
        line=dict(color="#457b9d", width=2),
    ))

    # Forecast (connect from last observed point)
    bridge_date  = [df["date"].iloc[-1]] + df_forecast["date"].tolist()
    bridge_value = [df["u_rate"].iloc[-1]] + df_forecast["u_rate_forecast"].tolist()

    fig5.add_trace(go.Scatter(
        x=bridge_date, y=bridge_value,
        mode="lines+markers", name="Forecast",
        line=dict(color="#e76f51", width=2.5, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
    ))

    fig5.add_vline(x=df["date"].iloc[-1], line_dash="dash",
                   line_color="grey", annotation_text="Forecast start")
    fig5.update_layout(
        title="Unemployment Rate — Historical + Forecast",
        template="plotly_white", height=450,
        xaxis_title="Date", yaxis_title="Unemployment Rate (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig5, use_container_width=True, key="forecast_chart")

# =====================  TAB 3 — Monitoring  [Q5(a) + Q5(b)]  ===============
with tab_mon:
    st.subheader("🩺 Model & Pipeline Quality Monitoring")

    # ── Metric 1: RMSE on last 12 months ─────────────────────────────────────
    from sklearn.metrics import mean_squared_error

    eval_df = df.tail(12).copy().reset_index(drop=True)
    # Re-create features for evaluation rows
    full_sorted = df[["date", "u_rate"]].dropna().sort_values("date").reset_index(drop=True)
    full_sorted["trend"]  = np.arange(len(full_sorted))
    full_sorted["m_sin"]  = np.sin(2 * np.pi * full_sorted["date"].dt.month / 12)
    full_sorted["m_cos"]  = np.cos(2 * np.pi * full_sorted["date"].dt.month / 12)
    full_sorted["lag_1"]  = full_sorted["u_rate"].shift(1)
    full_sorted["lag_3"]  = full_sorted["u_rate"].shift(3)
    full_sorted["lag_12"] = full_sorted["u_rate"].shift(12)
    eval_rows = full_sorted.dropna().tail(12)

    feat_cols = ["trend", "m_sin", "m_cos", "lag_1", "lag_3", "lag_12"]
    X_eval = eval_rows[feat_cols].values
    y_eval = eval_rows["u_rate"].values
    y_pred_eval = model.predict(scaler.transform(X_eval))

    rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))

    # ── Metric 2: Data freshness ─────────────────────────────────────────────
    latest_date = df["date"].max()
    days_since  = (pd.Timestamp.now() - latest_date).days

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Model RMSE (last 12 months)", f"{rmse:.4f}",
               help="Root-Mean-Square Error of model on the most recent 12 data points")
    mc2.metric("Latest Data Point", latest_date.strftime("%Y-%m-%d"),
               delta=f"-{days_since} days ago", delta_color="off")
    mc3.metric("Model Type", model_source.split("(")[0].strip().title())

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # Q5(b) — Data Drift Analysis
    # ────────────────────────────────────────────────────────────────────────
    st.subheader("📡 Statistical Data Drift Detection")

    split_idx   = int(len(df) * 0.8)
    train_part  = df.iloc[:split_idx]
    recent_part = df.iloc[split_idx:]

    train_u  = train_part["u_rate"].dropna()
    recent_u = recent_part["u_rate"].dropna()

    # Statistics
    mean_shift = recent_u.mean() - train_u.mean()
    std_shift  = recent_u.std()  - train_u.std()

    # KS test (optional)
    ks_stat, ks_p = None, None
    try:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(train_u, recent_u)
    except ImportError:
        pass

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Mean Shift", f"{mean_shift:+.3f}",
               help="Change in mean unemployment rate (recent − training)")
    dc2.metric("Std Shift", f"{std_shift:+.3f}",
               help="Change in standard deviation")
    if ks_stat is not None:
        dc3.metric("KS Statistic", f"{ks_stat:.4f}",
                   delta=f"p = {ks_p:.4f}",
                   delta_color="off",
                   help="Kolmogorov–Smirnov two-sample test statistic")
    else:
        dc3.metric("KS Test", "scipy not available")

    # ── Side-by-side histograms ──────────────────────────────────────────────
    fig6 = make_subplots(rows=1, cols=2,
                         subplot_titles=[
                             f"Training Period ({train_part['date'].min().year}–{train_part['date'].max().year})",
                             f"Recent Period ({recent_part['date'].min().year}–{recent_part['date'].max().year})",
                         ])
    fig6.add_trace(go.Histogram(x=train_u, nbinsx=20, name="Training",
                                marker_color="#457b9d", opacity=0.85), row=1, col=1)
    fig6.add_trace(go.Histogram(x=recent_u, nbinsx=20, name="Recent",
                                marker_color="#e76f51", opacity=0.85), row=1, col=2)
    fig6.update_layout(
        title="Distribution of Unemployment Rate — Training vs Recent",
        template="plotly_white", height=380, showlegend=True,
        legend=dict(orientation="h", y=-0.18),
    )
    fig6.update_xaxes(title_text="Unemployment Rate (%)", row=1, col=1)
    fig6.update_xaxes(title_text="Unemployment Rate (%)", row=1, col=2)
    fig6.update_yaxes(title_text="Count", row=1, col=1)
    fig6.update_yaxes(title_text="Count", row=1, col=2)
    st.plotly_chart(fig6, use_container_width=True, key="drift_hist")

    # ── Interpretation ───────────────────────────────────────────────────────
    st.markdown("##### 📝 Drift Interpretation")
    interpretation_parts = []

    if abs(mean_shift) < 0.2:
        interpretation_parts.append(
            "The **mean unemployment rate** in the recent period is very close to the training period "
            f"(shift of {mean_shift:+.3f}pp), indicating **no significant mean drift**."
        )
    elif mean_shift < 0:
        interpretation_parts.append(
            f"The **mean unemployment rate has decreased** by {abs(mean_shift):.3f}pp in the recent period, "
            "suggesting an improving labour market that **differs from training data patterns**."
        )
    else:
        interpretation_parts.append(
            f"The **mean unemployment rate has increased** by {mean_shift:.3f}pp in the recent period, "
            "signalling potential economic stress and a **distribution shift**."
        )

    if abs(std_shift) < 0.1:
        interpretation_parts.append(
            "The **variability** remains similar (std shift of "
            f"{std_shift:+.3f}), so volatility is largely unchanged."
        )
    else:
        direction = "increased" if std_shift > 0 else "decreased"
        interpretation_parts.append(
            f"The **variability has {direction}** (std shift {std_shift:+.3f}), "
            "indicating a change in data volatility that may affect model reliability."
        )

    if ks_stat is not None:
        if ks_p < 0.05:
            interpretation_parts.append(
                f"The **KS test** (statistic = {ks_stat:.4f}, p = {ks_p:.4f}) "
                "**rejects** the null hypothesis at the 5 % level — the two distributions are "
                "**statistically different**, confirming data drift."
            )
        else:
            interpretation_parts.append(
                f"The **KS test** (statistic = {ks_stat:.4f}, p = {ks_p:.4f}) "
                "**does not reject** the null hypothesis — there is **no statistically significant "
                "drift** detected at the 5 % level."
            )

    interpretation_parts.append(
        "\n**Recommendation**: "
        + ("Model retraining is **recommended** if the KS test or mean shift indicates drift, "
           "as predictions may degrade on out-of-distribution data."
           if (ks_p is not None and ks_p < 0.05) or abs(mean_shift) >= 0.3
           else "Current drift levels are acceptable. Continue monitoring and retrain periodically.")
    )

    st.info("\n\n".join(interpretation_parts))

    # ── Summary statistics table ─────────────────────────────────────────────
    st.markdown("##### Summary Statistics Comparison")
    summary = pd.DataFrame({
        "Statistic": ["Count", "Mean", "Std Dev", "Min", "25th Pctl", "Median", "75th Pctl", "Max"],
        "Training Period": [
            f"{len(train_u)}", f"{train_u.mean():.3f}", f"{train_u.std():.3f}",
            f"{train_u.min():.1f}", f"{train_u.quantile(0.25):.2f}",
            f"{train_u.median():.2f}", f"{train_u.quantile(0.75):.2f}", f"{train_u.max():.1f}",
        ],
        "Recent Period": [
            f"{len(recent_u)}", f"{recent_u.mean():.3f}", f"{recent_u.std():.3f}",
            f"{recent_u.min():.1f}", f"{recent_u.quantile(0.25):.2f}",
            f"{recent_u.median():.2f}", f"{recent_u.quantile(0.75):.2f}", f"{recent_u.max():.1f}",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:0.85rem;'>"
    "PMA MRTA2173 · Malaysia Labour Market Analytics Dashboard · "
    f"Data last updated: {latest_date.strftime('%B %Y')}"
    "</p>",
    unsafe_allow_html=True,
)
