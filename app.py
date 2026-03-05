import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InsightX – Data Analytics Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  – premium dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0a0e1a; color: #e0e6f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #1a2340 100%);
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2340 0%, #1e2d4a 100%);
    border: 1px solid #2a3d5e;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #7a9fc0 !important; font-size: 0.82rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc);
    color: #0a0e1a;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 28px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #33ddff, #00aadd);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,212,255,0.35);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: #1a2340;
    border-radius: 8px 8px 0 0;
    color: #7a9fc0;
    font-weight: 600;
    padding: 10px 22px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4ff22, #00d4ff11) !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Headings */
h1 { color: #00d4ff !important; font-weight: 700 !important; }
h2, h3 { color: #b0d0f0 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a3d5e;
    border-radius: 14px;
    padding: 20px;
    background: #111827;
}

/* Info/success/error boxes */
.stAlert { border-radius: 10px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Divider */
hr { border-color: #1e2d4a !important; }

/* Insight card */
.insight-card {
    background: linear-gradient(135deg, #0f1c35, #152038);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 14px 0;
    font-size: 0.93rem;
    line-height: 1.7;
}
.insight-card b { color: #00d4ff; }

/* Insight warning card */
.insight-warn {
    background: linear-gradient(135deg, #1f1200, #2a1800);
    border: 1px solid #5a3200;
    border-left: 4px solid #ffaa00;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.91rem;
}
.insight-warn b { color: #ffaa00; }

/* Insight success card */
.insight-ok {
    background: linear-gradient(135deg, #001f18, #002a20);
    border: 1px solid #004d32;
    border-left: 4px solid #00e87a;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.91rem;
}
.insight-ok b { color: #00e87a; }

/* Section label */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #4a7fa0;
    text-transform: uppercase;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  CACHED HELPERS
# ═════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Load CSV or Excel from raw bytes (cache-safe)."""
    import io
    buf = io.BytesIO(file_bytes)
    if file_name.endswith(".csv"):
        return pd.read_csv(buf)
    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return None


@st.cache_data(show_spinner=False)
def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T


@st.cache_data(show_spinner=False)
def get_correlation(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["float64", "int64", "int32", "float32"])
    if num.shape[1] > 1:
        return num.corr()
    return None


@st.cache_data(show_spinner=False)
def get_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


@st.cache_data(show_spinner=False)
def compute_skewness(df: pd.DataFrame) -> pd.Series:
    num = df.select_dtypes(include="number")
    return num.skew()


@st.cache_data(show_spinner=False)
def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """Return outlier counts per numeric column using IQR method."""
    result = {}
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out > 0:
            result[col] = n_out
    return result


# ─────────────────────────────────────────────
#  SMART COLUMN DETECTOR
# ─────────────────────────────────────────────
def smart_cat_cols(df: pd.DataFrame, max_unique: int = 30) -> list:
    """Categorical columns that are useful for charts (not IDs, not free-text)."""
    cols = []
    for col in df.select_dtypes(include="object").columns:
        n = df[col].nunique()
        if 1 < n <= max_unique:
            cols.append(col)
    return cols


def smart_num_cols(df: pd.DataFrame) -> list:
    """Numeric columns – exclude columns that look like pure IDs."""
    num = df.select_dtypes(include=["float64", "int64", "int32", "float32"]).columns.tolist()
    return [c for c in num if not (
        c.lower().endswith("id") or c.lower() == "id" or df[c].nunique() == len(df)
    )]


# ─────────────────────────────────────────────
#  CHART INSIGHT GENERATOR
# ─────────────────────────────────────────────
def chart_insight(df: pd.DataFrame, col: str, kind: str = "dist") -> str:
    """Generate a short text insight for a column."""
    lines = []
    if kind == "dist" and col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        mean, median = s.mean(), s.median()
        skew = s.skew()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        outliers = int(((s < q1 - 1.5*(q3-q1)) | (s > q3 + 1.5*(q3-q1))).sum())
        lines.append(f"**Mean:** {mean:,.2f} &nbsp;|&nbsp; **Median:** {median:,.2f} &nbsp;|&nbsp; **Std:** {s.std():,.2f}")
        if abs(skew) < 0.5:
            lines.append(f"Distribution is **approximately symmetric** (skewness = {skew:.2f}).")
        elif skew > 0.5:
            lines.append(f"Distribution is **right-skewed** (skewness = {skew:.2f}) — most values are clustered at the lower end with a long tail on the right.")
        else:
            lines.append(f"Distribution is **left-skewed** (skewness = {skew:.2f}) — most values are clustered at the higher end with a long tail on the left.")
        if outliers > 0:
            pct = outliers / len(s) * 100
            lines.append(f"⚠️ **{outliers} outliers** detected ({pct:.1f}% of values) using the IQR method.")
        else:
            lines.append("✅ No significant outliers detected.")
    elif kind == "cat" and col in df.select_dtypes(include="object").columns:
        vc = df[col].value_counts()
        top, top_pct = vc.index[0], vc.iloc[0] / len(df) * 100
        lines.append(f"**Most frequent value:** '{top}' ({top_pct:.1f}% of records)")
        lines.append(f"**Unique values:** {df[col].nunique()} distinct categories out of {len(df):,} rows.")
        if df[col].nunique() == 1:
            lines.append("⚠️ Only one unique value — this column has zero variance.")
    elif kind == "corr":
        lines.append(col)  # col = pre-generated text for correlation
    return " &nbsp;&nbsp; ".join(lines) if lines else ""


def scatter_insight(df: pd.DataFrame, x: str, y: str) -> str:
    """Generate insight for a scatter relationship."""
    try:
        clean = df[[x, y]].dropna()
        r, p = scipy_stats.pearsonr(clean[x], clean[y])
        strength = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
        direction = "positive" if r > 0 else "negative"
        pval_txt = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p ≥ 0.05)"
        return (f"**Pearson r = {r:.3f}** — {strength} {direction} correlation, "
                f"which is {pval_txt}. "
                + ("Higher values of one variable tend to go with higher values of the other." if r > 0.4
                   else ("Higher values of one variable tend to go with lower values of the other." if r < -0.4
                         else "No strong linear relationship detected.")))
    except Exception:
        return "Could not compute correlation — ensure both columns are numeric."


def top_correlations(corr: pd.DataFrame, n: int = 5) -> list:
    """Return top n correlated pairs as list of (col1, col2, r)."""
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]


# ═════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔮 InsightX")
    st.caption("Automated Data Analytics Platform")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        ["📂 Data Upload", "📑 Data Insights", "📊 Dashboard", "🔬 Advanced Analysis"],
        label_visibility="collapsed",
    )

# Session state
if "data" not in st.session_state:
    st.session_state["data"] = None

# Global filters (sidebar) – only when data exists on analysis pages
filtered_df = st.session_state["data"]

if st.session_state["data"] is not None and page in ["📊 Dashboard", "🔬 Advanced Analysis", "📑 Data Insights"]:
    df_raw = st.session_state["data"]
    filt_cats = smart_cat_cols(df_raw, max_unique=20)

    if filt_cats:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🔍 Global Filters")
            filters = {}
            for col in filt_cats:
                unique_vals = sorted(df_raw[col].dropna().unique().tolist())
                sel = st.multiselect(col, unique_vals, default=unique_vals, key=f"filter_{col}")
                filters[col] = sel

        filtered_df = df_raw.copy()
        for col, vals in filters.items():
            if vals:
                filtered_df = filtered_df[filtered_df[col].isin(vals)]
    else:
        filtered_df = df_raw


# ═════════════════════════════════════════════
#  PLOT THEME HELPER
# ═════════════════════════════════════════════
DARK_LAYOUT = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0a0e1a",
    font_color="#e0e6f0",
)


def theme(fig, **extra):
    fig.update_layout(**DARK_LAYOUT, **extra)
    return fig


def ic(html: str, kind: str = "info"):
    """Render an insight card."""
    cls = {"info": "insight-card", "warn": "insight-warn", "ok": "insight-ok"}.get(kind, "insight-card")
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  PAGE: DATA UPLOAD
# ═════════════════════════════════════════════
if page == "📂 Data Upload":
    st.title("📂 Data Upload & Overview")
    st.caption("Upload your CSV or Excel file to get started. InsightX will auto-analyse your data.")

    uploaded = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "xlsx", "xls"],
        help="Supported: CSV, Excel (.xlsx / .xls)"
    )

    if uploaded:
        with st.spinner("Loading data…"):
            df = load_data(uploaded.name, uploaded.getvalue())
        st.session_state["data"] = df
        st.success(f"✅ **{uploaded.name}** loaded successfully!")
        st.rerun()

    if st.session_state["data"] is not None:
        df = st.session_state["data"]

        # ── KPI row ──────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📋 Total Rows",       f"{df.shape[0]:,}")
        c2.metric("🗂️ Total Columns",   f"{df.shape[1]:,}")
        c3.metric("🔢 Numeric Cols",     f"{len(df.select_dtypes(include='number').columns)}")
        c4.metric("🔁 Duplicates",       f"{get_duplicates(df):,}")
        c5.metric("❓ Missing Values",   f"{df.isnull().sum().sum():,}")

        # ── Quick dataset health insight ──────────
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        dup_pct = get_duplicates(df) / len(df) * 100 if len(df) > 0 else 0
        health_lines = []
        if missing_pct == 0:
            health_lines.append("✅ <b>No missing values</b> — dataset is complete.")
        elif missing_pct < 5:
            health_lines.append(f"🟡 <b>{missing_pct:.1f}% missing data</b> — minor gaps, consider imputation.")
        else:
            health_lines.append(f"⚠️ <b>{missing_pct:.1f}% missing data</b> — significant gaps may affect analysis quality.")
        if dup_pct == 0:
            health_lines.append("✅ <b>No duplicate rows</b> detected.")
        elif dup_pct < 3:
            health_lines.append(f"🟡 <b>{get_duplicates(df)} duplicate rows</b> ({dup_pct:.1f}%) — consider removing them.")
        else:
            health_lines.append(f"⚠️ <b>{get_duplicates(df)} duplicate rows</b> ({dup_pct:.1f}%) — high duplication rate, clean before analysis.")
        ic(" &nbsp; | &nbsp; ".join(health_lines))

        st.markdown("---")

        # ── Preview & Stats ───────────────────────
        tab_prev, tab_stat, tab_types, tab_missing = st.tabs(
            ["🗃️ Data Preview", "📊 Statistics", "🔣 Column Types", "❓ Missing Analysis"]
        )

        with tab_prev:
            st.dataframe(df.head(50), use_container_width=True, height=380)

        with tab_stat:
            stats = get_summary_stats(df)
            st.dataframe(stats.style.format(precision=2), use_container_width=True, height=380)

        with tab_types:
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Type":   df.dtypes.astype(str).values,
                "Non-Null": df.count().values,
                "Null":   df.isnull().sum().values,
                "Null %": (df.isnull().sum() / len(df) * 100).round(1).values,
                "Unique": df.nunique().values,
            })
            st.dataframe(dtype_df, use_container_width=True, height=380)

        with tab_missing:
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Column", "Missing Count"]
            miss["Missing %"] = (miss["Missing Count"] / len(df) * 100).round(2)
            miss = miss[miss["Missing Count"] > 0].sort_values("Missing %", ascending=False)
            if miss.empty:
                st.success("🎉 No missing values found in any column!")
            else:
                st.dataframe(miss, use_container_width=True)
                fig = px.bar(miss, x="Column", y="Missing %",
                             color="Missing %", color_continuous_scale="Reds",
                             template="plotly_dark", title="Missing Data by Column (%)")
                theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                ic(f"<b>{len(miss)} columns</b> have missing values. The most affected column is "
                   f"<b>{miss.iloc[0]['Column']}</b> with <b>{miss.iloc[0]['Missing %']:.1f}%</b> missing. "
                   f"Consider imputation or removal before further analysis.")
    else:
        st.info("👆 Upload a file above to get started with InsightX.")


# ═════════════════════════════════════════════
#  PAGE: DATA INSIGHTS
# ═════════════════════════════════════════════
elif page == "📑 Data Insights":
    st.title("📑 Data Insights")
    st.caption("Auto-generated insights from your dataset — skewness, outliers, correlations, and more.")

    if filtered_df is None:
        st.error("⚠️ Please upload data in **📂 Data Upload** first.")
        st.stop()

    df = filtered_df
    num_cols = smart_num_cols(df)
    cat_cols = smart_cat_cols(df)

    if not num_cols and not cat_cols:
        st.warning("No suitable columns found for insights.")
        st.stop()

    # ─────────────── 1. Dataset Overview Narrative ─────────────────
    st.markdown("### 🔎 Dataset Overview")
    total_cells = df.shape[0] * df.shape[1]
    missing_total = df.isnull().sum().sum()
    ic(
        f"Your dataset contains <b>{df.shape[0]:,} rows</b> and <b>{df.shape[1]} columns</b> "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical). "
        f"Total data cells: <b>{total_cells:,}</b>. "
        f"Missing values: <b>{missing_total:,}</b> ({missing_total/total_cells*100:.1f}%). "
        f"Duplicate rows: <b>{get_duplicates(df):,}</b>."
    )

    st.markdown("---")

    # ─────────────── 2. Skewness Analysis ─────────────────
    if num_cols:
        st.markdown("### 📐 Distribution Shape (Skewness)")
        skew = compute_skewness(df[num_cols])

        col_a, col_b = st.columns([1, 1])
        with col_a:
            skew_df = skew.reset_index()
            skew_df.columns = ["Column", "Skewness"]
            skew_df["Shape"] = skew_df["Skewness"].apply(
                lambda x: "Right-skewed ▶" if x > 0.5 else ("Left-skewed ◀" if x < -0.5 else "Symmetric ◆")
            )
            skew_df = skew_df.sort_values("Skewness", ascending=False)
            colors = skew_df["Skewness"].tolist()
            fig = px.bar(skew_df, x="Column", y="Skewness",
                         color="Skewness", color_continuous_scale="RdBu_r",
                         title="Skewness per Numeric Column",
                         template="plotly_dark", text="Skewness")
            fig.add_hline(y=0.5,  line_dash="dot", line_color="#ffaa00", annotation_text="Right-skew threshold")
            fig.add_hline(y=-0.5, line_dash="dot", line_color="#ffaa00", annotation_text="Left-skew threshold")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            theme(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            highly_right = skew_df[skew_df["Skewness"] > 0.5]["Column"].tolist()
            highly_left  = skew_df[skew_df["Skewness"] < -0.5]["Column"].tolist()
            symmetric    = skew_df[(skew_df["Skewness"] >= -0.5) & (skew_df["Skewness"] <= 0.5)]["Column"].tolist()
            st.markdown("#### 🔍 Skewness Breakdown")
            if highly_right:
                ic(f"<b>Right-skewed columns ({len(highly_right)}):</b> {', '.join(f'<code>{c}</code>' for c in highly_right)}."
                   " These columns have a long tail towards higher values. Log or square-root transformation may help normalise them.", "warn")
            if highly_left:
                ic(f"<b>Left-skewed columns ({len(highly_left)}):</b> {', '.join(f'<code>{c}</code>' for c in highly_left)}."
                   " These columns have a long tail towards lower values.", "warn")
            if symmetric:
                ic(f"<b>Approximately symmetric ({len(symmetric)}):</b> {', '.join(f'<code>{c}</code>' for c in symmetric)}."
                   " These columns have well-balanced distributions.", "ok")

        st.markdown("---")

    # ─────────────── 3. Outlier Report ─────────────────
    if num_cols:
        st.markdown("### 🚨 Outlier Detection (IQR Method)")
        outlier_dict = detect_outliers_iqr(df[num_cols])

        if not outlier_dict:
            ic("✅ <b>No significant outliers</b> detected in any numeric column using the IQR method.", "ok")
        else:
            out_df = pd.DataFrame({
                "Column": list(outlier_dict.keys()),
                "Outlier Count": list(outlier_dict.values()),
            })
            out_df["% of Rows"] = (out_df["Outlier Count"] / len(df) * 100).round(2)
            out_df = out_df.sort_values("Outlier Count", ascending=False)

            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                fig = px.bar(out_df, x="Column", y="Outlier Count",
                             color="% of Rows", color_continuous_scale="OrRd",
                             title="Outliers per Column", template="plotly_dark",
                             text="Outlier Count")
                fig.update_traces(textposition="outside")
                theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                st.dataframe(out_df, use_container_width=True, height=300)
                worst = out_df.iloc[0]
                ic(f"<b>{worst['Column']}</b> has the most outliers: <b>{int(worst['Outlier Count'])}</b> rows "
                   f"(<b>{worst['% of Rows']:.1f}%</b> of all data). "
                   "Review these records — they may represent data errors or genuinely extreme events.", "warn")

        st.markdown("---")

    # ─────────────── 4. Correlation Insights ─────────────────
    if len(num_cols) >= 2:
        st.markdown("### 🔗 Correlation Insights")
        corr = get_correlation(df)

        if corr is not None:
            col_a, col_b = st.columns([1.4, 1])
            with col_a:
                fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                color_continuous_scale="RdBu_r",
                                title="Pearson Correlation Matrix",
                                template="plotly_dark", zmin=-1, zmax=1)
                fig.update_layout(**DARK_LAYOUT,
                                  coloraxis_colorbar=dict(title="r", tickvals=[-1, 0, 1]))
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("#### 🏆 Top Correlated Pairs")
                top_pairs = top_correlations(corr, n=8)
                pairs_df = pd.DataFrame(top_pairs, columns=["Column A", "Column B", "Pearson r"])
                pairs_df["Pearson r"] = pairs_df["Pearson r"].round(3)
                pairs_df["Strength"] = pairs_df["Pearson r"].abs().apply(
                    lambda x: "🔴 Strong" if x > 0.7 else ("🟡 Moderate" if x > 0.4 else "⚪ Weak")
                )
                st.dataframe(pairs_df, use_container_width=True, height=300)

                strong_pairs = [(a, b, r) for a, b, r in top_pairs if abs(r) > 0.7]
                if strong_pairs:
                    a, b, r = strong_pairs[0]
                    ic(f"<b>Strongest correlation:</b> <code>{a}</code> ↔ <code>{b}</code> "
                       f"(r = <b>{r:.3f}</b>). "
                       + ("A strong positive relationship — as one increases, so does the other."
                          if r > 0 else
                          "A strong negative relationship — as one increases, the other decreases."))
                elif top_pairs:
                    a, b, r = top_pairs[0]
                    ic(f"<b>Highest correlation:</b> <code>{a}</code> ↔ <code>{b}</code> "
                       f"(r = <b>{r:.3f}</b>). No strongly correlated pairs found — variables are largely independent.", "ok")

        st.markdown("---")

    # ─────────────── 5. Categorical Value Distribution ─────────────────
    if cat_cols:
        st.markdown("### 🏷️ Categorical Column Insights")
        for col in cat_cols[:6]:
            vc = df[col].value_counts()
            top_val = vc.index[0]
            top_pct = vc.iloc[0] / len(df) * 100
            dominant = top_pct > 60
            col_a, col_b = st.columns([1.4, 1])
            with col_a:
                vc_df = vc.head(15).reset_index()
                vc_df.columns = [col, "count"]
                fig = px.bar(vc_df, x=col, y="count",
                             color="count", color_continuous_scale="Teal",
                             title=f"Value Distribution — {col}",
                             template="plotly_dark")
                theme(fig, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                st.markdown(f"**Column:** `{col}`")
                st.metric("Unique Values", df[col].nunique())
                st.metric("Most Common", f"{top_val}")
                st.metric("Dominance", f"{top_pct:.1f}%")
                if dominant:
                    ic(f"<b>'{top_val}'</b> dominates with <b>{top_pct:.1f}%</b> of records. "
                       "This class imbalance may skew aggregations.", "warn")
                else:
                    ic(f"<b>'{top_val}'</b> is the most frequent value at <b>{top_pct:.1f}%</b>. "
                       "The distribution is reasonably spread across categories.", "ok")


# ═════════════════════════════════════════════
#  PAGE: EXECUTIVE DASHBOARD
# ═════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Executive Dashboard")

    if filtered_df is None:
        st.error("⚠️ Please upload data in **📂 Data Upload** first.")
        st.stop()

    df = filtered_df
    num_cols = smart_num_cols(df)
    cat_cols = smart_cat_cols(df)

    # ── KPI row ──────────────────────────────────
    st.markdown("### 📈 Key Metrics")
    kpi_items = [("📋 Rows", f"{len(df):,}"), ("🗂️ Columns", f"{df.shape[1]}")]
    for col in num_cols[:3]:
        kpi_items.append((f"Σ {col}", f"{df[col].sum():,.1f}"))
    if num_cols:
        kpi_items.append((f"Avg {num_cols[0]}", f"{df[num_cols[0]].mean():,.2f}"))
    if len(num_cols) >= 2:
        kpi_items.append((f"Max {num_cols[1]}", f"{df[num_cols[1]].max():,.2f}"))

    cols = st.columns(min(len(kpi_items), 6))
    for i, (label, val) in enumerate(kpi_items[:6]):
        cols[i].metric(label, val)

    st.markdown("---")

    # ── Row 1: Main chart + Donut ─────────────────
    left, right = st.columns([2, 1])

    with left:
        lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon", "longitude", "lng")), None)

        if lat_col and lon_col:
            st.subheader("🌍 Geospatial Heatmap")
            fig = px.density_mapbox(
                df, lat=lat_col, lon=lon_col, radius=10,
                center={"lat": df[lat_col].mean(), "lon": df[lon_col].mean()},
                zoom=3, mapbox_style="carto-darkmatter",
                title="Location Density", color_continuous_scale="Turbo",
            )
            theme(fig, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            ic(f"Geospatial heatmap based on <b>{lat_col}</b> and <b>{lon_col}</b>. "
               f"Hotspots indicate dense clusters of records. "
               f"Geographic centre of data: <b>({df[lat_col].mean():.3f}, {df[lon_col].mean():.3f})</b>.")

        elif len(num_cols) >= 2 and cat_cols:
            grp = (df.groupby(cat_cols[0])[num_cols[0]].sum()
                   .reset_index().sort_values(num_cols[0], ascending=False).head(20))
            st.subheader(f"📊 {num_cols[0]} by {cat_cols[0]}")
            fig = px.bar(grp, x=cat_cols[0], y=num_cols[0],
                         color=num_cols[0], color_continuous_scale="Teal",
                         title=f"Total {num_cols[0]} by {cat_cols[0]}",
                         template="plotly_dark")
            theme(fig, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            top_cat = grp.iloc[0]
            ic(f"<b>{top_cat[cat_cols[0]]}</b> has the highest total <b>{num_cols[0]}</b> "
               f"(<b>{top_cat[num_cols[0]]:,.2f}</b>). "
               f"Top 3 categories account for "
               f"<b>{grp.head(3)[num_cols[0]].sum()/grp[num_cols[0]].sum()*100:.1f}%</b> of the total.")

        elif len(num_cols) >= 2:
            st.subheader(f"📈 {num_cols[0]} vs {num_cols[1]}")
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                             color=num_cols[2] if len(num_cols) > 2 else None,
                             title=f"{num_cols[0]} vs {num_cols[1]}",
                             template="plotly_dark", color_continuous_scale="Viridis",
                             trendline="ols")
            theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            ic(scatter_insight(df, num_cols[0], num_cols[1]))

        elif len(num_cols) == 1:
            st.subheader(f"📊 Distribution of {num_cols[0]}")
            fig = px.histogram(df, x=num_cols[0], marginal="box",
                               template="plotly_dark", color_discrete_sequence=["#00d4ff"])
            theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            ic(chart_insight(df, num_cols[0], "dist"))
        else:
            st.info("No numeric columns found for the main chart.")

    with right:
        if cat_cols:
            chosen_cat = cat_cols[0]
            st.subheader(f"🍩 {chosen_cat} Share")
            vc = df[chosen_cat].value_counts().head(12).reset_index()
            vc.columns = [chosen_cat, "count"]
            fig = px.pie(vc, names=chosen_cat, values="count", hole=0.45,
                         title=f"Distribution – {chosen_cat}",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(**DARK_LAYOUT, legend=dict(font_size=10), margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            top_share = vc.iloc[0]["count"] / vc["count"].sum() * 100
            ic(f"<b>{vc.iloc[0][chosen_cat]}</b> is the dominant category with "
               f"<b>{top_share:.1f}%</b> share. "
               f"<b>{len(vc)}</b> categories are shown.")

        elif num_cols:
            st.subheader(f"📦 Box Plot – {num_cols[0]}")
            fig = px.box(df, y=num_cols[0], template="plotly_dark",
                         color_discrete_sequence=["#00d4ff"])
            theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            s = df[num_cols[0]].describe()
            ic(f"<b>Median:</b> {s['50%']:,.2f} &nbsp;|&nbsp; "
               f"<b>IQR:</b> {s['75%']-s['25%']:,.2f} &nbsp;|&nbsp; "
               f"<b>Range:</b> {s['min']:,.2f} – {s['max']:,.2f}")

    st.markdown("---")

    # ── Row 2: Mean of numeric cols ──
    if num_cols:
        st.markdown("### 📑 Numeric Column Averages")
        means = df[num_cols].mean().reset_index()
        means.columns = ["Column", "Mean"]
        fig = px.bar(means, x="Column", y="Mean",
                     color="Mean", color_continuous_scale="Blues",
                     template="plotly_dark", title="Average Value per Numeric Column",
                     text="Mean")
        fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
        theme(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        highest_mean_col = means.sort_values("Mean", ascending=False).iloc[0]
        ic(f"<b>{highest_mean_col['Column']}</b> has the highest average value "
           f"(<b>{highest_mean_col['Mean']:,.2f}</b>). "
           "Large differences between column means can indicate different units of measurement — consider normalisation for fair comparison.")

    # ── Row 3: Second dimension breakdown ──
    if len(cat_cols) > 1 and num_cols:
        st.markdown("### 🎯 Second Dimension Breakdown")
        c2_cat = cat_cols[1]
        grp2 = (df.groupby(c2_cat)[num_cols[0]].mean()
                .reset_index().sort_values(num_cols[0], ascending=False).head(15))
        grp2.columns = [c2_cat, f"Avg {num_cols[0]}"]
        fig = px.bar(grp2, x=c2_cat, y=f"Avg {num_cols[0]}",
                     color=f"Avg {num_cols[0]}", color_continuous_scale="Plasma",
                     template="plotly_dark",
                     title=f"Average {num_cols[0]} by {c2_cat}")
        theme(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        top2 = grp2.iloc[0]
        bot2 = grp2.iloc[-1]
        ic(f"<b>{top2[c2_cat]}</b> leads with avg <b>{top2[f'Avg {num_cols[0]}']:,.2f}</b>. "
           f"<b>{bot2[c2_cat]}</b> is lowest at <b>{bot2[f'Avg {num_cols[0]}']:,.2f}</b>. "
           f"Gap between top and bottom: <b>{top2[f'Avg {num_cols[0]}'] - bot2[f'Avg {num_cols[0]}']:,.2f}</b>.")

    # ── Row 4: Trend (if date/time-like column exists) ──
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "year" in c.lower() or "month" in c.lower()]
    if date_cols and num_cols:
        st.markdown("### 📅 Time-Based Trend")
        d_col = date_cols[0]
        try:
            df_t = df.copy()
            df_t[d_col] = pd.to_datetime(df_t[d_col], errors="coerce")
            df_t = df_t.dropna(subset=[d_col])
            df_t = df_t.sort_values(d_col)
            trend = df_t.set_index(d_col)[num_cols[0]].resample("ME").sum().reset_index()
            if len(trend) > 1:
                fig = px.area(trend, x=d_col, y=num_cols[0],
                              title=f"Monthly {num_cols[0]} Trend",
                              template="plotly_dark",
                              color_discrete_sequence=["#00d4ff"])
                fig.update_traces(fill="tozeroy", line_color="#00d4ff",
                                  fillcolor="rgba(0,212,255,0.15)")
                theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                first_val = trend[num_cols[0]].iloc[0]
                last_val = trend[num_cols[0]].iloc[-1]
                change = (last_val - first_val) / first_val * 100 if first_val else 0
                direction = "📈 increased" if change > 0 else "📉 decreased"
                ic(f"<b>{num_cols[0]}</b> has <b>{direction}</b> by <b>{abs(change):.1f}%</b> "
                   f"from the earliest to latest period. "
                   f"First period: <b>{first_val:,.2f}</b> | Last period: <b>{last_val:,.2f}</b>.")
        except Exception:
            pass


# ═════════════════════════════════════════════
#  PAGE: ADVANCED ANALYSIS
# ═════════════════════════════════════════════
elif page == "🔬 Advanced Analysis":
    st.title("🔬 Advanced Analysis")

    if filtered_df is None:
        st.error("⚠️ Please upload data in **📂 Data Upload** first.")
        st.stop()

    df = filtered_df
    num_cols = smart_num_cols(df)
    cat_cols = smart_cat_cols(df)
    all_cols = num_cols + cat_cols

    if not all_cols:
        st.warning("No suitable columns found for analysis.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Univariate", "🔗 Bivariate", "🌡️ Correlation", "📋 Group Analytics", "📦 Outlier Explorer"]
    )

    # ── Tab 1: Univariate ─────────────────────────────
    with tab1:
        st.subheader("Distribution Analysis")
        st.caption("Explore how individual columns are distributed.")
        col_to_plot = st.selectbox("Select Column", all_cols, key="uni_col")

        if col_to_plot in num_cols:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig = px.histogram(df, x=col_to_plot, marginal="box",
                                   title=f"Distribution of {col_to_plot}",
                                   template="plotly_dark",
                                   color_discrete_sequence=["#00d4ff"],
                                   nbins=40)
                theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                s = df[col_to_plot].describe()
                for stat, val in s.items():
                    st.metric(stat.capitalize(), f"{val:,.2f}")
            ic(chart_insight(df, col_to_plot, "dist"))

            # Violin plot below
            fig2 = px.violin(df, y=col_to_plot, box=True, points="outliers",
                             title=f"Violin Plot – {col_to_plot}",
                             template="plotly_dark",
                             color_discrete_sequence=["#7b61ff"])
            theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            q1 = df[col_to_plot].quantile(0.25)
            q3 = df[col_to_plot].quantile(0.75)
            ic(f"The violin plot shows the full distribution shape. "
               f"<b>25th percentile:</b> {q1:,.2f} &nbsp;|&nbsp; "
               f"<b>75th percentile:</b> {q3:,.2f} &nbsp;|&nbsp; "
               f"<b>IQR:</b> {q3-q1:,.2f}. "
               "The width of the violin at each value indicates the frequency of data at that level.")

        else:
            vc = df[col_to_plot].value_counts().head(20).reset_index()
            vc.columns = ["value", "count"]
            vc["pct"] = (vc["count"] / len(df) * 100).round(1)
            fig = px.bar(vc, x="value", y="count",
                         title=f"Top Values – {col_to_plot}",
                         template="plotly_dark",
                         color="count", color_continuous_scale="Teal",
                         text="pct")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            theme(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            ic(chart_insight(df, col_to_plot, "cat"))

    # ── Tab 2: Bivariate ─────────────────────────────
    with tab2:
        st.subheader("Relationship Analysis")
        st.caption("Explore how two variables relate to each other.")

        if len(num_cols) < 1:
            st.warning("Need at least one numeric column for bivariate analysis.")
        else:
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X Axis", num_cols, key="biv_x")
            y_col = c2.selectbox("Y Axis", num_cols, index=min(1, len(num_cols) - 1), key="biv_y")
            color_col = st.selectbox("Colour By (optional)", ["— None —"] + cat_cols, key="biv_c")
            clr = None if color_col == "— None —" else color_col

            chart_type = st.radio("Chart Type", ["Scatter (+ Trendline)", "Line", "Box", "Hexbin Density"],
                                  horizontal=True, key="biv_type")

            if chart_type == "Scatter (+ Trendline)":
                fig = px.scatter(df, x=x_col, y=y_col, color=clr,
                                 trendline="ols",
                                 template="plotly_dark",
                                 title=f"{x_col} vs {y_col}",
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 opacity=0.7)
            elif chart_type == "Line":
                fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, color=clr,
                              template="plotly_dark", title=f"{x_col} → {y_col}",
                              color_discrete_sequence=px.colors.qualitative.Bold)
            elif chart_type == "Box":
                fig = px.box(df, x=clr, y=y_col, template="plotly_dark",
                             title=f"{y_col} by {clr or 'overall'}",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            else:  # Hexbin Density
                fig = px.density_heatmap(df, x=x_col, y=y_col,
                                         marginal_x="histogram", marginal_y="histogram",
                                         color_continuous_scale="Viridis",
                                         title=f"Density – {x_col} vs {y_col}",
                                         template="plotly_dark")

            theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            ic(scatter_insight(df, x_col, y_col))

    # ── Tab 3: Correlation ─────────────────────────────
    with tab3:
        st.subheader("Correlation Heatmap")
        st.caption("Understand how numeric variables move together.")
        corr = get_correlation(df)
        if corr is not None:
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Pearson Correlation Matrix",
                            template="plotly_dark", zmin=-1, zmax=1)
            fig.update_layout(**DARK_LAYOUT,
                              coloraxis_colorbar=dict(title="r", tickvals=[-1, 0, 1]))
            st.plotly_chart(fig, use_container_width=True)

            # Insights on top pairs
            top_pairs = top_correlations(corr, n=6)
            st.markdown("#### 🔍 Top Correlation Insights")
            for a, b, r in top_pairs:
                strength = "strongly" if abs(r) > 0.7 else ("moderately" if abs(r) > 0.4 else "weakly")
                direction = "positively" if r > 0 else "negatively"
                kind = "ok" if abs(r) < 0.9 else "warn"
                ic(f"<code>{a}</code> ↔ <code>{b}</code>: r = <b>{r:.3f}</b> — "
                   f"{strength} {direction} correlated. "
                   + ("This high correlation could indicate multicollinearity if used in modelling." if abs(r) > 0.85 else
                      ("A meaningful linear relationship exists." if abs(r) > 0.5 else
                       "Relationship is weak; other factors likely at play.")), kind)
        else:
            st.warning("Need at least 2 numeric columns for correlation.")

    # ── Tab 4: Group Analytics ─────────────────────────────
    with tab4:
        st.subheader("Group Analytics")
        st.caption("Compare a numeric metric across categories.")
        if not cat_cols or not num_cols:
            st.warning("Need at least one categorical and one numeric column.")
        else:
            g_cat = st.selectbox("Group By", cat_cols, key="grp_cat")
            g_num = st.selectbox("Metric", num_cols, key="grp_num")
            agg_fn = st.radio("Aggregation", ["Sum", "Mean", "Count", "Max", "Min"], horizontal=True, key="grp_agg")

            agg_map = {"Sum": "sum", "Mean": "mean", "Count": "count", "Max": "max", "Min": "min"}
            grp = (df.groupby(g_cat)[g_num].agg(agg_map[agg_fn])
                   .reset_index().sort_values(g_num, ascending=False).head(25))
            grp.columns = [g_cat, f"{agg_fn}({g_num})"]
            metric_col = f"{agg_fn}({g_num})"

            col_a, col_b = st.columns([1.5, 1])
            with col_a:
                fig = px.bar(grp, x=g_cat, y=metric_col,
                             color=metric_col, color_continuous_scale="Viridis",
                             title=f"{agg_fn} of {g_num} by {g_cat}",
                             template="plotly_dark", text=metric_col)
                fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
                theme(fig, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                st.dataframe(grp, use_container_width=True, height=330)

            top_g = grp.iloc[0]
            bot_g = grp.iloc[-1]
            ratio = (top_g[metric_col] / bot_g[metric_col]) if bot_g[metric_col] != 0 else float("inf")
            ic(f"<b>{top_g[g_cat]}</b> ranks #1 with {agg_fn.lower()} <b>{g_num}</b> = "
               f"<b>{top_g[metric_col]:,.2f}</b>. "
               f"<b>{bot_g[g_cat]}</b> is at the bottom with <b>{bot_g[metric_col]:,.2f}</b>. "
               + (f"The top group is <b>{ratio:.1f}x</b> the bottom group." if ratio != float("inf") else ""))

            # Pie chart of contribution
            if agg_fn in ("Sum", "Count") and grp[metric_col].sum() > 0:
                fig2 = px.pie(grp, names=g_cat, values=metric_col, hole=0.4,
                              title=f"Proportional Share – {agg_fn} of {g_num}",
                              template="plotly_dark",
                              color_discrete_sequence=px.colors.qualitative.Bold)
                theme(fig2, margin=dict(t=40, b=0))
                st.plotly_chart(fig2, use_container_width=True)
                top3_share = grp.head(3)[metric_col].sum() / grp[metric_col].sum() * 100
                ic(f"Top 3 categories account for <b>{top3_share:.1f}%</b> of the total {agg_fn.lower()} of <b>{g_num}</b>.")

    # ── Tab 5: Outlier Explorer ─────────────────────────────
    with tab5:
        st.subheader("Outlier Explorer")
        st.caption("Visually identify and understand outliers in your numeric columns.")

        if not num_cols:
            st.warning("No numeric columns available.")
        else:
            selected_cols = st.multiselect(
                "Select Columns to Inspect",
                num_cols,
                default=num_cols[:min(4, len(num_cols))],
                key="out_cols"
            )

            if selected_cols:
                fig = px.box(df, y=selected_cols if len(selected_cols) > 1 else selected_cols[0],
                             template="plotly_dark",
                             title="Box Plot – Outlier Overview",
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             points="outliers")
                theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                ic("Each box shows the <b>IQR (25th–75th percentile)</b>. "
                   "Dots beyond the whiskers are <b>potential outliers</b>. "
                   "Long whiskers indicate high variability in that tail of the distribution.")

                # Per-column outlier stats
                st.markdown("#### 📊 Outlier Statistics")
                out_data = []
                for col in selected_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    low_bound = q1 - 1.5 * iqr
                    high_bound = q3 + 1.5 * iqr
                    n_low  = int((df[col] < low_bound).sum())
                    n_high = int((df[col] > high_bound).sum())
                    n_total = n_low + n_high
                    out_data.append({
                        "Column": col,
                        "Lower Bound": round(low_bound, 3),
                        "Upper Bound": round(high_bound, 3),
                        "Low Outliers": n_low,
                        "High Outliers": n_high,
                        "Total Outliers": n_total,
                        "% Outliers": round(n_total / len(df) * 100, 2),
                    })
                out_table = pd.DataFrame(out_data)
                st.dataframe(out_table, use_container_width=True)

                worst = out_table.sort_values("% Outliers", ascending=False).iloc[0]
                if worst["Total Outliers"] > 0:
                    ic(f"<b>{worst['Column']}</b> has the highest outlier rate: "
                       f"<b>{worst['Total Outliers']}</b> records ({worst['% Outliers']:.1f}%). "
                       f"Valid range (IQR method): <b>[{worst['Lower Bound']:,.2f}, {worst['Upper Bound']:,.2f}]</b>. "
                       "These could be genuine extreme values or data entry errors — review them carefully.", "warn")
                else:
                    ic("✅ No outliers detected in the selected columns.", "ok")
