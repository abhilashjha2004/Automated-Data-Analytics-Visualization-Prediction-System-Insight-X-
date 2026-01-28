import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import io

# Setup Page Configuration
st.set_page_config(
    page_title="InsightX",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0e1117;
        border-radius: 10px;
        font-weight: bold;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load Data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

# Sidebar Navigation
st.sidebar.title("InsightX 🔮")
st.sidebar.info("Drag & Drop Data Science Platform")
page = st.sidebar.radio("Navigation", ["📂 Data Upload", "📊 Executive Dashboard", "🔬 Advanced Analysis", "🤖 Predictive Modeling"])

# Session State for Data
if "data" not in st.session_state:
    st.session_state["data"] = None

# --- SIDEBAR: GLOBAL FILTERS (only for Dashboard & Analysis) ---
filtered_df = None
if st.session_state["data"] is not None and page in ["📊 Executive Dashboard", "🔬 Advanced Analysis"]:
    df = st.session_state["data"]
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Global Filters")
    
    # Auto-detect Filterable Columns (Categorical < 20 unique values)
    filterable_cols = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() < 20]
    
    filters = {}
    for col in filterable_cols:
        unique_vals = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=unique_vals)
        filters[col] = selected
    
    # Apply Filters
    filtered_df = df.copy()
    for col, selected_vals in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
else:
    filtered_df = st.session_state["data"]


# --- PAGE: DATA UPLOAD ---
if page == "📂 Data Upload":
    st.title("📂 Upload Data")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state["data"] = df
        st.success("File Uploaded Successfully!")
        st.rerun() # Fixed: experimental_rerun is deprecated
        
    if st.session_state["data"] is not None:
        df = st.session_state["data"]
        st.markdown("### 📋 Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Duplicates", df.duplicated().sum())
    else:
        st.info("Please upload a file to start using InsightX.")


# --- PAGE: EXECUTIVE DASHBOARD ---
elif page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    
    if st.session_state["data"] is not None:
        df_view = filtered_df
        
        # 1. KPI Row
        st.markdown("### 📈 Key Performance Indicators")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Rows", f"{len(df_view):,}")
        
        numeric_cols = df_view.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for i, col in enumerate(numeric_cols[:3]):
            if i + 1 < 4:
                val = df_view[col].sum()
                kpi_cols[i+1].metric(f"Total {col}", f"{val:,.0f}")
        
        st.markdown("---")
        
        # 2. Main Grid Layout
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # MAP or SCATTER
            lat_col = next((col for col in df_view.columns if col.lower() in ['lat', 'latitude']), None)
            lon_col = next((col for col in df_view.columns if col.lower() in ['lon', 'longitude']), None)
            
            if lat_col and lon_col:
                st.subheader("🌍 Geospatial Hotspots")
                fig_map = px.density_mapbox(
                    df_view, 
                    lat=lat_col, 
                    lon=lon_col, 
                    radius=10,
                    center=dict(lat=df_view[lat_col].mean(), lon=df_view[lon_col].mean()), 
                    zoom=3,
                    mapbox_style="carto-positron",
                    title="Location Density Heatmap"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            elif len(numeric_cols) > 1:
                st.subheader("📈 Primary Trend")
                fig_trend = px.scatter(df_view, x=numeric_cols[0], y=numeric_cols[1], color=numeric_cols[2] if len(numeric_cols) > 2 else None, title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Not enough data for map or scatter plot.")

        with c2:
            # DONUT CHART
            cat_cols = df_view.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                st.subheader("Distribution")
                fig_donut = px.pie(df_view, names=cat_cols[0], hole=0.4, title=f"By {cat_cols[0]}")
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                 st.info("No categorical data for distribution.")
        
        # 3. Aggregated Summary Charts (User Requested)
        st.markdown("### 📑 Visualization Summary")
        if len(numeric_cols) > 0:
            st.bar_chart(df_view[numeric_cols].mean())

    else:
        st.error("Please upload data in '📂 Data Upload' first.")


# --- PAGE: ADVANCED ANALYSIS ---
elif page == "🔬 Advanced Analysis":
    st.title("🔬 Advanced Analysis")
    
    if st.session_state["data"] is not None:
        df = filtered_df # Use filtered data
        
        # Variable Selection
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "Correlation"])
        
        with tab1:
            st.subheader("Distribution Analysis")
            col_to_plot = st.selectbox("Select Column to Visualize", numeric_cols + cat_cols)
            
            if col_to_plot in numeric_cols:
                fig = px.histogram(df, x=col_to_plot, marginal="box", title=f"Distribution of {col_to_plot}", color_discrete_sequence=['#00d4ff'])
            else:
                fig = px.bar(df[col_to_plot].value_counts().reset_index(), x='index', y=col_to_plot, title=f"Count of {col_to_plot}", color_discrete_sequence=['#00d4ff'])
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Relationship Analysis")
            x_col = st.selectbox("Select X Axis", numeric_cols)
            y_col = st.selectbox("Select Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            color_col = st.selectbox("Color By (Optional)", [None] + cat_cols)
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader("Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation.")

    else:
        st.error("Please upload data in '📂 Data Upload' first.")

# --- PAGE: PREDICTIVE MODELING ---
elif page == "🤖 Predictive Modeling":
    st.title("🤖 Predictive Analytics Engine")
    
    if st.session_state["data"] is not None:
        df = st.session_state["data"].copy()

        
        # Data Preparation (Simple)
        df = df.dropna()
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            try:
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception as e:
                st.warning(f"Skipping column '{col}' due to encoding error: {e}")
                df = df.drop(columns=[col])
            
        target = st.selectbox("Select Target Variable (What to predict?)", df.columns)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X = df.drop(columns=[target])
                y = df[target]
                
                # Determine task type
                is_classification = len(y.unique()) < 20 or y.dtype == 'object'
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if is_classification:
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    st.success(f"Model Trained! (Classification)")
                    st.metric("Accuracy", f"{acc:.2%}")
                else:
                    model = RandomForestRegressor()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    st.success(f"Model Trained! (Regression)")
                    st.metric("R2 Score", f"{r2:.2f}")
                
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance", color='Importance')
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Please upload data in the 'Upload & Insight' page first.")
