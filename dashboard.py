# dashboard.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Outage Duration Predictor", layout="wide")

st.title("Utility Outage Duration Prediction Dashboard")

# ── Load model & feature names ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model_path = "outage_model_rf.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success(f"Model loaded: {model_path}")
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}\n\nPlease run 3_train_model.py first!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    feature_names = [
        'temperature_c',
        'wind_speed_kmh',
        'rainfall_mm',
        'equipment_age_years',
        'load_mw',
        'customers_affected'
    ]
    return model, feature_names

model, feature_names = load_assets()

# ── Sidebar: Prediction inputs ───────────────────────────────────────────────
with st.sidebar:
    st.header("Predict Outage Duration")
    
    temp = st.slider("Temperature (°C)", -15, 50, 22)
    wind = st.slider("Wind Speed (km/h)", 0, 130, 25)
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, 4.5, step=0.5)
    age = st.slider("Equipment Age (years)", 0, 60, 15)
    load = st.slider("Load (MW)", 30, 700, 220)
    customers = st.slider("Customers Affected", 50, 20000, 1800, step=100)
    
    if st.button("Predict", type="primary"):
        input_array = np.array([[temp, wind, rain, age, load, customers]], dtype=float)
        pred = model.predict(input_array)[0]
        st.success(f"**Estimated outage duration: {pred:.2f} hours**")

# ── Main content ─────────────────────────────────────────────────────────────
st.subheader("Feature Importance – What Drives Outage Duration?")

if hasattr(model, "feature_importances_"):
    # Prepare sorted importances
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    # Two-column layout: bar + pie
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bar Chart** (precise ranking)")
        fig_bar = px.bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            title="Feature Importance (sorted)",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            color=importances.values,
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Pie Chart** (proportions at a glance)")
        fig_pie = px.pie(
            values=importances.values,
            names=importances.index,
            title="Relative Contribution",
            hole=0.35,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_white"
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial',
            hovertemplate="%{label}<br>Importance: %{value:.4f}<br>Percent: %{percent}"
        )
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(t=80, b=100)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Exact values table
    st.markdown("**Exact Feature Importance Values**")
    st.dataframe(
        importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Feature importances not available (model may not support it).")

# Footer
st.markdown("---")
st.caption("Outage Duration Prediction • Random Forest • Assignment Part A • 2026")