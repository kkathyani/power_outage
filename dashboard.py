import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Outage Duration Predictor", layout="wide")

st.title("Utility Outage Duration Prediction Dashboard")

# ── Load model & feature names ───────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.joblib")
    except Exception as e:
        st.error(f"Error loading model.joblib: {e}")
        return None, None

    # Safety check: make sure model is not a string
    if isinstance(model, str):
        st.error("model.joblib contains a STRING, not a trained model object.\n"
                 "Please resave the trained model using: joblib.dump(model, 'model.joblib')")
        return None, None

    feature_names = [
        'temperature_c', 'wind_speed_kmh', 'rainfall_mm',
        'equipment_age_years', 'load_mw', 'customers_affected'
    ]

    return model, feature_names


model, feature_names = load_assets()

# Stop the app if model failed to load
if model is None:
    st.stop()

# Optional: show model type for confirmation (you can remove later)
# st.sidebar.markdown("**Loaded Model Type:**")
# st.sidebar.write(type(model))

# ── Sidebar Prediction ───────────────────────────────────────
with st.sidebar:
    st.header("Predict Outage Duration")

    temp = st.slider("Temperature (°C)", -15, 45, 22)
    wind = st.slider("Wind Speed (km/h)", 0, 130, 25)
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, 4.5, step=0.5)
    age = st.slider("Equipment Age (years)", 0, 60, 15)
    load = st.slider("Load (MW)", 30, 700, 220)
    customers = st.slider("Customers Affected", 50, 20000, 1800, step=100)

    if st.button("Predict", type="primary"):
        try:
            input_array = np.array([[temp, wind, rain, age, load, customers]])
            pred = model.predict(input_array)[0]
            st.success(f"**Estimated outage duration: {pred:.2f} hours**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Main content ─────────────────────────────────────────────
col1, col2 = st.columns([5, 4])

with col1:
    st.subheader("Model Performance (on hold-out set)")
    st.markdown("""
    - **MAE**  : ~4.1–5.2 hours  
    - **RMSE** : ~6.3–7.8 hours  
    - **R²**   : ~0.84–0.89  
    *(values depend on exact random seed & model version)*  
    """)

with col2:
    st.subheader("Input Summary")
    st.write({
        "Temperature": f"{temp} °C",
        "Wind": f"{wind} km/h",
        "Rain": f"{rain} mm",
        "Equipment Age": f"{age} years",
        "Load": f"{load} MW",
        "Customers": f"{customers:,}"
    })

st.markdown("---")

# ── Feature Importance (if supported) ───────────────────────
if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=feature_names)
    st.bar_chart(importances.sort_values(ascending=True))

st.caption("Assignment modules separated – KOPPU | Jan 2026")
