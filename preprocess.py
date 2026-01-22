# 2_preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath="synthetic_outages.csv"):
    df = pd.read_csv(filepath)
    
    # Features & target
    feature_cols = [
        'temperature_c', 'wind_speed_kmh', 'rainfall_mm',
        'equipment_age_years', 'load_mw', 'customers_affected'
    ]
    X = df[feature_cols].copy()
    y = df['outage_duration_hours'].copy()
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame (optional - helps with reporting)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, scaler, feature_cols


def save_preprocessed_data(X_scaled, y, feature_cols, prefix="preprocessed"):
    pd.DataFrame(X_scaled, columns=feature_cols).to_csv(f"{prefix}_X.csv", index=False)
    y.to_csv(f"{prefix}_y.csv", index=False)
    print(f"Saved: {prefix}_X.csv and {prefix}_y.csv")


if __name__ == "__main__":
    X_s, y_s, sc, cols = load_and_preprocess_data()
    print("X shape:", X_s.shape)
    print("y shape:", y_s.shape)
    save_preprocessed_data(X_s, y_s, cols)