# 1_generate_synthetic_data.py
import numpy as np
import pandas as pd

def generate_synthetic_outage_data(n_samples=1200, random_state=42):
    """
    Generates synthetic dataset for power outage duration prediction
    """
    np.random.seed(random_state)
    
    # Features
    temperature = np.random.uniform(-12, 42, n_samples)          # °C
    wind_speed = np.random.uniform(0, 120, n_samples)            # km/h
    rainfall = np.random.uniform(0, 80, n_samples)               # mm
    equipment_age = np.random.uniform(1, 55, n_samples)          # years
    load_mw = np.random.uniform(40, 600, n_samples)              # MW
    customers_affected = np.random.randint(50, 15000, n_samples)
    
    # Realistic-ish synthetic target (hours)
    base_duration = (
        0.38 * wind_speed +
        0.42 * rainfall +
        0.28 * equipment_age +
        0.12 * load_mw +
        0.0008 * customers_affected -
        0.09 * temperature
    )
    
    # Add some realistic variability and floor at 0.1 hours
    noise = np.random.normal(0, 7, n_samples)
    outage_duration = np.maximum(base_duration + noise, 0.1)
    
    df = pd.DataFrame({
        'temperature_c': temperature,
        'wind_speed_kmh': wind_speed,
        'rainfall_mm': rainfall,
        'equipment_age_years': equipment_age,
        'load_mw': load_mw,
        'customers_affected': customers_affected,
        'outage_duration_hours': outage_duration
    })
    
    return df


if __name__ == "__main__":
    data = generate_synthetic_outage_data(1500)
    print(data.shape)
    print(data.head())
    data.to_csv("synthetic_outages.csv", index=False)
    print("Dataset saved → synthetic_outages.csv")