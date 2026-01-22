# 3_train_model.py
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor   ← you can swap here

def train_and_save_model(X, y, model_path="outage_model_rf.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    # model = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42)
    
    model.fit(X_train, y_train)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved → {model_path}")
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import numpy as np
    X = np.loadtxt("preprocessed_X.csv", delimiter=",", skiprows=1)
    y = np.loadtxt("preprocessed_y.csv", skiprows=1)
    train_and_save_model(X, y)