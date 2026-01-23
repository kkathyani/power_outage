# 3_train_model.py
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_and_save_model(X, y, model_path="model.joblib"):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Define model
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Save the trained model correctly (NOT a string)
    joblib.dump(model, model_path)

    print(f"Model trained and saved successfully → {model_path}")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load preprocessed data
    X = np.loadtxt("preprocessed_X.csv", delimiter=",", skiprows=1)
    y = np.loadtxt("preprocessed_y.csv", delimiter=",", skiprows=1)

    # Train and save model
    train_and_save_model(X, y)
