# 4_evaluate.py
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model_path="outage_model_rf.pkl", X_test=None, y_test=None):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    if X_test is None or y_test is None:
        print("No test data provided. Evaluation skipped.")
        return
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"MAE:  {mae:.3f} hours")
    print(f"RMSE: {rmse:.3f} hours")
    print(f"R²:   {r2:.3f}")
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": y_pred}


if __name__ == "__main__":
    # Example usage with saved split (you can also pass your own)
    X_test = np.loadtxt("preprocessed_X.csv", delimiter=",", skiprows=1)[:240]   # dummy
    y_test = np.loadtxt("preprocessed_y.csv", skiprows=1)[:240]
    evaluate_model(X_test=X_test, y_test=y_test)