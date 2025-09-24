import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class XGBoostModel:
    def __init__(self, model_path: str = "models/model.pkl"):
        self.model_path = model_path
        self.model = xgb.XGBRegressor(
            n_estimators=500,        
            max_depth=6,            
            learning_rate=0.05,      
            subsample=0.8,           
            colsample_bytree=0.8,   
            random_state=42,
            n_jobs=-1
        )

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            early_stopping_rounds=20,
            verbose=False
        )

        # Predictions
        y_pred = self.model.predict(X_test)

  
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n=== Training Results ===")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R²  : {r2:.4f}")

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f" Model saved to {self.model_path}")

    def load(self):
        self.model = joblib.load(self.model_path)
        print(f" Model loaded from {self.model_path}")
