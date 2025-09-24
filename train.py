#!/usr/bin/env python3
"""
train.py

Full pipeline:
- fetch data (DefiLlamaFetcher + CoinGeckoFetcher)
- feature engineering (lags, rolling, interactions)
- imputation (drop all-NaN columns, then mean-impute)
- LinearRegression baseline, LightGBM, XGBoost (GridSearchCV with TimeSeriesSplit)
- quantile LightGBM for confidence intervals
- save best XGBoost model (via XGBoostModel wrapper)
- generate plots (feature importances + residuals)
- export final PDF report (training_report.pdf) with metrics, best params, top features, plots, and CI example
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Project imports
from src.data.fetcher import DefiLlamaFetcher, CoinGeckoFetcher
from src.data.feature_engineer import FeatureEngineer
from src.models.xgboost_model import XGBoostModel

load_dotenv()


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 3, 5, 7]:
        df[f"apy_lag_{lag}"] = df["apy"].shift(lag)
        df[f"tvl_lag_{lag}"] = df["tvlUsd"].shift(lag)

    df["apy_roll_mean_3"] = df["apy"].rolling(3).mean()
    df["apy_roll_std_7"] = df["apy"].rolling(7).std()
    df["tvl_roll_std_7"] = df["tvlUsd"].rolling(7).std()

    if "apyBase" in df.columns and "tvl_growth" in df.columns:
        df["apyBase_tvlGrowth"] = df["apyBase"] * df["tvl_growth"]
    if "apy_vol" in df.columns and "price" in df.columns:
        df["apyVol_price"] = df["apy_vol"] * df["price"]

    if "tvlUsd" in df.columns:
        df["tvlUsd_log1p"] = np.log1p(df["tvlUsd"].clip(lower=0))
    if "apy" in df.columns:
        df["apy_log1p"] = np.log1p(df["apy"].clip(lower=0))

    return df


def impute_and_keep(X: pd.DataFrame):
    X2 = X.dropna(axis=1, how="all")
    imputer = SimpleImputer(strategy="mean")
    arr = imputer.fit_transform(X2)
    return pd.DataFrame(arr, columns=X2.columns, index=X2.index), list(X2.columns)

def save_feature_importance_plot(importance_df, path="feature_importances.png"):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    top = importance_df.head(15)
    ax.barh(range(len(top)), top["importance"].values[::-1], align="center")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def save_residuals_plot(y_true, y_pred, path="residuals.png"):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (Residual Plot)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_pdf_report(output_path, metrics, best_params, importance_df, figs, ci_example):
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Model Training Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Final Metrics", styles["Heading2"]))
    rows = [["Metric", "Value"]] + [[k, f"{v:.6f}"] for k, v in metrics.items()]
    tbl = Table(rows, colWidths=[200, 200])
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                             ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                             ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                             ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Best XGBoost Parameters", styles["Heading2"]))
    params_rows = [["Parameter", "Value"]] + [[str(k), str(v)] for k, v in best_params.items()]
    params_tbl = Table(params_rows, colWidths=[200, 200])
    params_tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(params_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Feature Importances (Top 15)", styles["Heading2"]))
    fi_rows = [["Feature", "Importance"]] + [[str(row["feature"]), f"{row['importance']:.6f}"] for _, row in importance_df.head(15).iterrows()]
    fi_tbl = Table(fi_rows, colWidths=[300, 100])
    fi_tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(fi_tbl)
    story.append(Spacer(1, 12))

    for title, img_path in figs.items():
        story.append(Paragraph(title, styles["Heading3"]))
        try:
            story.append(Image(img_path, width=450, height=300))
        except Exception as e:
            story.append(Paragraph(f"Could not add image {img_path}: {e}", styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Example Forecast with Confidence Interval", styles["Heading2"]))
    story.append(Paragraph(ci_example, styles["Normal"]))

    doc.build(story)
    print(f"PDF report written to: {output_path}")

def main():
    print("Fetching data...")
    fetcher = DefiLlamaFetcher()
    pool_id = "aa70268e-4b52-42bf-a116-608b370f9501"
    df = fetcher.get_pool_chart(pool_id)

    cg = CoinGeckoFetcher()
    prices = cg.get_token_prices("aave")
    df = df.merge(prices, on="timestamp", how="left")

    fe = FeatureEngineer()
    df = fe.create_features(df)
    df = add_advanced_features(df)

    df["target"] = df["apy"].shift(-1)

    for c in df.columns:
        if df[c].dtype == "object" and c not in ["timestamp"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    exclude = {"timestamp", "target", "apy", "tvlUsd", "date"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df[feature_cols] = df[feature_cols].ffill()
    df = df.dropna(subset=["target"])

    X_raw, y = df[feature_cols], df["target"]
    X, valid_features = impute_and_keep(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    print("Linear Regression R²:", r2_score(y_test, preds_lr))

    lgbm = LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    print("Tuning XGBoost...")
    xgb_wrapper = XGBoostModel(model_path=os.getenv("MODEL_PATH", "models/model.pkl"))
    xgb_est = xgb_wrapper.model
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [0.1, 1.0, 10],
        "n_estimators": [200, 500],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(xgb_est, param_grid, cv=tscv, scoring="r2", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30, verbose=False)

    best_params, best_xgb = grid.best_params_, grid.best_estimator_
    preds_xgb = best_xgb.predict(X_test)
    mse, rmse = mean_squared_error(y_test, preds_xgb), math.sqrt(mean_squared_error(y_test, preds_xgb))
    mae, r2 = mean_absolute_error(y_test, preds_xgb), r2_score(y_test, preds_xgb)

    importance_df = pd.DataFrame({"feature": valid_features, "importance": best_xgb.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    os.makedirs("models", exist_ok=True)
    xgb_wrapper.model = best_xgb
    xgb_wrapper.save()

    figs = {}
    figs["Feature Importances"] = save_feature_importance_plot(importance_df)
    figs["Residuals"] = save_residuals_plot(y_test.values, preds_xgb)


    q_low, q_high = 0.05, 0.95
    lgb_low = LGBMRegressor(objective="quantile", alpha=q_low, random_state=42)
    lgb_high = LGBMRegressor(objective="quantile", alpha=q_high, random_state=42)
    lgb_low.fit(X_train, y_train)
    lgb_high.fit(X_train, y_train)

    pred_point = preds_xgb[-1]
    pred_low = lgb_low.predict(X_test)[-1]
    pred_high = lgb_high.predict(X_test)[-1]
    ci_example = f"Expected APY tomorrow ≈ {pred_point:.2f}% (95% CI: {pred_low:.2f}% – {pred_high:.2f}%)"

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}
    pdf_path = "training_report.pdf"
    build_pdf_report(pdf_path, metrics, best_params, importance_df, figs, ci_example)

    print("Done — training_report.pdf created.")

if __name__ == "__main__":
    main()
