"""
app.py — TrainSight Flask Web Application
Machine Learning Training Time Predictor
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ========================
# Path Configuration
# ========================
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "data"  / "training_data_vs_training_time_updated.csv"
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"
META_PATH  = BASE_DIR / "models" / "model_metadata.json"

app = Flask(__name__)


# ========================
# Preprocessing (same as notebook & train_model.py)
# ========================
def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    df.dropna(subset=["training_data_size_mb", "training_time_minutes"], inplace=True)
    df["training_data_size_mb"] = pd.to_numeric(df["training_data_size_mb"], errors="coerce")
    df["training_time_minutes"] = pd.to_numeric(df["training_time_minutes"], errors="coerce")
    df.dropna(subset=["training_data_size_mb", "training_time_minutes"], inplace=True)
    df = df[df["training_time_minutes"] > 0]
    df = df[df["training_data_size_mb"] > 0]
    df.reset_index(drop=True, inplace=True)
    return df


# ========================
# Load Model & Metadata on Startup
# ========================
model    = None
metadata = {}

try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    print("[OK] Model dan metadata berhasil dimuat.")
except Exception as e:
    print(f"[WARN] Gagal memuat model: {e}")
    print("  Jalankan: python train_model.py terlebih dahulu.")


# ========================
# Routes
# ========================
@app.route("/")
def index():
    stats = {
        "total_data":  metadata.get("total_data",  "–"),
        "train_size":  metadata.get("train_size",  "–"),
        "test_size":   metadata.get("test_size",   "–"),
        "r2":          metadata.get("r2",          "–"),
    }
    return render_template("index.html", stats=stats, model_loaded=(model is not None))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result   = None
    error    = None
    warning  = None
    input_val = ""

    if request.method == "POST":
        raw = request.form.get("dataset_size", "").strip()
        input_val = raw

        # --- Validation ---
        if not raw:
            error = "Input wajib diisi."
        else:
            try:
                dataset_size = float(raw)
                if not np.isfinite(dataset_size):
                    raise ValueError("Nilai tidak valid (NaN atau Infinity).")
                if dataset_size <= 0:
                    error = "Ukuran dataset harus lebih besar dari nol."
                else:
                    # --- Prediction ---
                    input_df = pd.DataFrame({"training_data_size_mb": [dataset_size]})
                    pred_val = float(model.predict(input_df)[0])

                    # Range warning
                    x_min = metadata.get("x_train_min", 0)
                    x_max = metadata.get("x_train_max", 1e9)
                    if dataset_size < x_min or dataset_size > x_max:
                        warning = (
                            f"Input ({dataset_size:.2f} MB) berada di luar rentang data "
                            f"pelatihan ({x_min:.2f}–{x_max:.2f} MB). "
                            "Hasil prediksi mungkin kurang akurat."
                        )

                    if pred_val <= 0:
                        error = (
                            f"Model menghasilkan prediksi waktu negatif "
                            f"({pred_val:.4f} menit) untuk input ini. "
                            "Model tidak dapat memberikan estimasi yang masuk akal "
                            "untuk ukuran dataset tersebut."
                        )
                    else:
                        hours   = int(pred_val // 60)
                        minutes = pred_val - hours * 60

                        result = {
                            "dataset_size": dataset_size,
                            "pred_minutes": round(pred_val, 2),
                            "hours":        hours,
                            "minutes_rem":  round(minutes, 2),
                        }
            except ValueError as ve:
                error = f"Input tidak valid: {ve}"

    # Chart data from metadata
    chart = {
        "scatter_x": metadata.get("scatter_x", []),
        "scatter_y": metadata.get("scatter_y", []),
        "coef":      metadata.get("coefficient", 0),
        "intercept": metadata.get("intercept",   0),
        "x_min":     metadata.get("x_train_min", 0),
        "x_max":     metadata.get("x_train_max", 5000),
    }

    return render_template(
        "predict.html",
        result=result,
        error=error,
        warning=warning,
        input_val=input_val,
        chart=chart,
        model_loaded=(model is not None),
    )


@app.route("/visualization")
def visualization():
    chart = {
        "scatter_x": metadata.get("scatter_x", []),
        "scatter_y": metadata.get("scatter_y", []),
        "y_test":    metadata.get("y_test",    []),
        "y_pred":    metadata.get("y_pred",    []),
        "coef":      metadata.get("coefficient", 0),
        "intercept": metadata.get("intercept",   0),
        "x_min":     metadata.get("x_train_min", 0),
        "x_max":     metadata.get("x_train_max", 5000),
    }
    return render_template("visualization.html", chart=chart, model_loaded=(model is not None))


@app.route("/evaluation")
def evaluation():
    eval_data = {
        "mae":         metadata.get("mae",         "–"),
        "mse":         metadata.get("mse",         "–"),
        "rmse":        metadata.get("rmse",        "–"),
        "r2":          metadata.get("r2",          "–"),
        "intercept":   metadata.get("intercept",   "–"),
        "coefficient": metadata.get("coefficient", "–"),
        "train_size":  metadata.get("train_size",  "–"),
        "test_size":   metadata.get("test_size",   "–"),
    }
    return render_template("evaluation.html", eval_data=eval_data, model_loaded=(model is not None))


@app.route("/dataset")
def dataset():
    try:
        df = load_and_preprocess(DATA_PATH)
        rows = df[["training_data_size_mb", "training_time_minutes"]].round(2).to_dict(orient="records")
        stats = {
            "total":   len(rows),
            "x_min":   round(float(df["training_data_size_mb"].min()), 2),
            "x_max":   round(float(df["training_data_size_mb"].max()), 2),
            "x_mean":  round(float(df["training_data_size_mb"].mean()), 2),
            "y_min":   round(float(df["training_time_minutes"].min()), 2),
            "y_max":   round(float(df["training_time_minutes"].max()), 2),
            "y_mean":  round(float(df["training_time_minutes"].mean()), 2),
        }
    except Exception as e:
        rows  = []
        stats = {}
        print(f"Error loading dataset: {e}")
    return render_template("dataset.html", rows=rows, stats=stats)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
