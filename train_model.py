"""
train_model.py — TrainSight
Melatih model Regresi Linear dan menyimpan model + metadata evaluasi.
Preprocessing, test_size, dan random_state identik dengan notebook.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ========================
# Path Configuration
# ========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "training_data_vs_training_time_updated.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "linear_regression_model.pkl"
META_PATH  = MODELS_DIR / "model_metadata.json"


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """
    Memuat dan membersihkan dataset sesuai langkah notebook:
    A. Hapus kolom Unnamed
    B. Dropna pada kolom utama
    C. Konversi numerik
    D. Hapus nilai tidak valid (training_time_minutes <= 0 atau training_data_size_mb <= 0)
    E. Reset index
    Mengembalikan DataFrame bersih.
    """
    df = pd.read_csv(csv_path)

    # A. Hapus kolom 'Unnamed'
    unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    # B. Hapus missing values pada kolom utama
    df.dropna(subset=["training_data_size_mb", "training_time_minutes"], inplace=True)

    # C. Pastikan tipe numerik
    df["training_data_size_mb"] = pd.to_numeric(df["training_data_size_mb"], errors="coerce")
    df["training_time_minutes"] = pd.to_numeric(df["training_time_minutes"], errors="coerce")
    df.dropna(subset=["training_data_size_mb", "training_time_minutes"], inplace=True)

    # D. Hapus nilai tidak valid
    df = df[df["training_time_minutes"] > 0]
    df = df[df["training_data_size_mb"] > 0]

    # E. Reset index
    df.reset_index(drop=True, inplace=True)

    return df


def train_and_save():
    print("=" * 55)
    print("  TrainSight — Model Training")
    print("=" * 55)

    # 1. Load & Preprocessing
    print(f"\n[1/5] Memuat dataset dari: {DATA_PATH}")
    df = load_and_preprocess(DATA_PATH)
    print(f"      Jumlah data setelah preprocessing: {len(df)} baris")

    # 2. Feature / Target split
    X = df[["training_data_size_mb"]]
    y = df["training_time_minutes"]
    print(f"\n[2/5] Fitur X: training_data_size_mb | Target y: training_time_minutes")

    # 3. Train / Test split  — sama dengan notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )
    print(f"\n[3/5] Split data → Training: {len(X_train)} | Testing: {len(X_test)}")

    # 4. Training model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n[4/5] Model Regresi Linear berhasil dilatih.")

    # 5. Evaluasi & simpan
    predictions = model.predict(X_test)
    mae  = metrics.mean_absolute_error(y_test, predictions)
    mse  = metrics.mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2   = metrics.r2_score(y_test, predictions)

    print(f"\n[5/5] Hasil Evaluasi:")
    print(f"      MAE  : {mae:.4f}")
    print(f"      MSE  : {mse:.4f}")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R²   : {r2:.4f}")
    print(f"      Intercept  : {model.intercept_:.4f}")
    print(f"      Coefficient: {model.coef_[0]:.4f}")

    # Simpan model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✓ Model disimpan di: {MODEL_PATH}")

    # Simpan metadata
    metadata = {
        "total_data":       int(len(df)),
        "train_size":       int(len(X_train)),
        "test_size":        int(len(X_test)),
        "mae":              round(float(mae),  4),
        "mse":              round(float(mse),  4),
        "rmse":             round(float(rmse), 4),
        "r2":               round(float(r2),   4),
        "intercept":        round(float(model.intercept_), 4),
        "coefficient":      round(float(model.coef_[0]),   4),
        "x_train_min":      round(float(X_train["training_data_size_mb"].min()), 4),
        "x_train_max":      round(float(X_train["training_data_size_mb"].max()), 4),
        # Scatter data (all clean data)
        "scatter_x":        [round(v, 2) for v in X["training_data_size_mb"].tolist()],
        "scatter_y":        [round(v, 2) for v in y.tolist()],
        # Actual vs Predicted (test set)
        "y_test":           [round(v, 2) for v in y_test.tolist()],
        "y_pred":           [round(v, 2) for v in predictions.tolist()],
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata disimpan di: {META_PATH}")
    print("\n" + "=" * 55)
    print("  Selesai! Jalankan: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    train_and_save()
