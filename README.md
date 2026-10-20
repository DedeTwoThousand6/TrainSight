# TrainSight — Machine Learning Training Time Predictor

**Predict Smarter, Train Better.**

---

## Deskripsi

**TrainSight** adalah aplikasi web prediksi modern yang memperkirakan durasi pelatihan model Machine Learning
berdasarkan ukuran dataset menggunakan algoritma **Regresi Linear Sederhana**.

Aplikasi ini merupakan implementasi nyata dari riset akademik:
> *Analisis Pengaruh Ukuran Dataset terhadap Waktu Pelatihan Model Machine Learning Menggunakan Regresi Linear*

---

## Tujuan Penelitian

Menganalisis dan memodelkan hubungan antara **ukuran dataset pelatihan** (MB) dan **waktu training** (menit)
menggunakan Regresi Linear, sehingga estimasi waktu dapat dilakukan sebelum training dimulai.

---

## Dataset

- **Sumber:** [Kaggle — Training Data vs Training Time](https://www.kaggle.com/datasets/pushpjeetcholkar/training-data-vs-training-time)
- **File:** `data/training_data_vs_training_time_updated.csv`
- **Variabel X (Input):** `training_data_size_mb` — Ukuran dataset (MB)
- **Variabel Y (Output):** `training_time_minutes` — Waktu training (menit)
- **Total Data Awal:** 300 baris

---

## Metode

**Regresi Linear Sederhana:**
```
Y = a + bX
```
- `a` = Intercept (dari model)
- `b` = Koefisien (dari model)
- `X` = Dataset Size (MB)
- `Y` = Predicted Training Time (minutes)

**Split:** `test_size=0.3`, `random_state=101`

---

## Struktur Folder

```
TrainTime AI/
│
├── data/
│   └── training_data_vs_training_time_updated.csv
│
├── notebook/
│   └── regresi_linear_training_time.ipynb
│
├── models/
│   ├── linear_regression_model.pkl
│   └── model_metadata.json
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── visualization.html
│   ├── evaluation.html
│   ├── dataset.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Instalasi

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Cara Menjalankan

### Langkah 1 — Training Model
```bash
python train_model.py
```
Skrip ini akan:
- Membaca dan mempreproses dataset CSV
- Melatih model Regresi Linear
- Menyimpan model ke `models/linear_regression_model.pkl`
- Menyimpan metadata evaluasi ke `models/model_metadata.json`

### Langkah 2 — Jalankan Aplikasi Flask
```bash
python app.py
```

### Langkah 3 — Buka di Browser
```
http://127.0.0.1:5000
```

---

## Fitur Aplikasi

| Halaman | Route | Deskripsi |
|---|---|---|
| Home | `/` | Dashboard utama dengan statistik model |
| Prediction | `/predict` | Form prediksi waktu training |
| Visualization | `/visualization` | 4 grafik interaktif (Chart.js) |
| Model Evaluation | `/evaluation` | MAE, MSE, RMSE, R² |
| Dataset | `/dataset` | Tabel dataset dengan search & pagination |
| About | `/about` | Info aplikasi, metode, dan keterbatasan |

---

## Keterbatasan Model

Durasi training sesungguhnya dipengaruhi oleh banyak faktor:
- Jenis algoritma ML
- Spesifikasi CPU/GPU
- Jumlah epoch
- Kompleksitas model
- Konfigurasi training

Hasil aplikasi adalah **estimasi** berdasarkan pola dataset, bukan jaminan pada semua perangkat.

---

## Tech Stack

| Layer | Teknologi |
|---|---|
| Backend | Python · Flask · Scikit-learn · Pandas · NumPy · Joblib |
| Frontend | HTML5 · CSS3 · Bootstrap 5 · JavaScript |
| Visualisasi | Chart.js |
| Desain | Plus Jakarta Sans · Bootstrap Icons · CSS Animations |

---

*TrainSight — Predict Smarter, Train Better.*
