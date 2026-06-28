# Fraud Detection App

Replikasi jurnal **Xia (2022) — Credit Card Fraud Detection Based on SVM** dengan pendekatan dual-model SVM yang di-deploy sebagai API FastAPI dan GUI Streamlit.

## Deskripsi

Proyek ini mengimplementasikan SVM-based fraud detection pada dataset transaksi kartu kredit dari Kaggle (fraudTrain.csv / fraudTest.csv). Dua model SVM dengan optimasi berbeda (AUC dan F1) dilatih dan di-deploy secara side-by-side. Pengguna dapat menginput 6 fitur transaksi melalui form GUI (amount, category, dob, buyer state, merchant state, city population density), dan model akan memprediksi probabilitas fraud. Output ditampilkan sebagai fraud score 0-100 untuk kedua model sekaligus di GUI.

## Tech Stack

| Layer | Teknologi | Detail |
|-------|----------|--------|
| **Training** | Python 3.13, Jupyter Notebook | scikit-learn (SVM), pandas, matplotlib, seaborn, joblib |
| **Backend API** | FastAPI (uvicorn) | POST /predict, GET /health |
| **Frontend GUI** | Streamlit | Tab Assessment (prediksi) + Exploration (insight) |
| **Dataset** | Kaggle Fraud Detection | 1.2M transaksi (train), 555K transaksi (test) |
| **Artefak** | pickle (.pkl), PNG | model, encoder, scaler, evaluasi chart |

## Project Structure

```
├── api.py              # FastAPI backend
├── gui.py              # Streamlit frontend
├── train_models.ipynb  # Training notebook (Colab syntax, 10 step)
├── models/
│   ├── model_auc.pkl       # SVM RBF (C=10, gamma=0.01) — optimasi AUC
│   ├── model_f1.pkl        # SVM RBF (C=0.8, gamma=0.06) — optimasi F1
│   ├── encoder.pkl         # OrdinalEncoder
│   ├── scaler.pkl          # StandardScaler
│   ├── features.pkl        # Feature list
│   ├── kolom_kategori.pkl  # Kolom kategori
│   └── evaluasi_*.png      # Confusion Matrix + ROC Curve (5 model)
├── data/               # fraudTrain.csv, fraudTest.csv (gitignored)
└── requirements.txt    # Dependencies (tanpa versi)
```

## Quick Start

```bash
# Terminal 1 — API
python api.py

# Terminal 2 — GUI
streamlit run gui.py
```

API berjalan di `http://localhost:5000`, GUI di `http://localhost:8501`.

## Cara Kerja

1. **Training** dilakukan di `train_models.ipynb` — GridSearch pada 5 konfigurasi SVM.
2. **Artefak** (model, encoder, scaler) disimpan ke folder `models/`.
3. **API** memuat artefak, menerima 6 fitur dari user + 4 hidden feature (merchant, zip, job, gender) di-sample random dari 5000 baris training.
4. **Ensemble 25 samples** — rata-rata decision function dari 25 kombinasi hidden feature untuk hasil yang stabil.
5. **Fraud score** dihitung dari decision function via tanh(dec/2.0), diskalakan ke 0-100.
6. **Response dual model** — kedua model (AUC dan F1) dijalankan bersamaan, tanpa perlu memilih model.

## Model Performance

5 model SVM dilatih dan dievaluasi dengan Confusion Matrix + ROC Curve:

| Model | Kernel | C | gamma | Test AUC | Test F1 | Karakteristik |
|-------|--------|---|--------|--------|-------|--------------|
| 1a (AUC) | RBF | 10 | 0.01 | **0.8381** | 0.4291 | Diskriminasi terbaik, recall 0.65, precision 0.32 |
| 2a (F1) | RBF | 0.8 | 0.06 | 0.8116 | 0.2483 | Recall 0.60, precision 0.16, mendekati jurnal (F1 0.260) |
| 1b (AUC) | Poly | 0.01 | 0.1 | 0.7744 | 0.1634 | Recall 0.64, precision 0.09 (91% false positive) |
| 2b (F1) | Poly | 0.1 | 0.01 | 0.7601 | **0.3547** | Precision-recall paling seimbang, FP paling sedikit (77) |
| 2c (F1) | RBF | 100 | 0.1 | 0.7444 | 0.1939 | Overfitting ekstrem, recall 0.18 |

### Detail per model

**Model 1a — AUC RBF (C=10, gamma=0.01)**
Confusion Matrix test: TP=59, FN=32, FP=125, TN=9784. Recall fraud 0.65 (menangkap 2/3 fraud), precision 0.32 (68% alarm false positive). ROC AUC test 0.8381 — masih di atas diagonal tetapi gap dengan train AUC 0.9610 mengindikasikan overfitting. Dibanding jurnal (AUC test 0.90), replikasi lebih rendah 6 poin. Model ini digunakan API sebagai model "jurnal".

**Model 2a — F1 RBF (C=0.8, gamma=0.06)**
Confusion Matrix test: TP=55, FN=36, FP=289, TN=9620. Recall 0.60, precision 0.16 (hanya 16% flagged transaction benar-benar fraud). ROC AUC test 0.8116. F1 test 0.2483 mendekati jurnal (0.260) dengan selisih hanya 1 poin.

**Model 1b — AUC Poly (C=0.01, gamma=0.1)**
Confusion Matrix test: TP=58, FN=33, FP=586, TN=9323. Recall 0.64, precision 0.09 — 91% alarm salah. Paling agresif dengan FP terbanyak.

**Model 2b — F1 Poly (C=0.1, gamma=0.01)**
Confusion Matrix test: TP=36, FN=55, FP=77, TN=9832. Recall 0.40 (terendah), precision 0.32 (paling seimbang). ROC AUC test 0.7601 (terendah secara diskriminasi), namun F1 test 0.3547 justru tertinggi. Gap train-test paling kecil (train F1 0.4237 vs test 0.3547), overfitting paling minimal. Cocok untuk skenario di mana false alarm sangat mahal.

**Model 2c — F1 RBF (C=100, gamma=0.1)**
Confusion Matrix test: TP=16, FN=75, FP=57, TN=9852. Recall 0.18 — 82% fraud terlewat. ROC AUC test 0.7444 vs train 0.9997 (nyaris sempurna). Overfitting paling parah: model menghafal data training tetapi gagal generalisasi.

### Perbandingan dengan jurnal

| Metrik | Jurnal Xia (2022) | Replikasi | Selisih |
|--------|-------------------|-----------|---------|
| AUC test (RBF, C=10, gamma=0.01) | 0.90 | 0.8381 | -0.0619 |
| F1 test (RBF, C=0.8, gamma=0.06) | 0.260 | 0.2483 | -0.0117 |
| AUC train (RBF, C=10, gamma=0.01) | 0.87 | 0.9610 | +0.0910 |

Perbedaan hasil disebabkan oleh perbedaan versi dataset, preprocessing, dan versi scikit-learn. Overfitting terjadi pada replikasi (train AUC 0.961 > test AUC 0.838) sedangkan jurnal tidak (train 0.87 < test 0.90).

## API Endpoints

### GET /health
Cek status API.
```json
{"status": "ok", "model": "dual (auc + f1)"}
```

### POST /predict
Prediksi fraud.

**Request body:**
```json
{
  "amt": 100.0,
  "category": "misc_net",
  "dob": "1990-01-01",
  "buyer_state": "Ohio",
  "merchant_state": "New York",
  "city_pop_segment": "Less Dense"
}
```

**Response:**
```json
{
  "jurnal": {
    "fraud_score": 45.2,
    "is_fraud": false
  },
  "replikasi": {
    "fraud_score": 38.7,
    "is_fraud": false
  }
}
```

## Notes

- Training notebook menggunakan syntax asli Colab (tidak diubah).
- 6 fitur diisi user, 4 fitur sisanya (merchant, zip, job, gender) di-sample random dari training data.
- Tidak perlu geopy — koordinat menggunakan dictionary pusat 50 US states.
- Ensemble 25 samples per prediksi untuk mengurangi dominasi hidden feature sampling.
- predict_proba tidak digunakan — Platt scaling deprecated di sklearn 1.9, diganti decision function + tanh scaling.

## Referensi

Xia, J. (2022). Credit Card Fraud Detection Based on SVM. *Highlights in Science, Engineering and Technology*, Vol.23, hal.93-97. DOI: https://doi.org/10.54097/hset.v23i.3202
