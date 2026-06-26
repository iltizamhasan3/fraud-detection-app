# Fraud Detection App

Replikasi jurnal Xia (2022) — SVM-based fraud detection dengan FastAPI backend + Streamlit frontend.

## Quick Start

```bash
# Terminal 1 — API
python api.py

# Terminal 2 — GUI
streamlit run gui.py
```

## Stack

- **Backend**: FastAPI (`POST /predict`, `GET /health`)
- **Frontend**: Streamlit (Assessment + Exploration tabs)
- **Model**: SVM RBF (scikit-learn) — dual model: AUC-optimized & F1-optimized
- **Data**: [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle)

## Project Structure

```
├── api.py              # FastAPI backend
├── gui.py              # Streamlit frontend
├── train_models.ipynb  # Training notebook (Colab → local)
├── models/             # Trained artifacts (.pkl) + evaluation charts (.png)
├── data/               # fraudTrain.csv, fraudTest.csv (gitignored)
└── requirements.txt
```

## Notes

- Training notebook menggunakan syntax asli Colab (tidak diubah).
- 6 fitur diisi user, 4 fitur sisanya (merchant, zip, job, gender) di-sample random dari training data.
- Tidak perlu geopy — koordinat menggunakan dictionary pusat 50 US states.
