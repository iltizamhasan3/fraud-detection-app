import streamlit as st
import requests
import pandas as pd
import datetime

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection")
st.markdown("Deteksi fraud transaksi menggunakan **SVM (RBF Kernel)**")

API_URL = "http://localhost:5000"

with st.sidebar:
    st.header("Model Configuration")
    model_type = st.radio(
        "Pilih Model",
        options=["auc", "f1"],
        format_func=lambda x: "AUC-optimized (balanced)" if x == "auc" else "F1-optimized (precision-recall)"
    )
    st.caption("AUC model lebih baik untuk deteksi umum. F1 model lebih baik untuk meminimalkan false positive.")

st.subheader("Input Transaksi")

col1, col2 = st.columns(2)

with col1:
    merchant = st.text_input("Merchant", "fraud_Rippin, Kub and Mann")
    category = st.selectbox("Category", [
        "misc_net", "grocery_pos", "entertainment", "gas_transport",
        "misc_pos", "grocery_net", "shopping_pos", "shopping_net",
        "food_dining", "health_fitness", "personal_care", "kids_pets",
        "travel", "home", "hotel", "restaurant", "bar"
    ])
    amt = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
    gender = st.selectbox("Gender", ["M", "F"])
    zip_code = st.text_input("ZIP Code", "28654")

with col2:
    job = st.text_input("Job", "Engineer")
    dob = st.date_input("Date of Birth", datetime.date(1990, 1, 1))
    lat = st.number_input("Latitude", value=36.0788, format="%.4f")
    long = st.number_input("Longitude", value=-81.1781, format="%.4f")
    merch_lat = st.number_input("Merchant Latitude", value=36.0113, format="%.4f")
    merch_long = st.number_input("Merchant Longitude", value=-82.0483, format="%.4f")
    city_pop = st.number_input("City Population", min_value=0, value=3495, step=1)
    unix_time = st.number_input("Unix Time", value=int(datetime.datetime.now().timestamp()))

if st.button("Prediksi Fraud", type="primary", use_container_width=True):
    payload = {
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "gender": gender,
        "zip": zip_code,
        "job": job,
        "unix_time": unix_time,
        "dob": dob.isoformat(),
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "city_pop": city_pop,
        "model_type": model_type
    }

    with st.spinner("Memproses prediksi..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                st.subheader("Hasil Prediksi")

                if result["is_fraud"]:
                    st.error(f"FRAUD TERDETEKSI!")
                else:
                    st.success(f"TRANSAKSI NORMAL")

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Prediction", "Fraud" if result["is_fraud"] else "Normal")
                col_res2.metric("Probability", f"{result['probability']:.4f}")
                col_res3.metric("Model", result["model_used"])

                st.caption(f"Raw probability of fraud: {result['probability']:.6f}")
            else:
                st.error(f"Error: {resp.json().get('error', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Tidak dapat terhubung ke API. Pastikan Flask sudah berjalan di port 5000.")

st.divider()
st.caption("Proyek Replikasi: Xia, J. (2022) - Credit Card Fraud Detection Based on Support Vector Machine")
