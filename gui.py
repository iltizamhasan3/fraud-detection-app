import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://localhost:5000/predict"
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Fraud Detection System")
st.markdown("SVM-based fraud risk assessment — Xia (2022)")

tab1, tab2 = st.tabs(["Assessment", "Exploration"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction")
        amt = st.number_input("Amount ($)", min_value=0.01, value=100.0)
        category = st.selectbox("Category", [
            "misc_net", "grocery_pos", "entertainment", "gas_transport",
            "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
            "food_dining", "personal_care", "health_fitness", "travel",
            "kids_pets", "home"
        ])

        st.subheader("Customer")
        dob = st.text_input("Date of Birth", "1990-01-01")

    STATES = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
        "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas",
        "Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota",
        "Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey",
        "New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
        "Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas",
        "Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming",
        "District of Columbia"]

    st.subheader("Location")
    mc, uc = st.columns(2)
    with mc:
        merchant_state = st.selectbox("Merchant State", STATES, index=32)
    with uc:
        buyer_state = st.selectbox("Buyer State", STATES, index=32)
    density = st.selectbox("City Density", ["Less Dense", "Adequately Dense", "Densely Populated"])

    if st.button("Run Assessment", type="primary"):
        with st.spinner("Evaluating..."):
            try:
                r = requests.post(API_URL, json={
                    "amt": amt, "category": category,
                    "dob": dob,
                    "buyer_state": buyer_state,
                    "merchant_state": merchant_state,
                    "city_pop_segment": density,
                }, timeout=30)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.session_state["result"] = data
            except Exception as e:
                st.error(f"Connection error: {e}")

    with col2:
        st.subheader("Result")
        with st.container(border=True):
            if "result" in st.session_state:
                data = st.session_state["result"]
                # Support both old (single model) and new (dual model) response format
                if "jurnal" in data:
                    models = [(data["jurnal"], "Model Jurnal"), (data["replikasi"], "Model Replikasi")]
                else:
                    name = data.get("model_used", "Model")
                    models = [(data, name)]
                d1, d2 = st.columns(2) if len(models) > 1 else (st.columns(1) * 2)
                for col, (r, label) in zip([d1, d2], models):
                    with col:
                        st.write(f"**{label}**")
                        if r["is_fraud"]:
                            st.error("Fraud", icon="🚨")
                        else:
                            st.success("Safe", icon="✅")
                        st.metric("Fraud Score", f"{r['fraud_score']:.1f}/100")
                        st.progress(r["fraud_score"] / 100)
                        st.caption(f"Prediction: {'Fraud' if r['is_fraud'] else 'Normal'}")
                st.divider()
                st.write(f"**Transaction:** {data['location']} ({data['displacement_km']} km)")
            else:
                st.info("Run an assessment to see results")

with tab2:
    st.subheader("Dataset Overview")

    @st.cache_data
    def load_data():
        train = pd.read_csv(os.path.join(DATA_DIR, 'fraudTrain.csv'))
        test = pd.read_csv(os.path.join(DATA_DIR, 'fraudTest.csv'))
        df = pd.concat([train, test], ignore_index=True).head(100000)
        df['dob'] = pd.to_datetime(df['dob'])
        ref = pd.Timestamp('2020-12-31')
        df['age'] = ((ref - df['dob']).dt.days // 365).astype(int)
        return df

    with st.spinner("Loading dataset..."):
        df = load_data()

    total = len(df)
    fraud_count = df['is_fraud'].sum()
    normal_count = total - fraud_count
    fraud_pct = fraud_count / total * 100

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Transactions", f"{total:,}")
    col_b.metric("Normal", f"{normal_count:,}", f"{100-fraud_pct:.2f}%")
    col_c.metric("Fraud", f"{fraud_count:,}", f"{fraud_pct:.2f}%")
    col_d.metric("Fraud Ratio", f"1:{normal_count//fraud_count}")

    st.subheader("Class Distribution")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = ['#4CAF50', '#F44336']
    labels = ['Normal (0)', 'Fraud (1)']
    counts = [normal_count, fraud_count]
    axes[0].bar(labels, counts, color=colors, width=0.4)
    axes[0].set_title("Transaction Count")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 200, f"{v:,}", ha='center', fontweight='bold')
    axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    axes[1].set_title("Proportion")
    axes[2].bar(labels, [100-fraud_pct, fraud_pct], color=colors, width=0.4)
    axes[2].set_title("Percentage (%)")
    axes[2].set_ylabel("%")
    for i, v in enumerate([100-fraud_pct, fraud_pct]):
        axes[2].text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Fraud Rate by Category")
    cat_rate = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(cat_rate)), cat_rate.values * 100, color='#F44336', alpha=0.7)
    ax.set_yticks(range(len(cat_rate)))
    ax.set_yticklabels(cat_rate.index)
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Transaction Category")
    for bar, val in zip(bars, cat_rate.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val*100:.2f}%", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Amount Distribution: Normal vs Fraud")
    fig, ax = plt.subplots(figsize=(10, 4))
    normal_amt = df[df['is_fraud'] == 0]['amt']
    fraud_amt = df[df['is_fraud'] == 1]['amt']
    ax.hist(normal_amt, bins=80, alpha=0.6, label='Normal', color='#4CAF50', range=(0, 1000))
    ax.hist(fraud_amt, bins=80, alpha=0.6, label='Fraud', color='#F44336', range=(0, 1000))
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Transaction Amount Distribution (0–1000)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.subheader("Age Distribution: Normal vs Fraud")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[df['is_fraud'] == 0]['age'], bins=50, alpha=0.6, label='Normal', color='#4CAF50')
    ax.hist(df[df['is_fraud'] == 1]['age'], bins=50, alpha=0.6, label='Fraud', color='#F44336')
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Correlation with Fraud")
    df_eda = df.copy()
    df_eda['gender_num'] = (df_eda['gender'] == 'M').astype(int)
    cat_cols = ['category', 'merchant', 'job', 'location', 'city_pop_segment']
    for col in cat_cols:
        if col in df_eda.columns:
            df_eda[col + '_code'] = df_eda[col].astype('category').cat.codes
    numeric_cols = ['amt', 'age', 'unix_time', 'city_pop'] + [c + '_code' for c in cat_cols if c in df_eda.columns]
    numeric_cols = [c for c in numeric_cols if c in df_eda.columns]
    corr = df_eda[numeric_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud').sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_corr = ['#F44336' if v < 0 else '#4CAF50' for v in corr.values]
    ax.barh(range(len(corr)), corr.values, color=colors_corr, alpha=0.7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels([c.replace('_code', '') for c in corr.index])
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("Correlation with Fraud")
    ax.set_title("Feature Correlation with Target (is_fraud)")
    for i, v in enumerate(corr.values):
        ax.text(v + 0.001 if v >= 0 else v - 0.035, i, f"{v:.4f}", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Model Performance (from Training)")
    st.markdown("""
    | Metric | AUC Model | F1 Model |
    |---|---|---|
    | Validation AUC | 0.8432 | — |
    | Validation F1 | — | 0.2312 |
    | Kernel | poly | rbf |
    | C | 1 | 1 |
    | class_weight | balanced | balanced |
    """)
    st.caption("Full training results available in train_models.ipynb — STEP 5 & STEP 6")
