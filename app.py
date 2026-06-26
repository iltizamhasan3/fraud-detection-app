import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model_auc = joblib.load(os.path.join(MODEL_DIR, 'model_auc.pkl'))
model_f1 = joblib.load(os.path.join(MODEL_DIR, 'model_f1.pkl'))
encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
fitur = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
kolom_kategori = joblib.load(os.path.join(MODEL_DIR, 'kolom_kategori.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        input_df['dob'] = pd.to_datetime(input_df['dob'])
        ref_date = pd.Timestamp('2020-12-31')
        input_df['age'] = ((ref_date - input_df['dob']).dt.days // 365).astype(int)

        def segmentasi_kota(pop):
            if pop < 10_000:
                return 'Less Dense'
            elif pop < 100_000:
                return 'Adequately Dense'
            else:
                return 'Densely Populated'

        input_df['city_pop_segment'] = input_df['city_pop'].apply(segmentasi_kota)

        lat_diff = input_df['lat'].values[0] - input_df['merch_lat'].values[0]
        long_diff = input_df['long'].values[0] - input_df['merch_long'].values[0]
        input_df['displacement'] = np.sqrt((lat_diff * 111) ** 2 + (long_diff * 111) ** 2)

        def kategori_lokasi(d):
            if d < 50:
                return 'Nearby'
            elif d < 150:
                return 'Long Distance'
            else:
                return 'Far Away'

        input_df['location'] = input_df['displacement'].apply(kategori_lokasi)

        X = input_df[fitur].copy()
        X[kolom_kategori] = encoder.transform(X[kolom_kategori])
        X_scaled = scaler.transform(X)

        model_type = data.get('model_type', 'auc')
        if model_type == 'f1':
            model = model_f1
        else:
            model = model_auc

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0, 1]

        return jsonify({
            'prediction': int(pred),
            'probability': float(prob),
            'is_fraud': bool(pred == 1),
            'model_used': f"{'F1-optimized' if model_type == 'f1' else 'AUC-optimized'} SVM (rbf)"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
