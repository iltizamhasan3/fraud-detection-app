import os, datetime, random, math
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

app = FastAPI(title="Fraud Detection API")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model_auc = joblib.load(os.path.join(MODEL_DIR, 'model_auc.pkl'))
model_f1 = joblib.load(os.path.join(MODEL_DIR, 'model_f1.pkl'))
encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
fitur = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
kolom_kategori = joblib.load(os.path.join(MODEL_DIR, 'kolom_kategori.pkl'))

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'fraudTrain.csv')
DATA_POOL = None
if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH, nrows=5000)
    DATA_POOL = df_raw[['merchant', 'zip', 'job', 'gender']].dropna().reset_index(drop=True)
    print(f"[API] Loaded {len(DATA_POOL)} rows for hidden feature sampling")

STATE_COORDS = {
    "Alabama": (32.8, -86.9), "Alaska": (61.4, -152.3), "Arizona": (34.2, -111.9),
    "Arkansas": (34.9, -92.4), "California": (36.1, -119.7), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "Florida": (27.7, -81.5),
    "Georgia": (32.6, -83.4), "Hawaii": (21.3, -157.8), "Idaho": (44.2, -114.1),
    "Illinois": (40.0, -89.3), "Indiana": (39.9, -86.2), "Iowa": (41.9, -93.4),
    "Kansas": (38.5, -98.4), "Kentucky": (37.5, -84.8), "Louisiana": (30.9, -92.1),
    "Maine": (44.6, -69.1), "Maryland": (39.0, -76.7), "Massachusetts": (42.2, -71.6),
    "Michigan": (43.3, -84.6), "Minnesota": (45.7, -94.0), "Mississippi": (32.7, -89.7),
    "Missouri": (38.4, -92.5), "Montana": (46.9, -110.4), "Nebraska": (41.5, -99.8),
    "Nevada": (38.5, -116.5), "New Hampshire": (43.7, -71.6), "New Jersey": (40.3, -74.5),
    "New Mexico": (34.4, -106.1), "New York": (42.2, -75.5), "North Carolina": (35.6, -79.4),
    "North Dakota": (47.1, -100.3), "Ohio": (40.3, -82.8), "Oklahoma": (35.6, -97.4),
    "Oregon": (43.9, -120.6), "Pennsylvania": (40.9, -77.8), "Rhode Island": (41.6, -71.5),
    "South Carolina": (33.8, -80.5), "South Dakota": (44.4, -100.2), "Tennessee": (35.8, -86.4),
    "Texas": (31.5, -99.3), "Utah": (39.3, -111.7), "Vermont": (44.0, -72.7),
    "Virginia": (37.5, -78.9), "Washington": (47.3, -120.6), "West Virginia": (38.6, -80.7),
    "Wisconsin": (44.6, -89.8), "Wyoming": (42.8, -107.5),
    "District of Columbia": (38.9, -77.0),
}

class PredictInput(BaseModel):
    amt: float = Field(...)
    category: str = Field("misc_net")
    dob: str = Field("1990-01-01")
    buyer_state: str = Field("New York")
    merchant_state: str = Field("New York")
    city_pop_segment: str = Field("Less Dense")

def _compute_displacement(lat1, lon1, lat2, lon2):
    d = np.sqrt(((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 111) ** 2)
    if d < 50:
        return d, "Nearby"
    elif d < 150:
        return d, "Long Distance"
    return d, "Far Away"

@app.get("/health")
def health():
    return {"status": "ok"}

def _ensemble(model, data, user_lat, user_lon, merch_lat, merch_lon, displacement, location_cat):
    decs = []
    for _ in range(25):
        row = DATA_POOL.sample(1).iloc[0] if DATA_POOL is not None else None
        sm, sz, sj, sg = row if row is not None else ('fraud_Rippin, Kub and Mann', 28654, 'Psychologist, counselling', 'F')

        df = pd.DataFrame([{
            'merchant': sm, 'category': data.category, 'amt': data.amt,
            'gender': sg, 'zip': int(sz), 'job': sj,
            'unix_time': random.randint(1325376000, 1577836800),
            'lat': user_lat, 'long': user_lon,
            'merch_lat': merch_lat, 'merch_long': merch_lon,
            'dob': data.dob,
        }])
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = ((pd.Timestamp('2020-12-31') - df['dob']).dt.days // 365).astype(int)
        df['city_pop_segment'] = data.city_pop_segment
        df['displacement'] = displacement
        df['location'] = location_cat

        X = df[fitur].copy()
        X[kolom_kategori] = encoder.transform(X[kolom_kategori])
        decs.append(model.decision_function(scaler.transform(X))[0])

    avg_dec = np.mean(decs)
    score = round((np.tanh(avg_dec / 2.0) + 1) * 50, 2)
    pred = int(avg_dec > 0)
    return {"fraud_score": score, "prediction": pred, "is_fraud": bool(pred == 1)}

@app.post("/predict")
def predict(data: PredictInput):
    try:
        user_coords = STATE_COORDS.get(data.buyer_state)
        merch_coords = STATE_COORDS.get(data.merchant_state)

        if user_coords is None or merch_coords is None:
            displacement, location_cat = 25.0, "Nearby"
            user_lat, user_lon, merch_lat, merch_lon = 0, 0, 0, 0
        else:
            user_lat, user_lon = user_coords
            merch_lat, merch_lon = merch_coords
            displacement, location_cat = _compute_displacement(user_lat, user_lon, merch_lat, merch_lon)

        return {
            "jurnal": _ensemble(model_auc, data, user_lat, user_lon, merch_lat, merch_lon, displacement, location_cat),
            "replikasi": _ensemble(model_f1, data, user_lat, user_lon, merch_lat, merch_lon, displacement, location_cat),
            "location": location_cat,
            "displacement_km": round(displacement, 1),
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
