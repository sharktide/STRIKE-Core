# floodtrustnet_test.py
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, models, backend as K
import pandas as pd
import joblib
import numpy as np


@register_keras_serializable()
def rainfall_proximity_penalty(inputs):
    rainfall = inputs[:, 0]
    distance = inputs[:, 4]
    # Flash-risk zone: high rainfall + near river → penalty low = risky
    proximity_score = tf.sigmoid((150 - distance) * 0.04)
    rainfall_score = tf.sigmoid((rainfall - 90) * 0.3)
    penalty = rainfall_score * proximity_score
    return penalty[:, None] 

@register_keras_serializable()
def flood_risk_booster(inputs):
    slope = inputs[:, 3]
    rainfall = inputs[:, 0]
    # Steep slope + rain boosts alert
    slope_boost = tf.sigmoid((slope - 2.0) * 1.5)
    rain_boost = tf.sigmoid((rainfall - 60) * 0.25)
    combined = slope_boost * rain_boost
    return 1.0 + 0.25 * combined[:, None]

@register_keras_serializable()
def flood_suppression_mask(inputs):
    elevation = inputs[:, 2]
    rainfall = inputs[:, 0]
    # Flat & dry areas suppress score (high elevation, low rain)
    flatness = tf.sigmoid((elevation - 9.0) * 0.6)
    dryness = tf.sigmoid((20.0 - rainfall) * 0.2)
    suppression = flatness * dryness
    return 1.0 - 0.3 * suppression[:, None]

@register_keras_serializable()
def floodtrust_activation(x):
    return 0.5 + tf.sigmoid(x)

base_model = tf.keras.models.load_model("artifacts/floodnet_model.h5", custom_objects={
    'rainfall_proximity_penalty': rainfall_proximity_penalty,
    'flood_risk_booster': flood_risk_booster,
    'flood_suppression_mask': flood_suppression_mask
})

modulator = tf.keras.models.load_model("models/FloodTrustNet.h5", custom_objects={
    'floodtrust_activation': floodtrust_activation,
    'mse': tf.keras.losses.MeanSquaredError()
})

scaler = joblib.load("models/floodtrust_scaler.pkl")

# Canonical feature lists
floodnet_features = ['Rainfall', 'Water Level', 'Elevation', 'Slope', 'Distance from River']
trustnet_features = ['rainfall', 'water_level', 'elevation', 'slope', 'distance_from_river']

# Define your scenario once
scenario_base = {
    'Rainfall': 3.0,
    'Water Level': 600.0,
    'Elevation': 18.0,
    'Slope': 12.0,
    'Distance from River': 1200.0
}

# 1️⃣ Prepare input for FloodNet
base_features = ['Rainfall', 'Water Level', 'Elevation', 'Slope', 'Distance from River']
trust_features = ['rainfall', 'water_level', 'elevation', 'slope', 'distance_from_river']

# Example input
conditions = pd.DataFrame([{
    'Rainfall': 3.0,
    'Water Level': 600.0,
    'Elevation': 18.0,
    'Slope': 12.0,
    'Distance from River': 1200.0
}])

# === 4. Predict with Both Models
def predict_modulated(input_df):
    base_prob = base_model.predict(input_df)[0][0]

    trust_df = pd.DataFrame([{
        trust_features[i]: input_df.iloc[0][base_features[i]]
        for i in range(len(base_features))
    }])

    trust_score = modulator.predict(scaler.transform(trust_df))[0][0]
    final_prob = base_prob * trust_score

    prediction = "🌊 Flood" if final_prob >= 0.5 else "✅ No Flood"

    print(f"FloodNet: {base_prob:.3f} | TrustNet: ×{trust_score:.3f} → Final: {final_prob:.3f} → {prediction}")


# 5️⃣ Run it

conditions = pd.DataFrame([{ 
    'Rainfall': 3.0,
    'Water Level': 600.0,
    'Elevation': 18.0,
    'Slope': 12.0,
    'Distance from River': 1200.0
}])

def predict_with_trustnet(input_df): # Step 1: Base model prediction 
    base_prob = base_model.predict(input_df)[0][0] 
    print(base_prob) 
    return base_prob


def predict_flood(use_trustnet):
    base_Df = pd.DataFrame([{ 
        'Rainfall': 3.0,
        'Water Level': 600.0,
        'Elevation': 18.0,
        'Slope': 12.0,
        'Distance from River': 1200.0
    }])

    scenario = pd.DataFrame([{ 'rainfall': 3.0, 'water_level': 600.0, 'elevation': 18.0, 'slope': 12.0, 'distance_from_river': 1200.0 }])

    base_prob = base_model.predict(base_Df)[0][0]
    if use_trustnet:
        trust_factor = modulator.predict(scaler.transform(scenario))[0][0]
        adjusted_prob = np.clip(base_prob * trust_factor, 0, 1)
    else:
        adjusted_prob = base_prob

    print(adjusted_prob)

predict_flood(True)