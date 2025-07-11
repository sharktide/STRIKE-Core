import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, models, backend as K
import pandas as pd
import joblib
import numpy as np

@register_keras_serializable()
def cold_temp_penalty(inputs):
    temp = inputs[:, 0]
    penalty = tf.where(
        temp > 295.0,
        1.0,
        tf.where(
            temp < 290.0,
            0.0,
            (temp - 290.0) / 5.0
        )
    )
    return penalty[:, None]

@register_keras_serializable()
def fire_risk_booster(inputs):
    temp = inputs[:, 0]
    humidity = inputs[:, 1]
    wind = inputs[:, 2]
    veg = inputs[:, 3]

    # Boost ranges
    temp_boost = tf.sigmoid((temp - 305.0) * 1.2)
    humidity_boost = tf.sigmoid((20.0 - humidity) * 0.5)
    wind_boost = tf.sigmoid((wind - 15.0) * 0.8)
    veg_boost = tf.sigmoid((veg - 70.0) * 0.5)

    # Combine and scale
    combined = temp_boost * humidity_boost * wind_boost * veg_boost
    boost = 1.0 + 0.3 * combined  # Up to 30% increase in fire score
    return boost[:, None]

@register_keras_serializable()
def fire_suppression_mask(inputs):
    temp = inputs[:, 0]
    humidity = inputs[:, 1]
    wind = inputs[:, 2]

    # Suppress if warm but humid and still
    temp_flag = tf.sigmoid((temp - 293.0) * 1.2)
    humid_flag = tf.sigmoid((humidity - 50.0) * 0.4)
    wind_flag = 1 - tf.sigmoid((wind - 5.0) * 0.8)

    suppression = temp_flag * humid_flag * wind_flag
    penalty = 1.0 - 0.3 * suppression  # Max 30% suppression
    return penalty[:, None]

@register_keras_serializable()
def firetrust_activation(x):
    return 0.5 + tf.sigmoid(x)  # output in range [0.5, 1.5]

model = tf.keras.models.load_model("wildfires.h5", custom_objects={
    'cold_temp_penalty': cold_temp_penalty,
    'fire_risk_booster': fire_risk_booster,
    'fire_suppression_mask': fire_suppression_mask
})

trust_model = tf.keras.models.load_model("FireTrustNet.h5", custom_objects={
    'firetrust_activation': firetrust_activation,
    'mse': tf.keras.losses.MeanSquaredError()
})

trust_scaler = joblib.load("firetrust_scaler.pkl")

# Simulated LA wildfire conditions
la_conditions = pd.DataFrame([{
    'temperature': 315.0,  # ~39°C in Kelvin
    'humidity': 10.0,
    'wind_speed': 4.0,
    'vegetation_index': 0.35,
    'elevation': 800.0
}])

# Predict
def predict_with_trustnet(input_df):
    # Step 1: Base model prediction
    base_prob = model.predict(input_df)[0][0]

    # Step 2: FireTrustNet modulation
    X_scaled = trust_scaler.transform(input_df)
    trust_factor = trust_model.predict(X_scaled)[0][0]

    # Step 3: Adjusted prediction
    adjusted_prob = np.clip(base_prob * trust_factor, 0, 1)
    verdict = "🔥 FIRE LIKELY" if adjusted_prob > 0.5 else "🌿 No Fire"
    return adjusted_prob, verdict

adjusted_prob, risk = predict_with_trustnet(la_conditions)
print(f"Prediction: {risk} | Adjusted Probability: {adjusted_prob:.4f}")

import numpy as np
import matplotlib.pyplot as plt

temps = np.linspace(280, 320, 100)
sweep_df = pd.DataFrame({
    'temperature': temps,
    'humidity': [12.0]*100,
    'wind_speed': [40.0]*100,
    'vegetation_index': [2.0]*100,
    'elevation': [500.0]*100
})


raw_probs = model.predict(sweep_df).flatten()
sweep_scaled = trust_scaler.transform(sweep_df)
trust_mods = trust_model.predict(sweep_scaled).flatten()
adjusted_probs = np.clip(raw_probs * trust_mods, 0, 1)

plt.plot(temps, raw_probs, linestyle='--', label='Base Model', color='gray')
plt.plot(temps, adjusted_probs, label='With FireTrustNet', color='orangered')
plt.axvline(305, linestyle=':', color='red', label='Boost Zone')
plt.xlabel("Temperature (K)")
plt.ylabel("Fire Probability")
plt.title("🔥 Model vs. FireTrustNet Adjustment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
