from objects import *
import joblib

FireNet = tf.keras.models.load_model("models/FireNet.h5", custom_objects={
    "cold_temp_penalty": cold_temp_penalty,
    "fire_risk_booster": fire_risk_booster,
    "fire_suppression_mask": fire_suppression_mask
})

FireTrustNet = tf.keras.models.load_model("models/FireTrustNet.h5", custom_objects={
    "firetrust_activation": firetrust_activation,
    "mse": tf.keras.losses.MeanSquaredError()
})

FireScaler = joblib.load("models/firetrust_scaler.pkl")

FloodNet = tf.keras.models.load_model("models/FV-FloodNet.h5", custom_objects={
    "rainfall_proximity_penalty": rainfall_proximity_penalty,
    "flood_risk_booster": flood_risk_booster,
    "flood_suppression_mask": flood_suppression_mask
})

FloodTrustNet = tf.keras.models.load_model("models/FV-FloodTrustNet.h5", custom_objects={
    "floodtrust_activation": floodtrust_activation,
    "mse": tf.keras.losses.MeanSquaredError()
})

FloodScaler = joblib.load("models/FV-floodtrust_scaler.pkl")