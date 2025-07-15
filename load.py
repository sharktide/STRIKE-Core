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

PV_FloodNet = tf.keras.models.load_model("models/PV-FloodNet.h5", custom_objects={
    'convergence_suppressor': convergence_suppressor,
    'drainage_penalty': drainage_penalty,
    'surface_runoff_amplifier': surface_runoff_amplifier,
    'clip_modulation': clip_modulation,
})

PV_FloodTrustNet = tf.keras.models.load_model("models/PV-FloodTrustNet.h5", custom_objects={
    'floodtrust_activation': floodtrust_activation,
    'mse': tf.keras.losses.MeanSquaredError()
})

PV_FloodScaler = joblib.load("models/PV-floodtrust_scaler.pkl")

FlashFloodNet = tf.keras.models.load_model("models/FlashFloodNet.h5", custom_objects={
    'drainage_penalty': drainage_penalty,
    'intensity_slope_amplifier': intensity_slope_amplifier,
    'convergence_suppressor': convergence_suppressor,
    'clip_modulation': clip_modulation
})

FlashFloodTrustNet = tf.keras.models.load_model("models/FlashFloodTrustNet.h5", custom_objects={
    'mse': tf.keras.losses.MeanSquaredError(),
    'trust_activation': trust_activation
})

FlashFloodScaler = joblib.load("models/flashFloodtrustscaler.pkl")

FireNet.summary()
FireTrustNet.summary()
FloodNet.summary()
FloodTrustNet.summary()
PV_FloodNet.summary()