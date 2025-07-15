import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Registered TrustNet Modulation Functions ---
@register_keras_serializable()
def surface_runoff_amplifier(inputs):
    rain = inputs[:, 0]
    impervious = inputs[:, 1]
    return (1.0 + 0.5 * tf.sigmoid((rain - 30) * 0.1) * tf.sigmoid((impervious - 0.6) * 10))[:, None]

@register_keras_serializable()
def drainage_penalty(inputs):
    dd = inputs[:, 2]
    return (1.0 - 0.4 * tf.sigmoid((dd - 3.5) * 2))[:, None]

@register_keras_serializable()
def convergence_suppressor(inputs):
    ci = inputs[:, 4]
    return (1.0 + 0.3 * tf.sigmoid((ci - 0.5) * 8))[:, None]

@register_keras_serializable()
def clip_modulation(x):
    return tf.clip_by_value(x, 0.7, 1.3)

# --- Load PluvialNet Model ---
model = tf.keras.models.load_model("models/PL-PluvialNet-3.h5", custom_objects={
    "clip_modulation": clip_modulation,
    'surface_runoff_amplifier': surface_runoff_amplifier,
    'drainage_penalty': drainage_penalty,
    'convergence_suppressor': convergence_suppressor
})

# --- Prediction Function ---
def predict_pluvial(sample):
    sample = pd.DataFrame([sample], columns=[
        "rainfall_intensity", "impervious_ratio", "drainage_density",
        "urbanization_index", "convergence_index"
    ])
    prob = model.predict(sample)[0][0]
    risk = "🌊 FLOOD LIKELY" if prob > 0.5 else "✅ No Flood"
    return prob, risk

# --- Example Test Input ---
la_sample = [42.0, 0.68, 1.6, 0.85, 0.73]
prob, risk = predict_pluvial(la_sample)
print(f"{risk} | Adjusted Probability: {prob:.4f}")

def build_feature_sweep(feature_name, sweep_vals, fixed_vals):
    """
    Constructs a DataFrame by sweeping one feature and fixing the others.
    
    feature_name: str — name of the feature to sweep
    sweep_vals: list or np.array — values to sweep through
    fixed_vals: dict — fixed values for the remaining features
    """
    data = {f: [fixed_vals[f]] * len(sweep_vals) for f in fixed_vals}
    data[feature_name] = sweep_vals
    return pd.DataFrame(data)

# --- Sensitivity Sweep Plot (e.g., Rainfall Intensity) ---
sweep_vals = np.linspace(0, 100, 100)

fixed_conditions = {
    "rainfall_intensity": 50.0,
    "impervious_ratio": 0.7,
    "drainage_density": 2.0,
    "urbanization_index": 0.8,
    "convergence_index": 0.75
}

sweep_df = build_feature_sweep("urbanization_index", sweep_vals, fixed_conditions)


sweep_probs = model.predict(sweep_df).flatten()

plt.figure(figsize=(8, 5))
plt.plot(sweep_vals, sweep_probs, label="PluvialNet", color='blue')
plt.xlabel("Rainfall Intensity (mm/hr)")
plt.ylabel("Predicted Flood Probability")
plt.title("📈 Flood Risk vs. Rainfall Intensity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Min prob: {sweep_probs.min():.4f}, Max prob: {sweep_probs.max():.4f}")
