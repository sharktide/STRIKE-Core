import numpy as np
import pandas as pd

la_conditions = pd.DataFrame([{
    'temperature': 310.0,
    'humidity': 10.0,
    'wind_speed': 6.0,
    'vegetation_index': 0.8,
    'elevation': 1500.0
}])

input_array = la_conditions.to_numpy().astype(np.float32)

import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: input_array})[0]
adjusted_prob = result[0][0]
verdict = "🔥 FIRE LIKELY" if adjusted_prob > 0.5 else "🌿 No Fire"
print(f"Prediction: {verdict} | Adjusted Probability: {adjusted_prob:.4f}")
