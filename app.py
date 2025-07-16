import gradio as gr
import pandas as pd
import numpy as np
import joblib
from load import *
from helper import *
from matplotlib import pyplot as plt
import tensorflow as tf


def predict_fire(temp, temp_unit, humidity, wind, wind_unit, veg, elev, elev_unit, use_trust):
    input_data = {
        "temperature": convert_temperature(temp, temp_unit),
        "humidity": humidity,
        "wind_speed": convert_wind_speed(wind, wind_unit),
        "vegetation_index": veg,
        "elevation": convert_elevation(elev, elev_unit)
    }

    input_df = pd.DataFrame([input_data])
    base_prob = FireNet.predict(input_df)[0][0]
    if use_trust:
        trust_score = FireTrustNet.predict(FireScaler.transform(input_df))[0][0]
        final = np.clip(base_prob * trust_score, 0, 1)
    else:
        final = base_prob
    if final > 0.49:
        verdict = "🔥 FIRE LIKELY"
    elif final > 0.43 and final < 0.50:
        verdict = "⚠️ Fire Possible"
    else:
        verdict = "🛡️ Fire Unlikely"
    return f"{verdict} ({final:.2f})"

def predict_flood(rainfall_val, rainfall_unit, water_level_val, elevation_val, elev_unit,
                  slope_val, distance_val, distance_unit, use_trustnet):
    # Unit conversion
    rainfall = convert_rainfall(rainfall_val, rainfall_unit)
    elevation = convert_elevation(elevation_val, elev_unit)
    distance = convert_distance(distance_val, distance_unit)

    # Construct input for FloodNet
    base_df = pd.DataFrame([{
        "Rainfall": rainfall,
        "Water Level": water_level_val,
        "Elevation": elevation,
        "Slope": slope_val,
        "Distance from River": distance
    }])

    base_prob = FloodNet.predict(base_df)[0][0]

    if use_trustnet:
        trust_df = pd.DataFrame([{
            "rainfall": rainfall,
            "water_level": water_level_val,
            "elevation": elevation,
            "slope": slope_val,
            "distance_from_river": distance
        }])
        trust_score = FloodTrustNet.predict(FloodScaler.transform(trust_df))[0][0]
        final = np.clip(base_prob * trust_score, 0, 1)
    else:
        final = base_prob

    if final > 0.49:
        verdict = "🏞️ FV-FLOOD LIKELY"
    elif final > 0.43 and final < 0.50:
        verdict = "⚠️ FV-Flood Possible"
    else:
        verdict = "🛡️ FV-Flood Unlikely"
    return f"{verdict} ({final:.2f})"

def predict_pluvial_flood(rain, imp, drain, urban, conv, use_trust, rainfall_unit):
    print(rainfall_unit)
    rain = convert_rainfall_intensity(rain, rainfall_unit)
    print(rain)
    input_data = {
        "rainfall_intensity": rain,
        "impervious_ratio": imp,
        "drainage_density": drain,
        "urbanization_index": urban,
        "convergence_index": conv
    }
    input_df = pd.DataFrame([input_data])
    base_prob = PV_FloodNet.predict(input_df)[0][0]

    if use_trust:
        trust_score = PV_FloodTrustNet.predict(PV_FloodScaler.transform(input_df))[0][0]
        final = np.clip(base_prob * trust_score, 0, 1)
    else:
        final = base_prob

    if final > 0.52:
        verdict = "🛶 PV-FLOOD LIKELY"
    elif 0.45 < final <= 0.52:
        verdict = "⚠️ PV-Flood Possible"
    else:
        verdict = "🛡️ PV-Flood Unlikely"

    return f"{verdict} ({final:.2f})"

def predict_flash_flood(rainfall, slope, drainage, saturation, convergence, use_trust):
    input_data = {
        "rainfall_intensity": rainfall,
        "slope": slope,
        "drainage_density": drainage,
        "soil_saturation": saturation,
        "convergence_index": convergence
    }
    input_df = pd.DataFrame([input_data])
    base_pred = FlashFloodNet.predict(input_df)[0][0]
    if use_trust:
        trust_score = FlashFloodTrustNet.predict(FlashFloodScaler.transform(input_df))[0][0]
        adjusted = np.clip(base_pred * trust_score, 0, 1)
    else:
        adjusted = base_pred

    if adjusted > 0.55:
        return f"🌩️ FLASH FLOOD LIKELY ({adjusted:.2f})"
    elif 0.40 < adjusted <= 0.55:
        return f"⚠️ Flash Flood Possible ({adjusted:.2f})"
    else:
        return f"🛡️ Flash Flood Unlikely ({adjusted:.2f})"
    
def predict_quake(dotM0, sdr, coulomb, afd, fsr, use_trust):
    dotM0 = dotM0 * 1e16

    input_data = {
        "seismic_moment_rate": dotM0,
        "surface_displacement_rate": sdr,
        "coulomb_stress_change": coulomb,
        "average_focal_depth": afd,
        "fault_slip_rate": fsr
    }
    input_df = pd.DataFrame([input_data])
    base_pred = QuakeNet.predict(input_df)[0][0]
    if use_trust:
        trust_score = QuakeTrustNet.predict(QuakeTrustScaler.transform(input_df))[0][0]
        adjusted = np.clip(base_pred * trust_score, 0, 1)
    else:
        adjusted = base_pred

    if adjusted > 0.55:
        return f"🌎 EARTHQUAKE LIKELY ({adjusted:.2f})"
    elif 0.40 < adjusted <= 0.55:
        return f"⚠️ Earthquake Possible ({adjusted:.2f})"
    else:
        return f"🛡️ Earthquake Unlikely ({adjusted:.2f})"

def generate_plot(axis, use_trustnet):
    sweep_values = np.linspace({
        "temperature": (280, 320),
        "humidity": (0, 100),
        "wind_speed": (0, 50),
        "vegetation_index": (0.0, 2.0),
        "elevation": (0, 3000)
    }[axis][0], {
        "temperature": (280, 320),
        "humidity": (0, 100),
        "wind_speed": (0, 50),
        "vegetation_index": (0.0, 2.0),
        "elevation": (0, 3000)
    }[axis][1], 100)

    base_input = {
        "temperature": 300.0,
        "humidity": 30.0,
        "wind_speed": 10.0,
        "vegetation_index": 1.0,
        "elevation": 500.0
    }

    sweep_df = pd.DataFrame([{
        **base_input,
        axis: val
    } for val in sweep_values])

    raw_probs = FireNet.predict(sweep_df).flatten()
    if use_trustnet:
        trust_mods = FireTrustNet.predict(FireScaler.transform(sweep_df)).flatten()
        adjusted_probs = np.clip(raw_probs * trust_mods, 0, 1)
    else:
        adjusted_probs = raw_probs

    fig, ax = plt.subplots()
    ax.plot(sweep_values, raw_probs, "--", color="gray", label="Base Model")
    if use_trustnet:
        ax.plot(sweep_values, adjusted_probs, color="orangered", label="With FireTrustNet")
    ax.set_xlabel(axis.replace("_", " ").title())
    ax.set_ylabel("Fire Probability")
    ax.set_title(f"Fire Probability vs. {axis.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True)
    return fig

def generate_flood_plot(axis, use_trustnet):
    sweep_range = {
        "rainfall": (0, 150),
        "water_level": (0, 8000),
        "elevation": (0, 20),
        "slope": (0, 20),
        "distance_from_river": (0, 2000)
    }

    values = np.linspace(*sweep_range[axis], 100)

    base_example = {
        "rainfall": 50.0,
        "water_level": 3000.0,
        "elevation": 5.0,
        "slope": 2.0,
        "distance_from_river": 100.0
    }

    # Build test cases by sweeping one input
    inputs = pd.DataFrame([
        {**base_example, axis: v} for v in values
    ])

    # Predict with FloodNet
    floodnet_inputs = inputs.rename(columns={
        "rainfall": "Rainfall",
        "water_level": "Water Level",
        "elevation": "Elevation",
        "slope": "Slope",
        "distance_from_river": "Distance from River"
    })

    base_probs = FloodNet.predict(floodnet_inputs).flatten()

    if use_trustnet:
        trust_inputs = inputs.copy()
        trust_scores = FloodTrustNet.predict(FloodScaler.transform(trust_inputs)).flatten()
        modulated_probs = np.clip(base_probs * trust_scores, 0, 1)
    else:
        modulated_probs = base_probs

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(values, base_probs, "--", color="gray", label="FloodNet")
    if use_trustnet:
        ax.plot(values, modulated_probs, color="blue", label="With FloodTrustNet")
    ax.set_xlabel(axis.replace("_", " ").title())
    ax.set_ylabel("FV Flood Probability")
    ax.set_title(f"FV Flood Probability vs. {axis.replace('_', ' ').title()}")
    ax.grid(True)
    ax.legend()
    return fig

def generate_pluvial_plot(axis, use_trust):
    sweep_range = {
        "rainfall_intensity": (0, 160),
        "impervious_ratio": (0.0, 1.0),
        "drainage_density": (1.0, 5.0),
        "urbanization_index": (0.0, 1.0),
        "convergence_index": (0.0, 1.0)
    }

    sweep_values = np.linspace(*sweep_range[axis], 100)
    base_input = {
        "rainfall_intensity": 60.0,
        "impervious_ratio": 0.5,
        "drainage_density": 2.5,
        "urbanization_index": 0.6,
        "convergence_index": 0.5
    }

    sweep_df = pd.DataFrame([
        {**base_input, axis: val} for val in sweep_values
    ])

    base_probs = PV_FloodNet.predict(sweep_df).flatten()
    if use_trust:
        trust_mods = PV_FloodTrustNet.predict(PV_FloodScaler.transform(sweep_df)).flatten()
        adjusted = np.clip(base_probs * trust_mods, 0, 1)
    else:
        adjusted = base_probs

    fig, ax = plt.subplots()
    ax.plot(sweep_values, base_probs, "--", color="gray", label="Base Model")
    if use_trust:
        ax.plot(sweep_values, adjusted, color="royalblue", label="With PV-FloodTrustNet")

    ax.set_xlabel(axis.replace("_", " ").title())
    ax.set_ylabel("PV Flood Probability")
    ax.set_title(f"PV Flood Probability vs. {axis.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True)
    return fig

def generate_flash_plot(axis, use_trust):
    sweep_values = np.linspace(
        {"rainfall_intensity": 0, "slope": 0, "drainage_density": 1.0,
         "soil_saturation": 0.3, "convergence_index": 0.0}[axis],
        {"rainfall_intensity": 150, "slope": 30, "drainage_density": 5.0,
         "soil_saturation": 1.0, "convergence_index": 1.0}[axis],
        100
    )
    base_input = {
        "rainfall_intensity": 90,
        "slope": 15,
        "drainage_density": 3.0,
        "soil_saturation": 0.7,
        "convergence_index": 0.5
    }

    sweep_df = pd.DataFrame([{**base_input, axis: val} for val in sweep_values])
    raw_probs = FlashFloodNet.predict(sweep_df).flatten()
    if use_trust:
        trust_mods = FlashFloodTrustNet.predict(FlashFloodScaler.transform(sweep_df)).flatten()
        adjusted_probs = np.clip(raw_probs * trust_mods, 0, 1)
    else:
        adjusted_probs = raw_probs

    fig, ax = plt.subplots()
    ax.plot(sweep_values, raw_probs, "--", color="gray", label="Base Model")
    if use_trust:
        ax.plot(sweep_values, adjusted_probs, color="darkcyan", label="With FlashFloodTrustNet")
    ax.set_xlabel(axis.replace("_", " ").title())
    ax.set_ylabel("Flash Flood Probability")
    ax.set_title(f"Flash Flood Probability vs. {axis.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True)
    return fig

def generate_quake_plot(axis, use_trustnet):

    axis_ranges = {
        "seismic_moment_rate": (5e14, 2.5e16),
        "surface_displacement_rate": (0, 100),
        "coulomb_stress_change": (0, 700),
        "average_focal_depth": (0, 60),
        "fault_slip_rate": (0, 20)
    }
    sweep_values = np.linspace(*axis_ranges[axis], 100)

    # Baseline input for all other features
    base_input = {
        "seismic_moment_rate": 1.5e16,
        "surface_displacement_rate": 35,
        "coulomb_stress_change": 300,
        "average_focal_depth": 18,
        "fault_slip_rate": 7.0
    }

    # Create sweep dataframe
    sweep_df = pd.DataFrame([{**base_input, axis: val} for val in sweep_values])
    raw_preds = QuakeNet.predict(sweep_df).flatten()

    if use_trustnet:
        scaled_df = QuakeTrustScaler.transform(sweep_df)
        trust_scores = QuakeTrustNet.predict(scaled_df).flatten()
        modulated = np.clip(raw_preds * trust_scores, 0, 1)
    else:
        modulated = raw_preds

    # Plot
    fig, ax = plt.subplots()
    ax.plot(sweep_values, raw_preds, "--", color="gray", label="QuakeNet")
    if use_trustnet:
        ax.plot(sweep_values, modulated, color="darkred", label="With QuakeTrustNet")
    ax.set_xlabel(axis.replace("_", " ").title())
    ax.set_ylabel("Quake Probability")
    ax.set_title(f"Earthquake Likelihood vs. {axis.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True)

    return fig

with gr.Blocks(theme=gr.themes.Default(), css=".tab-nav-button { font-size: 1.1rem !important; padding: 0.8em; } ") as demo:
    gr.Markdown("# ClimateNet - A family of tabular classification models to predict natural disasters")

    with gr.Tab("🔥Wildfires"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    temp = gr.Slider(280, 330, value=300, label="Temperature (K)")
                    temp_unit = gr.Dropdown(["K", "°C", "°F"], value="K", label="", scale=0.2)

                temp_unit.change(fn=update_temp_slider, inputs=temp_unit, outputs=temp)

                with gr.Row():
                    humidity = gr.Slider(0, 100, value=30, label="Humidity (%)")
                    gr.Dropdown(["%"], value="%", label="", scale=0.1)

                with gr.Row():
                    wind_speed = gr.Slider(0, 50, value=10, label="Wind Speed (m/s)")
                    wind_unit = gr.Dropdown(["m/s", "km/h", "mp/h"], value="m/s", label="", scale=0.2)

                wind_unit.change(update_wind_slider, inputs=wind_unit, outputs=wind_speed)

                with gr.Row():
                    elevation = gr.Slider(0, 3000, value=500, label="Elevation (m)")
                    elev_unit = gr.Dropdown(["m", "ft"], value="m", label="", scale=0.2)

                elev_unit.change(update_elevation_slider, inputs=elev_unit, outputs=elevation)

                with gr.Row():
                    vegetation_index = gr.Slider(0.0, 2.0, value=1.0, label="Vegetation Index (NDVI)")
                    gr.Dropdown(["NDVI"], value="NDVI", label="", scale=0.2)
                use_trust = gr.Checkbox(label="Use FireTrustNet", value=True)
                sweep_axis = gr.Radio(["temperature", "humidity", "wind_speed", "vegetation_index", "elevation"], 
                                      label="Sweep Axis", value="temperature")
                predict_btn = gr.Button("Predict")
            with gr.Column():
                with gr.Accordion("ℹ️ Feature Definitions", open=False):
                    gr.Markdown("""
                **Temperaure:** Current Temperature

                **Humidity:** Current Humidity

                **Wind Speed:** Current Wind Speed

                **Elevation:** Current Elevation Relative to Sea Level

                **Vegitation Index:** Your area's NDVI score.
                    """)
                output = gr.Textbox(label="Wildfire Risk Verdict")
                plot_output = gr.Plot(label="Trust Modulation Plot")

    predict_btn.click(
        fn=lambda t, tu, h, w, wu, v, e, eu, trust, axis: (
            predict_fire(t, tu, h, w, wu, v, e, eu, trust),
            generate_plot(axis, trust)
        ),
        inputs=[
            temp, temp_unit,
            humidity,
            wind_speed, wind_unit,
            vegetation_index,
            elevation, elev_unit,
            use_trust,
            sweep_axis
        ],
        outputs=[output, plot_output]
    )

    with gr.Tab("🌊 Fluvial Floods"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    rainfall = gr.Slider(0, 200, value=50, label="Rainfall (mm)")
                    rainfall_unit = gr.Dropdown(["mm", "in"], value="mm", label="", scale=0.2)

                with gr.Row():
                    water_level = gr.Slider(0, 8000, value=3000, label="Relative Water Level (mm)")
                    gr.Dropdown(["mm"], value="mm", label="", scale=0.2)

                with gr.Row():
                    elevation_flood = gr.Slider(0, 20, value=5, label="Relative Elevation (m)")
                    elev_flood_unit = gr.Dropdown(["m", "ft"], value="m", label="", scale=0.2)

                with gr.Row():
                    slope = gr.Slider(0.0, 20.0, value=2.0, label="Slope (°)")
                    gr.Dropdown(["°"], label="",scale=0.2)
                with gr.Row():
                    distance = gr.Slider(0, 2000, value=100, label="Distance from River (m)")
                    distance_unit = gr.Dropdown(["m", "ft"], value="m", label="", scale=0.2)

                elev_flood_unit.change(update_flood_elevation_slider, inputs=elev_flood_unit, outputs=elevation_flood)
                distance_unit.change(update_flood_distance_slider, inputs=distance_unit, outputs=distance)
                rainfall_unit.change(update_flood_rainfall_slider, inputs=rainfall_unit, outputs=rainfall)
                use_trust_flood = gr.Checkbox(label="Use FV-FloodTrustNet", value=True)

                flood_sweep_axis = gr.Radio(
                    ["rainfall", "water_level", "elevation", "slope", "distance_from_river"],
                    label="Sweep Axis", value="rainfall"
                )

                predict_btn_flood = gr.Button("Predict")
            
            with gr.Column():
                with gr.Accordion("ℹ️ Feature Definitions", open=False):
                    gr.Markdown("""
                **Rainfall:** Total recent precipitation - Last 24 hours.

                **Relative Water Level:** Height of river assuming river is 2.5m (8.202 ft) deep. Adjust accordingly.

                **Relative Elevation:** Ground height relative to nearest body of water (river).

                **Slope:** Terrain gradient measured in degrees.

                **Distance from River:** Horizontal distance from riverbed in meters. This does not account for levees or terrain barriers.
                    """)

                flood_output = gr.Textbox(label="FV-Flood Risk Verdict")
                flood_plot = gr.Plot(label="Trust Modulation Plot")

    predict_btn_flood.click(
    fn=lambda r, ru, wl, e, eu, s, d, du, trust, axis: (
        predict_flood(r, ru, wl, e, eu, s, d, du, trust),
        generate_flood_plot(axis, trust)
    ),
    inputs=[
        rainfall, rainfall_unit,
        water_level,
        elevation_flood, elev_flood_unit,
        slope,
        distance, distance_unit,
        use_trust_flood,
        flood_sweep_axis
    ],
    outputs=[flood_output, flood_plot]
    )
    with gr.Tab("🌧️ Pluvial Floods"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    rain_input = gr.Slider(0, 150, value=12, label="Rainfall Intensity (mm/hr)")
                    rain_unit_dropdown = gr.Dropdown(["mm/hr", "in/hr"], value="mm/hr", label="", scale=0.2)
                with gr.Row():
                    imp_input = gr.Slider(0.0, 1.0, value=0.5, label="Impervious Ratio")
                    gr.Dropdown(["ISR"], value="ISR", label="", scale=0.2)
                with gr.Row():
                    drain_input = gr.Slider(1.0, 5.0, value=2.5, label="Drainage Density")
                    gr.Dropdown(["L/A"], value="L/A", label="", scale=0.2)
                with gr.Row():
                    urban_input = gr.Slider(0.0, 1.0, value=0.6, label="Urbanization Index")
                    gr.Dropdown(["uP/tP"], value="uP/tP", label="", scale=0.2)
                with gr.Row():
                    conv_input = gr.Slider(0.0, 1.0, value=0.5, label="Convergence Index")
                    gr.Dropdown(["CI"], value="CI", label="", scale=0.2)
                rain_unit_dropdown.change(update_rain_slider, inputs=rain_unit_dropdown, outputs=rain_input)
                use_trust_pv = gr.Checkbox(label="Use PV-FloodTrustNet", value=True)
                pv_sweep_axis = gr.Radio(
                    ["rainfall_intensity", "impervious_ratio", "drainage_density", "urbanization_index", "convergence_index"],
                    label="Sweep Axis", value="rainfall_intensity"
                )
                pv_predict_btn = gr.Button("Predict")

            with gr.Column():
                with gr.Accordion("ℹ️ Feature Definitions", open=False):
                    gr.Markdown("""
    **Rainfall Intensity:** Recent precipitation rate, typically measured in mm/hr.

    **Impervious Ratio:** Proportion of surface area that cannot absorb water.

    **Drainage Density:** Drainage channel length per unit area.

    **Urbanization Index:** Estimate of built-up density and infrastructure pressure.

    **Convergence Index:** Terrain feature promoting water pooling or runoff directionality.
                    """)
                pv_output = gr.Textbox(label="PV-Flood Risk Verdict")
                pv_plot = gr.Plot(label="Trust Modulation Plot")

        pv_predict_btn.click(
            fn=lambda r, ra, i, d, u, c, trust, axis: (
                predict_pluvial_flood(r, i, d, u, c, trust, ra),
                generate_pluvial_plot(axis, trust)
            ),
            inputs=[rain_input, rain_unit_dropdown, imp_input, drain_input, urban_input, conv_input, use_trust_pv, pv_sweep_axis],
            outputs=[pv_output, pv_plot]
        )

    with gr.Tab("🌩️ Flash Floods"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    rainfall_intensity = gr.Slider(0, 150, value=12, label="Rainfall Intensity (mm/hr)")
                    rainfall_unit_dropdown = gr.Dropdown(["mm/hr", "in/hr"], value="mm/hr", label="", scale=0.2)
                with gr.Row():
                    slope_input = gr.Slider(0, 30, value=15, label="Slope (°)")
                    gr.Dropdown(["°"], label="", scale=0.1)

                with gr.Row():
                    drainage_input = gr.Slider(1.0, 5.0, value=3.0, label="Drainage Density")
                    gr.Dropdown(["L/A"], value="L/A", label="", scale=0.2)

                with gr.Row():
                    saturation_input = gr.Slider(0.3, 1.0, value=0.7, label="Soil Saturation")
                    gr.Dropdown(["VWC"], value="VWC", label="", scale=0.2)

                with gr.Row():
                    convergence_input = gr.Slider(0.0, 1.0, value=0.5, label="Convergence Index")
                    gr.Dropdown(["CI"], value="CI", label="", scale=0.2)

                rainfall_unit_dropdown.change(update_rain_slider, inputs=rainfall_unit_dropdown, outputs=rainfall_intensity)

                use_trust_flash = gr.Checkbox(label="Use FlashFloodTrustNet", value=True)

                flash_sweep_axis = gr.Radio(
                    ["rainfall_intensity", "slope", "drainage_density", "soil_saturation", "convergence_index"],
                    label="Sweep Axis", value="rainfall_intensity"
                )

                flash_predict_btn = gr.Button("Predict")

            with gr.Column():
                with gr.Accordion("ℹ️ Feature Definitions", open=False):
                    gr.Markdown("""
    **Rainfall Intensity:** Measured in mm/hr or in/hr.

    **Slope:** Terrain gradient in degrees.

    **Drainage Density:** Total stream length per unit area.

    **Soil Saturation:** Volumetric water content — higher values = wetter ground.

    **Convergence Index:** Measures topographical tendency to channel runoff.
    """)
                flash_output = gr.Textbox(label="Flash Flood Risk Verdict")
                flash_plot = gr.Plot(label="Trust Modulation Plot")

        flash_predict_btn.click(
            fn=lambda r, ru, s, d, ss, c, trust, axis: (
                predict_flash_flood(convert_rainfall_intensity(r, ru), s, d, ss, c, trust),
                generate_flash_plot(axis, trust)
            ),
            inputs=[
                rainfall_intensity, rainfall_unit_dropdown,
                slope_input, drainage_input, saturation_input,
                convergence_input, use_trust_flash, flash_sweep_axis
            ],
            outputs=[flash_output, flash_plot]
        )

    with gr.Tab("🌎 Earthquakes"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    moment_input = gr.Slider(
                        minimum=0.5, maximum=25.0, value=15.0,
                        label="Seismic Moment Rate (×10¹⁶ Nm/s)"
                    )
                    gr.Dropdown(["Nm/s"], value="Nm/s", label="", scale=0.2)

                with gr.Row():
                    displacement_input = gr.Slider(0, 100, value=35, label="Surface Displacement Rate (mm/yr)")
                    gr.Dropdown(["mm/yr"], value="mm/yr", label="", scale=0.2)

                with gr.Row():
                    stress_input = gr.Slider(0, 700, value=300, label="Coulomb Stress Change (Pa)")
                    gr.Dropdown(["Pa"], value="Pa", label="", scale=0.2)

                with gr.Row():
                    depth_input = gr.Slider(0, 60, value=18, label="Average Focal Depth (km)")
                    gr.Dropdown(["km"], value="km", label="", scale=0.2)

                with gr.Row():
                    slip_input = gr.Slider(0, 20, value=7.0, label="Fault Slip Rate (mm/yr)")
                    gr.Dropdown(["mm/yr"], value="mm/yr", label="", scale=0.2)

                use_trust_quake = gr.Checkbox(label="Use QuakeTrustNet", value=True)

                quake_sweep_axis = gr.Radio(
                    ["seismic_moment_rate", "surface_displacement_rate",
                    "coulomb_stress_change", "average_focal_depth", "fault_slip_rate"],
                    label="Sweep Axis", value="seismic_moment_rate"
                )

                quake_predict_btn = gr.Button("Predict")

            with gr.Column():
                with gr.Accordion("ℹ️ Feature Definitions", open=False):
                    gr.Markdown("""
    **Seismic Moment Rate (Nm/s):** Total energy release rate from seismic events.

    **Surface Displacement Rate (mm/yr):** Horizontal/vertical ground motion observed via GPS/InSAR.

    **Coulomb Stress Change (Pa):** Fault stress changes post-seismic activity.

    **Average Focal Depth (km):** Typical depth of earthquakes in the region.

    **Fault Slip Rate (mm/yr):** Long-term fault motion due to tectonic loading.
                    """)

                quake_output = gr.Textbox(label="Earthquake Risk Verdict")
                quake_plot = gr.Plot(label="Trust Modulation Plot")

        quake_predict_btn.click(
            fn=lambda m, d, s, dp, sl, trust, axis: (
                predict_quake(m, d, s, dp, sl, trust),
                generate_quake_plot(axis, trust)
            ),
            inputs=[
                moment_input, displacement_input,
                stress_input, depth_input, slip_input,
                use_trust_quake, quake_sweep_axis
            ],
            outputs=[quake_output, quake_plot]
        )


demo.launch(share=False)