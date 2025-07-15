import gradio as gr

def convert_temperature(value, unit):
    return value if unit == "K" else (value + 273.15 if unit == "°C" else (value - 32) * 5/9 + 273.15)

def convert_wind_speed(value, unit):
    return value if unit == "m/s" else (value / 3.6 if unit == "km/h" else value * 0.44704)

def convert_elevation(value, unit):
    return value if unit == "m" else value * 0.3048

def convert_rainfall(val, unit):
    return val if unit == "mm" else val * 25.4

def convert_elevation(val, unit):
    return val if unit == "m" else val * 0.3048

def convert_distance(val, unit):
    return val if unit == "m" else val * 0.3048

def convert_rainfall_intensity(val, unit):
    return val if unit == "mm/hr" else val * 25.4
                                
                                
def update_temp_slider(unit):
    if unit == "K":
        return gr.update(minimum=280, maximum=330, value=300, label="Temperature (K)")
    elif unit == "°C":
        return gr.update(minimum=5, maximum=60, value=25, label="Temperature (°C)")
    elif unit == "°F":
        return gr.update(minimum=40, maximum=140, value=80, label="Temperature (°F)")

def update_wind_slider(unit):
    if unit == "m/s":
        return gr.update(minimum=0, maximum=50, value=10, label="Wind Speed (m/s)")
    elif unit == "km/h":
        return gr.update(minimum=0, maximum=180, value=36, label="Wind Speed (km/h)")
    elif unit == "mp/h":
        return gr.update(minimum=0, maximum=110, value=22, label="Wind Speed (mp/h)")

def update_elevation_slider(unit):
    if unit == "m":
        return gr.update(minimum=0, maximum=3000, value=500, label="Elevation (m)")
    elif unit == "ft":
        return gr.update(minimum=0, maximum=10000, value=1600, label="Elevation (ft)")

def update_flood_elevation_slider(unit):
    if unit == "m":
        return gr.update(minimum=0, maximum=20, value=5, label="Relative Elevation (m)")
    elif unit == "ft":
        return gr.update(minimum=0, maximum=60, value=15, label="Relative Elevation (ft)")
    
def update_flood_distance_slider(unit):
    if unit == "m":
        return gr.update(minimum=0, maximum=2000, value=100, label="Distance from River (m)")
    elif unit == "ft":
        return gr.update(minimum=0, maximum=6000, value=300, label="Distance from River (ft)")

def update_flood_rainfall_slider(unit):
    if unit == "mm":
        return gr.update(minimum=0, maximum=500, value=25, label="Rainfall (mm)")
    elif unit == "in":
        return gr.update(minimum=0, maximum=20, value=1, label="Rainfall (in)")

def update_rain_slider(unit):
    if unit == "in/hr":
        return gr.update(minimum=0, maximum=10, value=0.5, label="Rainfall Intensity (in/hr)")
    elif unit == "mm/hr":
        return gr.update(minimum=0, maximum=150, value=12, label="Rainfall Intensity (mm/hr)")