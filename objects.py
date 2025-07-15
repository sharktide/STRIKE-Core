from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, models, backend as K
import tensorflow as tf

####################################################################################################
# FireNet
####################################################################################################

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

    temp_boost = tf.sigmoid((temp - 305.0) * 1.2)
    humidity_boost = tf.sigmoid((20.0 - humidity) * 0.5)
    wind_boost = tf.sigmoid((wind - 15.0) * 0.8)
    veg_boost = tf.sigmoid((veg - 70.0) * 0.5)

    combined = temp_boost * humidity_boost * wind_boost * veg_boost
    boost = 1.0 + 0.3 * combined
    return boost[:, None]

@register_keras_serializable()
def fire_suppression_mask(inputs):
    temp = inputs[:, 0]
    humidity = inputs[:, 1]
    wind = inputs[:, 2]

    temp_flag = tf.sigmoid((temp - 293.0) * 1.2)
    humid_flag = tf.sigmoid((humidity - 50.0) * 0.4)
    wind_flag = 1 - tf.sigmoid((wind - 5.0) * 0.8)

    suppression = temp_flag * humid_flag * wind_flag
    penalty = 1.0 - 0.3 * suppression
    return penalty[:, None]


####################################################################################################
# FloodNets
####################################################################################################

@register_keras_serializable()
def rainfall_proximity_penalty(inputs):
    rainfall = inputs[:, 0]
    distance = inputs[:, 4]
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
    flatness = tf.sigmoid((elevation - 9.0) * 0.6)
    dryness = tf.sigmoid((20.0 - rainfall) * 0.2)
    suppression = flatness * dryness
    return 1.0 - 0.3 * suppression[:, None]

@register_keras_serializable()
def surface_runoff_amplifier(inputs):
    rain = inputs[:, 0]
    impervious = inputs[:, 1]
    rain_boost = tf.sigmoid((rain - 60) * 0.06)
    impervious_boost = tf.sigmoid((impervious - 0.6) * 10)
    return (1.0 + 0.3 * rain_boost * impervious_boost)[:, None]

@register_keras_serializable()
def drainage_penalty(inputs):
    dd = inputs[:, 2]
    return (1.0 - 0.4 * tf.sigmoid((dd - 3.5) * 2))[:, None]

@register_keras_serializable()
def convergence_suppressor(inputs):
    ci = inputs[:, 4]
    return (1.0 + 0.3 * tf.sigmoid((ci - 0.5) * 8))[:, None]

####################################################################################################
# Activation for all TrustNets
####################################################################################################

@register_keras_serializable()
def firetrust_activation(x):
    return 0.5 + tf.sigmoid(x)

@register_keras_serializable()
def floodtrust_activation(x):
    return 0.5 + tf.sigmoid(x)

####################################################################################################
# Clip Modulation
####################################################################################################

def clip_modulation(x):
    return tf.clip_by_value(x, 0.7, 1.3)