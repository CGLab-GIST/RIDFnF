import tensorflow as tf

def L2Loss(y_true, y_pred):
    diff_squared = tf.math.squared_difference(y_true, y_pred)
    diff_squared_mean = tf.reduce_mean(diff_squared, axis=-1)
    return diff_squared_mean


