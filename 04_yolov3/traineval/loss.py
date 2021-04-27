import tensorflow as tf


class LossBase:
    def __init__(self):
        pass

    def __call__(self, feat, pred):
        dummy_large = tf.reduce_mean(tf.square(pred["feature_l"]))
        dummy_medium = tf.reduce_mean(tf.square(pred["feature_m"]))
        dummy_small = tf.reduce_mean(tf.square(pred["feature_s"]))
        dummy_total = dummy_large + dummy_medium + dummy_small
        return dummy_total, {"dummy_large": dummy_large, "dummy_medium": dummy_medium, "dummy_small": dummy_small}




