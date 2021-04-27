import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa


class CustomConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True):
        # save arguments for Conv2D layer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn

    def __call__(self, x, filters, name=None):
        CustomConv2D.CALL_COUNT += 1
        index = CustomConv2D.CALL_COUNT
        name = f"conv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name
        print("conv name", name)

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding,
                          use_bias=not self.bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                          bias_initializer=tf.constant_initializer(0.), name=name,
                          )(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)

        if self.bn:
            x = layers.BatchNormalization()(x)
        return x


def slice_features(feature, output_composition):
    index = 0
    slices = dict()
    for name, channel in output_composition:
        slices[name] = feature[..., index:index+channel]
        index += channel
    return slices
