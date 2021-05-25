import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import utils.util_function as uf


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

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding,
                          use_bias=not self.bn,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                          bias_initializer=tf.constant_initializer(0.), name=name,
                          )(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)

        if self.bn:
            x = layers.BatchNormalization()(x)
        return x


def non_maximum_suppression(pred):
    """
    :param pred: merged predictions, (batch, N, 5+K), N: sum of HWAs over scales(l,m,s)
    :return:
    """
    pred = uf.slice_features(pred)
    boxes = uf.convert_box_format_yxhw_to_tlbr(pred["bbox"])   # (batch, N, 4)
    categories = tf.argmax(pred["category"], axis=-1)          # (batch, N)
    batch, numbox, numctgr = pred["category"].shape

    batch_result = [[] for i in range(batch)]
    for ctgr_idx in range(numctgr):
        ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)   # (batch, N)
        ctgr_boxes = ctgr_mask[..., tf.newaxis] * boxes                 # (batch, N, 4)
        ctgr_scores = pred["object"] * pred["category"][..., ctgr_idx] * ctgr_mask   # (batch, N)
        for frame_idx in range(batch):
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes=ctgr_boxes,
                scores=ctgr_scores,
                max_output_size=50,
                iou_threshold=0.5,
                score_threshold=0.5,
                soft_nms_sigma=0.5
            )
            # gather: box, scores, object, ctgr_prob
            output = 0
            batch_result[frame_idx].append(output)

    batch_result = [tf.concat(ctgr_result, axis=-1) for ctgr_result in batch_result]
    return batch_result


def test_nms():
    pass

