import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import utils.util_function as uf
import numpy as np


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


@tf.function
def non_maximum_suppression_batch(pred, max_output=50):
    """
    :param pred: merged predictions, dict of (batch, N, dim), N: sum of HWAs over scales(l,m,s)
    :return:
    """
    pred = uf.slice_features(pred)
    boxes = uf.convert_box_format_yxhw_to_tlbr(pred["bbox"])    # (batch, N, 4)
    categories = tf.argmax(pred["category"], axis=-1)           # (batch, N)
    best_probs = tf.reduce_max(pred["category"], axis=-1)       # (batch, N)
    objectness = tf.squeeze(pred["object"])                     # (batch, N)
    scores = objectness * best_probs            # (batch, N)
    batch, numbox, numctgr = pred["category"].shape

    batch_indices = [[] for i in range(batch)]
    for ctgr_idx in range(numctgr):
        ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)   # (batch, N)
        ctgr_boxes = boxes * ctgr_mask[..., tf.newaxis]                 # (batch, N, 4)
        ctgr_scores = scores * ctgr_mask                                # (batch, N)
        for frame_idx in range(batch):
            selected_indices = tf.image.non_max_suppression(
                boxes=ctgr_boxes[frame_idx],
                scores=ctgr_scores[frame_idx],
                max_output_size=max_output,
                iou_threshold=0.3,
                score_threshold=0.8,
            )
            # zero padding that works in tf.function
            numsel = tf.shape(selected_indices)[0]
            zero = tf.ones((max_output - numsel), dtype=tf.int32) * -1
            selected_indices = tf.concat([selected_indices, zero], axis=0)
            batch_indices[frame_idx].append(selected_indices)

    # make batch_indices, valid_mask as fixed shape tensor
    batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
    batch_indices = tf.stack(batch_indices, axis=0)             # (batch, K*max_output)
    valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
    batch_indices = tf.maximum(batch_indices, 0)

    # list of (batch, N) -> (batch, N, 4)
    categories = tf.cast(categories, dtype=tf.float32)
    result = tf.stack([categories, objectness, best_probs, scores], axis=-1)
    result = tf.concat([boxes, result], axis=-1)                # (batch, N, 8)
    result = tf.gather(result, batch_indices, batch_dims=1)     # (batch, K*max_output, 8)
    result = result * valid_mask[..., tf.newaxis]               # (batch, K*max_output, 8)
    return result


@tf.function
def non_maximum_suppression_compare(pred, max_output=50):
    """
    :param pred: merged predictions, dict of (batch, N, dim), N: sum of HWAs over scales(l,m,s)
    :return:
    """
    pred = uf.slice_features(pred)
    boxes = uf.convert_box_format_yxhw_to_tlbr(pred["bbox"])    # (batch, N, 4)
    categories = tf.argmax(pred["category"], axis=-1)           # (batch, N)
    best_probs = tf.reduce_max(pred["category"], axis=-1)       # (batch, N)
    objectness = tf.squeeze(pred["object"])                     # (batch, N)
    scores = objectness * best_probs            # (batch, N)
    batch, numbox, numctgr = pred["category"].shape

    ctgr_fl = tf.cast(categories, dtype=tf.float32)
    output = tf.stack([ctgr_fl, objectness, best_probs, scores], axis=-1)
    output = tf.concat([boxes, output], axis=-1)                # (batch, N, 8)

    batch_result = [[] for i in range(batch)]
    for ctgr_idx in range(numctgr):
        ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)   # (batch, N)
        ctgr_boxes = boxes * ctgr_mask[..., tf.newaxis]                 # (batch, N, 4)
        ctgr_scores = scores * ctgr_mask                                # (batch, N)
        for frame_idx in range(batch):
            selected_indices = tf.image.non_max_suppression(
                boxes=ctgr_boxes[frame_idx],
                scores=ctgr_scores[frame_idx],
                max_output_size=max_output,
                iou_threshold=0.3,
                score_threshold=0.8,
            )
            unit_output = tf.gather(output[frame_idx], selected_indices)    # (M_fc, 8)
            batch_result[frame_idx].append(unit_output)

    # list of (M_f, 8)
    batch_result = [tf.concat(ctgr_result, axis=0) for ctgr_result in batch_result]
    return batch_result


def test_nms():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N = 4, 1000
    pred = tf.random.uniform((B, N, 9), dtype=tf.float32)
    result = non_maximum_suppression_compare(pred, 18)
    print(result[0].shape)
    print(result[0].numpy())


def test_compare_nms():
    """
    compare fixed shape nms vs variable shape nms
    [non_maximum_suppression_batch] mean time: 0.01789      (fixed shape)
    [non_maximum_suppression_compare] mean time: 0.01803    (variable shape)
    """
    from timeit import default_timer as timer
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N = 16, 40000

    # warm-up
    for i in range(10):
        pred = tf.random.uniform((B, N, 9), dtype=tf.float32)
        result = non_maximum_suppression_batch(pred, 50)

    start = timer()
    for i in range(100):
        pred = tf.random.uniform((B, N, 9), dtype=tf.float32)
        result = non_maximum_suppression_batch(pred, 50)
    print(f"[non_maximum_suppression_batch] mean time: {(timer() - start) / 100.:.5f}")

    # warm-up
    for i in range(10):
        pred = tf.random.uniform((B, N, 9), dtype=tf.float32)
        result = non_maximum_suppression_compare(pred, 50)

    start = timer()
    for i in range(100):
        pred = tf.random.uniform((B, N, 9), dtype=tf.float32)
        result = non_maximum_suppression_compare(pred, 50)
    print(f"[non_maximum_suppression_compare] mean time: {(timer() - start) / 100.:.5f}")


if __name__ == "__main__":
    # test_nms()
    test_compare_nms()

