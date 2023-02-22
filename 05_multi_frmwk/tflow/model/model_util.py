import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

import tflow.utils.util_function as tuf
import config as cfg


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


class NonMaximumSuppressionBox:
    def __init__(self, max_out=cfg.NmsInfer.MAX_OUT,
                 iou_thresh=cfg.NmsInfer.IOU_THRESH,
                 score_thresh=cfg.NmsInfer.SCORE_THRESH,
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

    def __call__(self, pred):
        """
        :param pred: {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
        :return: (batch, max_out, 6), 6: bbox, score, category
        """
        pred = {key: tf.concat(data, axis=1) for key, data in pred.items() if isinstance(data, list) and not key.endswith("_logit")}
        boxes = tuf.convert_box_format_yxhw_to_tlbr(pred["yxhw"])  # (batch, N, 4)
        categories = tf.argmax(pred["category"], axis=-1)  # (batch, N)
        best_probs = tf.reduce_max(pred["category"], axis=-1)  # (batch, N)
        objectness = pred["object"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = pred["category"].shape

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(numctgr):
            ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)  # (batch, N)
            ctgr_boxes = boxes * ctgr_mask[..., tf.newaxis]  # (batch, N, 4)
            ctgr_scores = scores * ctgr_mask  # (batch, N)

            # TODO: use non_max_suppression_padded(), it processes a batch at ONCE
            for frame_idx in range(batch):
                selected_indices = tf.image.non_max_suppression(
                    boxes=ctgr_boxes[frame_idx],
                    scores=ctgr_scores[frame_idx],
                    max_output_size=self.max_out[ctgr_idx],
                    iou_threshold=self.iou_thresh[ctgr_idx],
                    score_threshold=self.score_thresh[ctgr_idx],
                )
                # zero padding that works in tf.function
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, zero], axis=0)
                batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)  # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)

        # list of (batch, N) -> (batch, N, 4)
        categories = tf.cast(categories, dtype=tf.float32)
        # "bbox": 4, "object": 1, "category": 1
        result = tf.stack([scores, categories], axis=-1)
        result = tf.concat([pred["yxhw"], result], axis=-1)  # (batch, N, 6)
        result = tf.gather(result, batch_indices, batch_dims=1)  # (batch, K*max_output, 6)
        result = result * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 6)
        return result
