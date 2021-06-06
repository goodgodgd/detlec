import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import utils.util_function as uf
import numpy as np

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


class NonMaximumSuppression:
    def __init__(self, max_out=cfg.NMS.MAX_OUT,
                 iou_thresh=cfg.NMS.IOU_THRESH,
                 score_thresh=cfg.NMS.SCORE_THRESH,
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

    @tf.function
    def __call__(self, pred, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'bbox': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'bbox': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            scales = [key for key in pred if "feature_" in key]
            slice_keys = list(pred[scales[0]].keys())    # ['bbox', 'object', 'category']
            merged_pred = {}
            # merge pred features over scales
            for key in slice_keys:
                # list of (batch, HWA in scale, dim)
                scaled_preds = [pred[scale_name][key] for scale_name in scales]
                scaled_preds = tf.concat(scaled_preds, axis=1)      # (batch, N, dim)
                merged_pred[key] = scaled_preds
            pred = merged_pred

        boxes = uf.convert_box_format_yxhw_to_tlbr(pred["bbox"])    # (batch, N, 4)
        categories = tf.argmax(pred["category"], axis=-1)           # (batch, N)
        best_probs = tf.reduce_max(pred["category"], axis=-1)       # (batch, N)
        objectness = tf.squeeze(pred["object"])                     # (batch, N)
        scores = objectness * best_probs                            # (batch, N)
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
                    max_output_size=self.max_out,
                    iou_threshold=self.iou_thresh,
                    score_threshold=self.score_thresh,
                )
                # zero padding that works in tf.function
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out - numsel), dtype=tf.int32) * -1
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


def test_nms():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N = 4, 1000
    pred = {"feature_l": {"bbox": tf.random.uniform((B, N, 4), dtype=tf.float32),
                          "object": tf.random.uniform((B, N, 1), dtype=tf.float32),
                          "category": tf.random.uniform((B, N, 4), dtype=tf.float32),
                          },
            "feature_m": {"bbox": tf.random.uniform((B, N, 4), dtype=tf.float32),
                          "object": tf.random.uniform((B, N, 1), dtype=tf.float32),
                          "category": tf.random.uniform((B, N, 4), dtype=tf.float32),
                          },
            "feature_s": {"bbox": tf.random.uniform((B, N, 4), dtype=tf.float32),
                          "object": tf.random.uniform((B, N, 1), dtype=tf.float32),
                          "category": tf.random.uniform((B, N, 4), dtype=tf.float32),
                          }
            }
    result = NonMaximumSuppression(max_out=18)(pred, False)
    print(result.shape)
    print(result[0].numpy())


if __name__ == "__main__":
    test_nms()

