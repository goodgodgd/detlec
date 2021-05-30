import tensorflow as tf
import utils.util_function as uf


def count_true_positives(grtr, pred, iou_thresh=0.5, per_class=False):
    """
    :param grtr: GT object information (batch, N, 5) / 5: bbox, category index
    :param pred: nms result (batch, M, 8) / 8: (bbox, category index, objectness, ctgr prob, score)
    :return:
    """
    iou = uf.compute_iou_general(grtr, pred)    # (batch, N, M)
    best_iou = tf.reduce_max(iou, axis=-1)      # (batch, N)
    best_idx = tf.argmax(iou, axis=-1)          # (batch, N)
    pred_aligned = tf.gather(pred, best_idx, batch_dims=1)              # (batch, N, 8)
    valid_iou = tf.cast(best_iou > iou_thresh, dtype=tf.float32)        # (batch, N)
    valid_grtr = tf.cast(grtr[..., 0] > 0, dtype=tf.float32)            # (batch, N) y > 0
    valid_pred = tf.cast(pred_aligned[..., -1] > 0, dtype=tf.float32)   # (batch, N) score > 0
    grtr_ctgr = tf.cast(grtr[..., 4], dtype=tf.int32)                   # (batch, N)
    pred_ctgr = tf.cast(pred_aligned[..., 4], dtype=tf.int32)           # (batch, N)

    grtr_onehot = tf.one_hot(grtr_ctgr) * valid_grtr            # (batch, N, K)
    pred_onehot = tf.one_hot(pred_ctgr) * valid_pred            # (batch, N, K)
    ctgr_match = tf.cast(grtr_onehot == pred_onehot, dtype=tf.float32) * valid_iou[..., tf.newaxis]     # (batch, N, K)

    grtr_ctgr_count = tf.reduce_sum(grtr_onehot, axis=[0, 1])   # (K)
    pred_ctgr_count = tf.reduce_sum(pred_onehot, axis=[0, 1])   # (K)
    trpo_ctgr_count = tf.reduce_sum(ctgr_match, axis=[0, 1])    # (K)

    if per_class:
        return {"trpo": trpo_ctgr_count, "grtr": grtr_ctgr_count, "pred": pred_ctgr_count}
    else:
        grtr_count = tf.reduce_sum(grtr_ctgr_count)    # ()
        pred_count = tf.reduce_sum(pred_ctgr_count)    # ()
        trpo_count = tf.reduce_sum(trpo_ctgr_count)    # ()
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def test_count_tps():
    pass


