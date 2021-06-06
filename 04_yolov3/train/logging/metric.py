import tensorflow as tf
import utils.util_function as uf


def count_true_positives(grtr, pred, num_ctgr, iou_thresh=0.5, per_class=False):
    """
    :param grtr: slices of GT object info {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :return:
    """
    splits = split_true_false(grtr, pred, iou_thresh)

    valid_grtr = tf.cast(grtr["yxhw"][..., 0] > 0, dtype=tf.float32)        # (batch, N) y > 0
    valid_pred = tf.cast(pred["score"][..., 0] > 0, dtype=tf.float32)       # (batch, M) score > 0
    valid_trpo = tf.cast(splits["grtr_tp"]["yxhw"][..., 0] > 0, dtype=tf.float32)   # (batch, M) y > 0

    if per_class:
        grtr_count = count_per_class(grtr, valid_grtr, num_ctgr)
        pred_count = count_per_class(pred, valid_pred, num_ctgr)
        trpo_count = count_per_class(splits["grtr_tp"], valid_trpo, num_ctgr)
        return {"trpo": trpo_count.numpy(), "grtr": grtr_count.numpy(), "pred": pred_count.numpy()}
    else:
        grtr_count = tf.reduce_sum(valid_grtr)
        pred_count = tf.reduce_sum(valid_pred)
        trpo_count = tf.reduce_sum(valid_trpo)
        return {"trpo": trpo_count.numpy(), "grtr": grtr_count.numpy(), "pred": pred_count.numpy()}


def split_true_false(grtr, pred, iou_thresh=0.5):
    """
    :param grtr: slices of GT object info {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :return:
    """
    grtr_ctgr = tf.cast(grtr["category"][..., 0], dtype=tf.int32)           # (batch, N)
    iou = uf.compute_iou_general(grtr["yxhw"], pred["yxhw"])                # (batch, N, M)
    best_iou = tf.reduce_max(iou, axis=-1)                                  # (batch, N)
    iou_match = tf.cast(best_iou > iou_thresh, dtype=tf.float32)            # (batch, N)
    best_idx = tf.cast(tf.argmax(iou, axis=-1), dtype=tf.int32)             # (batch, N)
    pred_ctgr_aligned = tf.gather(pred["category"][..., 0], best_idx, batch_dims=1)     # (batch, N, 8)
    pred_ctgr_aligned = tf.cast(pred_ctgr_aligned, dtype=tf.int32)          # (batch, N)
    ctgr_match = tf.cast(grtr_ctgr == pred_ctgr_aligned, dtype=tf.float32)  # (batch, N)
    grtr_tp_mask = (iou_match * ctgr_match)[..., tf.newaxis]                # (batch, N, 1)
    grtr_fn_mask = tf.convert_to_tensor(1, dtype=tf.float32) - grtr_tp_mask # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}

    B, M, _ = pred["category"].shape
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    best_idx_onehot = tf.one_hot(best_idx, depth=M) * grtr_tp_mask          # (batch, N, M)
    pred_tp_mask = tf.reduce_max(best_idx_onehot, axis=1)[..., tf.newaxis]  # (batch, M, 1)
    pred_fp_mask = tf.convert_to_tensor(1, dtype=tf.float32) - pred_tp_mask  # (batch, N, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}
    return {"grtr_tp": grtr_tp, "grtr_fn": grtr_fn, "pred_tp": pred_tp, "pred_fp": pred_fp}


def count_per_class(boxes, mask, num_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = tf.cast(boxes["category"][..., 0], dtype=tf.int32)  # (batch, N')
    boxes_onehot = tf.one_hot(boxes_ctgr, depth=num_ctgr) * mask[..., tf.newaxis]  # (batch, N', K)
    boxes_count = tf.reduce_sum(boxes_onehot, axis=1)
    return boxes_count


import numpy as np


def test_count_true_positives():
    print("===== start count_true_positives")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N, K = 4, 10, 8  # N: num of TP
    NF = 20             # NF: num of false data

    for i in range(100):
        # create similar grtr and pred boxes that have iou > 0.5
        # preventing zero height or width
        grtr_tp_boxes = np.tile(np.random.uniform(0, 1, (1, N, 10)) * 0.95 + 0.05, (B, 1, 1))
        # different boxes are supposed to have low iou
        grtr_tp_boxes[..., 0] = np.linspace(0, 0.3*(N-1), N).reshape((1, N)) + 0.1
        grtr_tp_boxes[..., 4] = np.random.randint(0, K, N).reshape((1, N))    # set category indices
        pred_tp_boxes = grtr_tp_boxes.copy()
        # add small noise to boxes
        pred_tp_boxes[..., :4] += np.tile(grtr_tp_boxes[..., 2:4], 2) * (np.random.uniform(0, 1, (B, N, 4)) - 0.5) * 0.1
        grtr_tp_boxes = tf.convert_to_tensor(grtr_tp_boxes[..., :5])
        pred_tp_boxes = tf.convert_to_tensor(pred_tp_boxes[..., :8])
        iou = uf.compute_iou_general(grtr_tp_boxes, pred_tp_boxes)
        if (tf.linalg.diag_part(iou).numpy() <= 0.5).any():
            print("grtr vs pred (aligned) iou:", tf.linalg.diag_part(iou).numpy())
            print("grtr_boxes", grtr_tp_boxes[0])
            print("pred_boxes", pred_tp_boxes[0])
        assert (tf.linalg.diag_part(iou).numpy() > 0.5).all()

        # create different grtr and pred boxes
        grtr_fn_boxes = np.tile(np.random.uniform(0, 1, (1, NF, 10)) * 0.95 + 0.05, (B, 1, 1))
        # different boxes are supposed to have low iou
        grtr_fn_boxes[..., 0] = np.linspace(0, 0.3 * (NF - 1), NF).reshape((1, NF)) + 5.1
        grtr_fn_boxes[..., 4] = np.random.randint(0, K, NF).reshape((1, NF))    # set category indices
        pred_fp_boxes = grtr_fn_boxes.copy()
        pred_fp_boxes[:, :5, :2] += 2           # zero iou
        pred_fp_boxes[:, 5:10, 4] = (pred_fp_boxes[:, 5:10, 4] + 1) % K         # different category
        pred_fp_boxes[:, 10:15, :] = 0          # zero pred box
        grtr_fn_boxes[:, 15:20, :] = 0          # zero gt box

        # grtr_boxes, pred_boxes: N similar boxes, NF different boxes
        grtr_boxes = tf.cast(tf.concat([grtr_tp_boxes, grtr_fn_boxes[..., :5]], axis=1), dtype=tf.float32)
        pred_boxes = tf.cast(tf.concat([pred_tp_boxes, pred_fp_boxes[..., :8]], axis=1), dtype=tf.float32)
        grtr_boxes = uf.slice_bbox(grtr_boxes, True)
        pred_boxes = uf.slice_bbox(pred_boxes, False)

        # EXECUTE
        result = count_true_positives(grtr_boxes, pred_boxes, K)
        # true positive is supposed to be 40
        print("result", result)
        assert result["trpo"] == B * N
        assert result["grtr"] == B * (N + NF - 5)
        assert result["pred"] == B * (N + NF - 5)

    print("!!! pass test_count_true_positives")


if __name__ == "__main__":
    test_count_true_positives()

