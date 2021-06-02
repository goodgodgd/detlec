import tensorflow as tf
import utils.util_function as uf


def count_true_positives(grtr, pred, num_ctgr, iou_thresh=0.5, per_class=False, verbose=False):
    """
    :param grtr: GT object information (batch, N, 5) / 5: yxhw, category index
    :param pred: nms result (batch, M, 8) / 8: (yxhw, category index, objectness, ctgr prob, score)
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :return:
    """
    iou = uf.compute_iou_general(grtr, pred)    # (batch, N, M)
    best_iou = tf.reduce_max(iou, axis=-1)      # (batch, N)
    best_idx = tf.argmax(iou, axis=-1)          # (batch, N)
    pred_aligned = tf.gather(pred, best_idx, batch_dims=1)              # (batch, N, 8)
    valid_iou = tf.cast(best_iou > iou_thresh, dtype=tf.float32)        # (batch, N)
    valid_grtr = tf.cast(grtr[..., 0] > 0, dtype=tf.float32)            # (batch, N) y > 0
    valid_pred = tf.cast(pred[..., -1] > 0, dtype=tf.float32)           # (batch, M) score > 0
    grtr_ctgr = tf.cast(grtr[..., 4], dtype=tf.int32)                   # (batch, N)
    pred_ctgr = tf.cast(pred[..., 4], dtype=tf.int32)                   # (batch, M)
    pred_ctgr_aligned = tf.cast(pred_aligned[..., 4], dtype=tf.int32)           # (batch, N)
    ctgr_match = tf.cast(grtr_ctgr == pred_ctgr_aligned, dtype=tf.float32)      # (batch, N)

    grtr_onehot = tf.one_hot(grtr_ctgr, depth=num_ctgr) * valid_grtr[..., tf.newaxis]       # (batch, N, K)
    pred_onehot = tf.one_hot(pred_ctgr, depth=num_ctgr) * valid_pred[..., tf.newaxis]       # (batch, M, K)
    match_onehot = grtr_onehot * ctgr_match[..., tf.newaxis] * valid_iou[..., tf.newaxis]   # (batch, N, K)

    if verbose:
        print("ctgr_match", ctgr_match)
        print("valid_iou", valid_iou)
        print("best_idx", best_idx)
        print("best_iou", best_iou)
        print("grtr_onehot", grtr_onehot[0])
        print("match_onehot", match_onehot[0])
        print("valid_pred", valid_pred[0])

    if per_class:
        grtr_count = tf.reduce_sum(grtr_onehot, axis=1)
        pred_count = tf.reduce_sum(pred_onehot, axis=1)
        trpo_count = tf.reduce_sum(match_onehot, axis=1)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}
    else:
        grtr_count = tf.reduce_sum(valid_grtr)
        pred_count = tf.reduce_sum(valid_pred)
        trpo_count = tf.reduce_sum(match_onehot)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


import numpy as np


def test_count_true_positives():
    print("===== start count_true_positives")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    B, N, K = 4, 10, 8
    count = 0

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
        N1 = 20
        grtr_fn_boxes = np.tile(np.random.uniform(0, 1, (1, N1, 10)) * 0.95 + 0.05, (B, 1, 1))
        # different boxes are supposed to have low iou
        grtr_fn_boxes[..., 0] = np.linspace(0, 0.3 * (N1 - 1), N1).reshape((1, N1)) + 5.1
        grtr_fn_boxes[..., 4] = np.random.randint(0, K, N1).reshape((1, N1))    # set category indices
        pred_fp_boxes = grtr_fn_boxes.copy()
        pred_fp_boxes[:, :5, :2] += 2           # zero iou
        pred_fp_boxes[:, 5:10, 4] = (pred_fp_boxes[:, 5:10, 4] + 1) % K         # different category
        pred_fp_boxes[:, 10:15, :] = 0          # zero pred box
        grtr_fn_boxes[:, 15:20, :] = 0          # zero gt box

        # grtr_boxes, pred_boxes: N similar boxes, N1 different boxes
        grtr_boxes = tf.concat([grtr_tp_boxes, grtr_fn_boxes[..., :5]], axis=1)
        pred_boxes = tf.concat([pred_tp_boxes, pred_fp_boxes[..., :8]], axis=1)

        result = count_true_positives(grtr_boxes, pred_boxes, K)
        result = {key: val.numpy() for key, val in result.items()}
        # true positive is supposed to be 40
        print("result:", i, result)
        if result["trpo"] == B * N:
            count += 1
        else:
            print("grtr_boxes", grtr_boxes[0])
            print("pred_boxes", pred_boxes[0])
            count_true_positives(grtr_boxes, pred_boxes, K, verbose=True)

    print("count:", count)
    assert count == 100
    print("!!! pass test_count_true_positives")


if __name__ == "__main__":
    test_count_true_positives()

