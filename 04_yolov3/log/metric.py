import numpy as np
import utils.util_function as uf
import config as cfg


def count_true_positives(grtr, pred, num_ctgr, iou_thresh=cfg.ModelOutput.TP_IOU_THRESH, per_class=False):
    """
    :param grtr: GT instances   {'yxhw': (batch, N, 4), 'object': (batch, N, 1), 'category': (batch, N, 1)}
    :param pred: pred instances {'yxhw': (batch, M, 4), 'object': (batch, M, 1), 'category': (batch, M, 1)}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :return:
    """
    splits = split_tp_fp_fn(grtr, pred, iou_thresh)
    split_count = {}
    for split_key in splits:    # for ["grtr_tp", "grtr_fn", "pred_tp", "pred_fp"]
        split_mask = splits[split_key]["yxhw"][..., 2:3] > 0
        if per_class:
            split_count[split_key] = count_per_class(splits[split_key], split_mask, num_ctgr)
        else:
            split_count[split_key] = np.sum(split_mask)

    grtr_cnt = np.sum(grtr["yxhw"][..., 2:3] > 0)
    pred_cnt = np.sum(pred["yxhw"][..., 2:3] > 0)
    assert grtr_cnt == (split_count["grtr_tp"] + split_count["grtr_fn"])
    assert pred_cnt == (split_count["pred_tp"] + split_count["pred_fp"])
    if split_count["grtr_tp"] != split_count["pred_tp"]:
        print("!!ERR!!", grtr_cnt, pred_cnt, split_count)
    # assert split_count["grtr_tp"] == split_count["pred_tp"]

    return {"trpo": split_count["grtr_tp"],
            "grtr": (split_count["grtr_tp"] + split_count["grtr_fn"]),
            "pred": (split_count["pred_tp"] + split_count["pred_fp"])}


def split_tp_fp_fn(grtr, pred, iou_thresh):
    batch, M, _ = pred["category"].shape
    valid_mask = grtr["object"]
    iou = uf.compute_iou_general(grtr["yxhw"], pred["yxhw"])  # (batch, N, M)
    best_iou = np.max(iou, axis=-1)  # (batch, N)
    best_idx = np.argmax(iou, axis=-1)  # (batch, N)

    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)
    # category (batch, M, 1), best idx (batch, N) -> (batch, N)
    pred_ctgr_aligned = np.take_along_axis(pred["category"][..., 0], best_idx, 1)
    ctgr_match = grtr["category"][..., 0] == pred_ctgr_aligned      # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)
    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)

    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}

    # _best_iou__ = best_iou[0]
    # _best_idx__ = best_idx[0]
    # _grtr_ctgr__ = grtr["category"][0, :, 0]
    # _pred_ctgr__ = pred["category"][0, :, 0]
    # _iou_match__ = iou_match[0]
    # _ctgr_match__ = ctgr_match[0]
    # _grtr_tp_mask__ = grtr_tp_mask[0, :, 0]
    # _grtr_fn_mask__ = grtr_fn_mask[0, :, 0]
    # _pred_tp_mask__ = pred_tp_mask[0, :, 0]
    # _grtr_height__ = grtr["yxhw"][0, :, 2]
    # _pred_height__ = pred["yxhw"][0, :, 2]
    # __sum_result = {"grtr_sum": np.sum(_grtr_height__ > 0),
    #                 "pred_sum": np.sum(_pred_height__ > 0),
    #                 "grtr_tp_sum": np.sum(grtr_tp["yxhw"][0,:,0] > 0),
    #                 "grtr_fn_sum": np.sum(grtr_fn["yxhw"][0,:,0] > 0),
    #                 "pred_tp_sum": np.sum(pred_tp["yxhw"][0, :, 0] > 0),
    #                 "pred_fp_sum": np.sum(pred_fp["yxhw"][0, :, 0] > 0),
    #                 }

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def indices_to_binary_mask(best_idx, valid_mask, depth):
    best_idx_onehot = one_hot(best_idx, depth) * valid_mask  # (batch, M, N) * (batch, M, 1)
    binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1)  # (batch, M, 1)
    sum_mask = np.expand_dims(np.sum(best_idx_onehot, axis=1), axis=-1)  # (batch, M, 1)
    bin_sum = np.sum(binary_mask)
    sum_sum = np.sum(sum_mask)
    if bin_sum != sum_sum:
        print("!!ERR trpo overlap", bin_sum, sum_sum)

    return binary_mask.astype(np.float32)


def get_iou_thresh_per_class(grtr_ctgr, iou_thresh):
    """
    :param grtr_ctgr: (batch, N) 
    :param iou_thresh: (num_ctgr)
    :return: 
    """
    ctgr_idx = grtr_ctgr.astype(np.int32)[..., 0]                       # (batch, N)
    iou_thresh = np.asarray(iou_thresh, np.float32)[np.newaxis, ...]    # (1, num_ctgr)
    iou_thresh = np.take_along_axis(iou_thresh, ctgr_idx, axis=1)       # (batch, N)
    return iou_thresh


def count_per_class(boxes, mask, num_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = boxes["category"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count


def one_hot(grtr_category, category_shape):
    one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
    return one_hot_data
