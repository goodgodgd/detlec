import sys
import numpy as np
import tensorflow as tf

import config as cfg


def set_gpu_configs():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32)


def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, 0, 1)
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.uint8)


def convert_box_format_tlbr_to_yxhw(boxes_tlbr):
    """
    :param boxes_tlbr: type=tf.Tensor or np.array, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_yx = (boxes_tlbr[..., 0:2] + boxes_tlbr[..., 2:4]) / 2    # center y,x
    boxes_hw = boxes_tlbr[..., 2:4] - boxes_tlbr[..., 0:2]          # y2,x2 = y1,x1 + h,w
    output = [boxes_yx, boxes_hw]
    output = concat_box_output(output, boxes_tlbr)
    return output


def convert_box_format_yxhw_to_tlbr(boxes_yxhw):
    """
    :param boxes_yxhw: type=tf.Tensor or np.array, shape=(numbox, dim) or (batch, numbox, dim)
    :return:
    """
    boxes_tl = boxes_yxhw[..., 0:2] - (boxes_yxhw[..., 2:4] / 2)    # y1,x1 = cy,cx + h/2,w/2
    boxes_br = boxes_tl + boxes_yxhw[..., 2:4]                      # y2,x2 = y1,x1 + h,w
    output = [boxes_tl, boxes_br]
    output = concat_box_output(output, boxes_yxhw)
    return output


def concat_box_output(output, boxes):
    num, dim = boxes.shape[-2:]
    # if there is more than bounding box, append it  e.g. category, distance
    if dim > 4:
        auxi_data = boxes[..., 4:]
        output.append(auxi_data)

    if tf.is_tensor(boxes):
        output = tf.concat(output, axis=-1)
        output = tf.cast(output, boxes.dtype)
    else:
        output = np.concatenate(output, axis=-1)
        output = output.astype(boxes.dtype)
    return output


def merge_and_slice_features(features, is_gt):
    """
    :param features: this dict has keys feature_l,m,s and corresponding tensors are in (batch, grid_h, grid_w, anchors, dims)
    :param is_gt: is ground truth feature map?
    :return: sliced feature maps in each scale
    """
    scales = [key for key in features if "feature" in key]  # ['feature_l', 'feature_m', 'feature_s']
    sliced_features = {}
    for key in scales:
        raw_feat = features[key]
        merged_feat = merge_dim_hwa(raw_feat)
        channel_compos = cfg.Model.Output.get_channel_composition(is_gt)
        slices = slice_feature(merged_feat, channel_compos)
        sliced_features[key] = slices

    if "bboxes" in features:
        bbox_compos = cfg.Model.Output.get_bbox_composition(is_gt)
        sliced_features["bboxes"] = slice_feature(features["bboxes"], bbox_compos)

    other_features = {key: val for key, val in features.items() if key not in sliced_features}
    sliced_features.update(other_features)
    return sliced_features


def slice_feature(feature, channel_composition):
    """
    :param feature: (batch, grid_h, grid_w, anchors, dims)
    :param channel_composition:
    :return: sliced feature maps
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    slices = tf.split(feature, channels, axis=-1)
    slices = dict(zip(names, slices))           # slices = {'bbox': (B,H,W,A,4), 'object': (B,H,W,A,1), ...}
    return slices


def merge_dim_hwa(feature_map):
    """
    :param feature_map: (batch, grid_h, grid_w, anchor, 5+K)
    :return: (batch, grid_h * grid_w * anchor, 5+K)
    """
    batch, grid_h, grid_w, anchor, featdim = feature_map.shape
    merged_feat = tf.reshape(feature_map, (batch, grid_h * grid_w * anchor, featdim))
    return merged_feat


def compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: GT bounding boxes in yxhw format (batch, HWA, D(>4))
    :param pred_yxhw: predicted bounding boxes aligned with GT in yxhw format (batch, HWA, D(>4))
    :return: iou (batch, HWA)
    """
    if grtr_tlbr is None:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxhw)
    if pred_tlbr is None:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxhw)
    inter_tl = tf.maximum(grtr_tlbr[..., :2], pred_tlbr[..., :2])
    inter_br = tf.minimum(grtr_tlbr[..., 2:4], pred_tlbr[..., 2:4])
    inter_hw = inter_br - inter_tl
    positive_mask = tf.cast(inter_hw > 0, dtype=tf.float32)
    inter_hw = inter_hw * positive_mask
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]
    pred_area = pred_yxhw[..., 2] * pred_yxhw[..., 3]
    grtr_area = grtr_yxhw[..., 2] * grtr_yxhw[..., 3]
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)
    return iou


def compute_iou_general(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: GT bounding boxes in yxhw format (batch, N1, D1(>4))
    :param pred_yxhw: predicted bounding box in yxhw format (batch, N2, D2(>4))
    :return: iou (batch, HWA)
    """
    grtr_yxhw = tf.expand_dims(grtr_yxhw, axis=-2)  # (batch, N1, 1, D1)
    pred_yxhw = tf.expand_dims(pred_yxhw, axis=-3)  # (batch, 1, N2, D2)

    if grtr_tlbr is None:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxhw)  # (batch, N1, 1, D1)
    if pred_tlbr is None:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxhw)  # (batch, 1, N2, D2)

    inter_tl = tf.maximum(grtr_tlbr[..., :2], pred_tlbr[..., :2])       # (batch, N1, N2, 2)
    inter_br = tf.minimum(grtr_tlbr[..., 2:4], pred_tlbr[..., 2:4])     # (batch, N1, N2, 2)
    inter_hw = inter_br - inter_tl                                      # (batch, N1, N2, 2)
    inter_hw = tf.maximum(inter_hw, 0)
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]                    # (batch, N1, N2)

    pred_area = pred_yxhw[..., 2] * pred_yxhw[..., 3]                   # (batch, 1, N2)
    grtr_area = grtr_yxhw[..., 2] * grtr_yxhw[..., 3]                   # (batch, N1, 1)
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)      # (batch, N1, N2)
    return iou


def test_iou_general():
    print("===== start test_iou_general")
    # iou([0.5, 0.5, 0.2, 0.4, 0], [0.6, 0.7, 0.4, 0.2, 0]) = 1/7
    # iou([0.5, 0.5, 0.2, 0.4, 0], [0.4, 0.3, 0.4, 0.4, 0]) = 1/5
    grtr = tf.constant([[[0.5, 0.5, 0.2, 0.4, 0],
                         [0, 0, 0, 0, 1]],
                        [[0.5, 0.5, 0.2, 0.4, 2],
                         [0, 0, 0, 0, 3]],
                        ], dtype=tf.float32)
    pred = tf.constant([[[0, 0, 0, 0, 1],
                         [0.6, 0.7, 0.4, 0.2, 0],
                         [0.4, 0.3, 0.4, 0.4, 0]],
                        [[0, 0, 0, 0, 1],
                         [0.6, 0.7, 0.4, 0.2, 0],
                         [0.4, 0.3, 0.4, 0.4, 0]],
                        ], dtype=tf.float32)

    # EXECUTE
    iou = compute_iou_general(grtr, pred)
    # TEST
    iou = iou.numpy()
    print("iou:", iou)
    # iou has little error due to 1e-5 in "iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)"
    assert np.isclose(iou[0, 0, 1], 1. / 7., atol=0.001)
    assert np.isclose(iou[0, 0, 2], 1. / 5., atol=0.001)
    assert np.isclose(iou[:, 1], 0).all()
    assert np.isclose(iou[:, :, 0], 0).all()
    print("!!! test_iou_general passed")


if __name__ == "__main__":
    test_iou_general()



