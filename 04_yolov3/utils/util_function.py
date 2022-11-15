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
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.uint8)


def convert_box_format_tlbr_to_yxhw(boxes_tlbr):
    """
    :param boxes_tlbr: type=tf.Tensor or np.array, shape=(numbox, channel) or (batch, numbox, channel)
    :return:
    """
    boxes_yx = (boxes_tlbr[..., 0:2] + boxes_tlbr[..., 2:4]) / 2    # center y,x
    boxes_hw = boxes_tlbr[..., 2:4] - boxes_tlbr[..., 0:2]          # y2,x2 = y1,x1 + h,w
    output = [boxes_yx, boxes_hw]
    output = concat_box_output(output, boxes_tlbr)
    return output


def convert_box_format_yxhw_to_tlbr(boxes_yxhw):
    """
    :param boxes_yxhw: type=tf.Tensor or np.array, shape=(numbox, channel) or (batch, numbox, channel)
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


def slice_features_and_merge_dims(featin, composition):
    """
    :param featin: [(batch, grid_h, grid_w, anchors, channels) x 3]
    :param composition: e.g. {"yxhw": 4, "object": 1, "category": 1}
    :return: {"yxhw": [(batch, grid_h * grid_w * anchors, 4) x 3], "object": ..., "category": ...}
    """
    newfeat = []
    for scale_data in featin:
        slices = slice_feature(scale_data, composition)
        slices = {key: merge_dim_hwa(fmap) for key, fmap in slices.items()}
        newfeat.append(slices)
    featout = scale_align_featmap(newfeat)
    return featout


def slice_feature(feature, channel_composition):
    """
    :param feature: (batch, grid_h, grid_w, anchors, channels)
    :param channel_composition: e.g. {"yxhw": 4, "object": 1, "category": 1}
    :return: {"yxhw": (batch, grid_h, grid_w, anchors, 4), "object": ..., "category": ...}
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    slices = tf.split(feature, channels, axis=-1)
    slices = dict(zip(names, slices))  # slices = {'yxhw': (B,H,W,A,4), 'object': (B,H,W,A,1), ...}
    return slices


def merge_dim_hwa(features):
    """
    :param features: (batch, grid_h, grid_w, anchor, channels)
    :return: (batch, grid_h * grid_w * anchor, channels)
    """
    batch, grid_h, grid_w, anchor, featdim = features.shape
    merged_feat = tf.reshape(features, (batch, grid_h*grid_w*anchor, featdim))
    return merged_feat


def scale_align_featmap(features):
    """
    :param features: [{"yxhw": (B,HWA,4), "object": (B,HWA,1), "category": (B,HWA,1)} x 3]
    :return: {"yxhw": [(B,HWA,4)x3], "object": [(B,HWA,1)x3], "category": [(B,HWA,1)x3]}
    """
    align_feat = dict()
    for slice_key in features[0].keys():
        align_feat[slice_key] = [features[scale_index][slice_key] for scale_index in range(len(features))]
    return align_feat


def compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: ordered GT bounding boxes in yxhw format (batch, HWA, 4)
    :param pred_yxhw: ordered predicted bounding box in yxhw format (batch, HWA, 4)
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
    iou = inter_area / (pred_area + grtr_area - inter_area + 0.00001)
    return iou


def print_structure(title, data, key=""):
    if isinstance(data, list):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif isinstance(data, dict):
        for subkey, datum in data.items():
            print_structure(title, datum, f"{key}/{subkey}")
    elif isinstance(data, tuple):
        for i, datum in enumerate(data):
            print_structure(title, datum, f"{key}/{i}")
    elif type(data) == np.ndarray:
        print(title, key, data.shape, "np", data.dtype)
    elif tf.is_tensor(data):
        print(title, key, data.shape, "tf", data.dtype)
    else:
        print(title, key, data)


def convert_to_numpy(data):
    if isinstance(data, list):
        for i, datum in enumerate(data):
            data[i] = convert_to_numpy(datum)
    elif isinstance(data, dict):
        for key, datum in data.items():
            data[key] = convert_to_numpy(datum)
    elif tf.is_tensor(data):
        return data.numpy()
    return data




