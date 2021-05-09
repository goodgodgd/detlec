import sys
import numpy as np
import tensorflow as tf

from config import Config as cfg


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
    # Note the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32) * 2 - 1


def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype((im_tensor + 1.) / 2., dtype=tf.uint8)


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
    num, dim = boxes.shape
    # if there is more than bounding box, append it  e.g. category, distance
    if dim > 4:
        auxi_data = boxes[4:]
        output.append(auxi_data)

    if tf.is_tensor(boxes):
        output = tf.concat(output, axis=-2)
        output = tf.cast(output, boxes.dtype)
    else:
        output = np.concatenate(output, axis=-2)
        output = output.astype(boxes.dtype)
    return output


def slice_features(feature, output_composition=cfg.Model.Output.OUT_COMPOSITION):
    index = 0
    slices = dict()
    for name, channel in output_composition:
        slices[name] = feature[..., index:index+channel]
        index += channel
    return slices


def merge_dim_hwa(feature_map):
    """
    :param feature_map: (batch, grid_h, grid_w, anchor, 5+K)
    :return: (batch, grid_h * grid_w * anchor, 5+K)
    """
    batch, grid_h, grid_w, anchor, featdim = feature_map.shape
    merged_feat = tf.reshape(feature_map, (batch, grid_h*grid_w*anchor, featdim))
    return merged_feat


def compute_iou_aligned(grtr_yxwh, pred_yxwh, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxwh: ordered GT bounding boxes in yxhw format (batch, HWA, 4)
    :param pred_yxwh: ordered predicted bounding box in yxhw format (batch, HWA, 4)
    :return: iou (batch, HWA)
    """
    if not grtr_tlbr:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxwh)
    if not pred_tlbr:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxwh)
    inter_tl_min = tf.maximum(grtr_tlbr[:2], pred_tlbr[:2])
    inter_br_max = tf.maximum(grtr_tlbr[2:4], pred_tlbr[2:4])
    inter_hw = inter_br_max - inter_tl_min
    positive_mask = tf.cast(inter_hw > 0, dtype=tf.float32)
    inter_hw = inter_hw * positive_mask
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]
    pred_area = pred_yxwh[..., 2] * pred_yxwh[..., 3]
    grtr_area = grtr_yxwh[..., 2] * grtr_yxwh[..., 3]
    iou = inter_area / (pred_area + grtr_area - inter_area + 0.00001)
    return iou

