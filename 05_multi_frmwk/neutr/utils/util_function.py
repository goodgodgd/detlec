import sys
import numpy as np
import cv2


def to_float_image(image):
    image = image / 255. * 2. - 1.
    return image


def to_uint8_image(image):
    image = ((image + 1) * 255. / 2.).astype(np.uint8)
    return image


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


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

    output = np.concatenate(output, axis=-1)
    output = output.astype(boxes.dtype)
    return output


def slice_feature(feature, channel_composition):
    """
    :param feature: (batch, grid_h, grid_w, anchors, channels)
    :param channel_composition: e.g. {"yxhw": 4, "object": 1, "category": 1}
    :return: {"yxhw": (batch, grid_h, grid_w, anchors, 4), "object": ..., "category": ...}
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    split_indices = [sum(channels[:i+1]) for i in range(len(channels)-1)]
    slices = np.split(feature, split_indices, axis=-1)
    slices = dict(zip(names, slices))  # slices = {'yxhw': (B,H,W,A,4), 'object': (B,H,W,A,1), ...}
    return slices


def compute_iou_general(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: GT bounding boxes in yxhw format (batch, N1, D1(>4))
    :param pred_yxhw: predicted bounding box in yxhw format (batch, N2, D2(>4))
    :return: iou (batch, N1, N2)
    """
    grtr_yxhw = np.expand_dims(grtr_yxhw, axis=-2)  # (batch, N1, 1, D1)
    pred_yxhw = np.expand_dims(pred_yxhw, axis=-3)  # (batch, 1, N2, D2)
    if grtr_tlbr is None:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxhw)  # (batch, N1, 1, D1)
    if pred_tlbr is None:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxhw)  # (batch, 1, N2, D2)

    inter_tl = np.maximum(grtr_tlbr[..., :2], pred_tlbr[..., :2])  # (batch, N1, N2, 2)
    inter_br = np.minimum(grtr_tlbr[..., 2:4], pred_tlbr[..., 2:4])  # (batch, N1, N2, 2)
    inter_hw = inter_br - inter_tl  # (batch, N1, N2, 2)
    inter_hw = np.maximum(inter_hw, 0)
    inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (batch, N1, N2)

    pred_area = pred_yxhw[..., 2] * pred_yxhw[..., 3]  # (batch, 1, N2)
    grtr_area = grtr_yxhw[..., 2] * grtr_yxhw[..., 3]  # (batch, N1, 1)
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)  # (batch, N1, N2)
    return iou



def draw_boxes(image, bboxes, category_names, box_format="yxhw"):
    """
    :param image: (height, width, 3), np.uint8
    :param bboxes: (N, 6), np.float32 (0~1) or np.int32 (pixel scale)
    :param category_names: list of category names
    :param box_format: "yxhw": [y, x, h, w, category] or "2pt": [y1, x1, y2, x2, category]
    """
    image = image.copy()
    bboxes = bboxes.copy()
    if np.max(bboxes[:, :4]) <= 1:
        height, width = image.shape[:2]
        bboxes[:, :4] *= np.array([[height, width, height, width]], np.float32)
    if box_format == "yxhw":
        bboxes = convert_box_format_yxhw_to_tlbr(bboxes)
    bboxes = bboxes[bboxes[:, 2] > 0, :]
    bboxes = bboxes.astype(np.int32)

    for i, bbox in enumerate(bboxes):
        pt1, pt2 = (bbox[1], bbox[0]), (bbox[3], bbox[2])
        cat_index = int(bbox[5])
        category = category_names[cat_index]
        image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), thickness=2)
        image = cv2.putText(image, f"{i}{category}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

