import numpy as np
import torch


def convert_box_format_tlbr_to_yxhw(boxes_tlbr):
    """
    :param boxes_tlbr: (batch, 4+?, N)
    :return: (batch, 4+?, N)
    """
    tlbr_components = torch.split(boxes_tlbr, 2, dim=-2)
    boxes_yx = (tlbr_components[0] + tlbr_components[1]) / 2    # yx = (y1x1 + y2x2)/2
    boxes_hw = tlbr_components[1] - tlbr_components[0]          # hw = y2x2 - y1x1
    output = [boxes_yx, boxes_hw]
    if len(tlbr_components) > 2:
        output += list(tlbr_components[2:])
    output = torch.cat(output, dim=-2)
    return output


def convert_box_format_yxhw_to_tlbr(boxes_yxhw):
    """
    :param boxes_yxhw: (batch, 4+?, N)
    :return: (batch, 4+?, N)
    """
    yxhw_components = torch.split(boxes_yxhw, 2, dim=-2)
    boxes_tl = yxhw_components[0] - yxhw_components[1] / 2      # y1x1 = cy,cx + h/2,w/2
    boxes_br = yxhw_components[0] + yxhw_components[1] / 2      # y1x1 = cy,cx + h/2,w/2
    output = [boxes_tl, boxes_br]
    if len(boxes_yxhw) > 2:
        output += list(boxes_yxhw[2:])
    output = torch.cat(output, dim=-2)
    return output


def slice_features_and_merge_dims(featin, composition):
    """
    :param featin: [(batch, channels, anchors, grid_h, grid_w) x 3]
    :param composition: e.g. {"yxhw": 4, "object": 1, "category": 1}
    :return: {"yxhw": [(batch, 4, anchors * grid_h * grid_w) x 3], "object": ..., "category": ...}
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
    :param feature: (batch, channels, anchors, grid_h, grid_w)
    :param channel_composition: e.g. {"yxhw": 4, "object": 1, "category": 1}
    :return: {"yxhw": (batch, 4, anchors, grid_h, grid_w), "object": ..., "category": ...}
    """
    names, channels = list(channel_composition.keys()), list(channel_composition.values())
    slices = torch.split(feature, channels, dim=1)
    slices = dict(zip(names, slices))  # slices = {'yxhw': (B,4,A,H,W,4), 'object': (B,1,A,H,W), ...}
    return slices


def merge_dim_hwa(feature):
    """
    :param feature: (batch, channels, anchor, grid_h, grid_w)
    :return: (batch, channels, anchor * grid_h * grid_w)
    """
    batch, channel, anchor, grid_h, grid_w = feature.shape
    merged_feat = feature.view(batch, channel, anchor*grid_h*grid_w)
    return merged_feat


def scale_align_featmap(features):
    """
    :param features: [{"yxhw": (B,4,HWA), "object": (B,1,HWA), "category": (B,1,HWA)} x 3]
    :return: {"yxhw": [(B,4,HWA)x3], "object": [(B,1,HWA)x3], "category": [(B,1,HWA)x3]}
    """
    align_feat = dict()
    for slice_key in features[0].keys():
        align_feat[slice_key] = [features[scale_index][slice_key] for scale_index in range(len(features))]
    return align_feat


def compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr=None, pred_tlbr=None):
    """
    :param grtr_yxhw: ordered GT bounding boxes in yxhw format (batch, 4, HWA)
    :param pred_yxhw: ordered predicted bounding box in yxhw format (batch, 4, HWA)
    :return: iou (batch, HWA)
    """
    if grtr_tlbr is None:
        grtr_tlbr = convert_box_format_yxhw_to_tlbr(grtr_yxhw)
    if pred_tlbr is None:
        pred_tlbr = convert_box_format_yxhw_to_tlbr(pred_yxhw)
    inter_tl = torch.maximum(grtr_tlbr[:, :2], pred_tlbr[:, :2])
    inter_br = torch.minimum(grtr_tlbr[:, 2:4], pred_tlbr[:, 2:4])
    inter_hw = inter_br - inter_tl
    positive_mask = (inter_hw > 0).type(torch.float32)
    inter_hw = inter_hw * positive_mask
    inter_area = inter_hw[:, 0] * inter_hw[:, 1]
    pred_area = pred_yxhw[:, 2] * pred_yxhw[:, 3]
    grtr_area = grtr_yxhw[:, 2] * grtr_yxhw[:, 3]
    iou = inter_area / (pred_area + grtr_area - inter_area + 0.00001)
    return iou


def convert_to_numpy(data):
    if isinstance(data, list):
        for i, datum in enumerate(data):
            data[i] = convert_to_numpy(datum)
    elif isinstance(data, dict):
        for key, datum in data.items():
            data[key] = convert_to_numpy(datum)
    elif torch.is_tensor(data):
        data =  data.detach().cpu().numpy()
        if data.ndim > 2:
            data = np.moveaxis(data, 1, -1)   # convert channel first to channel last format
    return data


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
    elif torch.is_tensor(data):
        print(title, key, data.shape, "pt", data.dtype)
    else:
        print(title, key, data)



