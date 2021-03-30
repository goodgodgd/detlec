import numpy as np
import cv2

import util_class as uc
import util_function as uf
import tfrecord.tfr_util as tu
from config import Config as cfg


class ExampleMaker:
    def __init__(self, data_reader, hw_shape):
        self.data_reader = data_reader
        self.hw_shape = hw_shape
        self.max_bbox = cfg.Dataset.MAX_BBOX_PER_IMAGE
        self.feat_scales = cfg.Model.FEATURE_SCALES
        self.feat_order = cfg.Model.FEATURE_ORDER
        self.anchors_pixel = cfg.Model.ANCHORS_PIXEL
        self.crop_and_resize = CropperAndResizer(hw_shape)

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        raw_hw_shape = example["image"].shape[:2]
        example["bboxes"] = self.data_reader.get_bboxes(index)
        example = self.crop_and_resize(example)
        # y1x1y2x2 -> yxhw
        example["bboxes"] = uf.convert_box_format_2pt_to_yxhw(example["bboxes"])
        example = self.assign_bbox_over_feature_map(example, raw_hw_shape)
        example["bboxes"] = self.fix_bbox_len(example["bboxes"])
        if index % 100 == 10:
            self.show_example(example)
        return example

    def assign_bbox_over_feature_map(self, example, raw_hw_shape):
        # anchors are derived from raw image shape
        # anchors_ratio: anchor sizes normalized by image size (0~1)
        anchors_ratio = self.anchors_pixel / np.array([raw_hw_shape])
        # feature map sizes are derived from tfrecord image shape
        # feat_sizes: {"feature_l": hw_shape / 64, ...}
        feat_sizes = {key: np.array(self.hw_shape) // scale for key, scale in self.feat_scales.items()}
        large, medium, small = make_gt_features(example["bboxes"], anchors_ratio, feat_sizes, self.feat_order)
        example["feature_l"] = large
        example["feature_m"] = medium
        example["feature_s"] = small
        return example

    def fix_bbox_len(self, bboxes):
        if bboxes.shape[0] < self.max_bbox:
            new_bboxes = np.zeros((self.max_bbox, 5), dtype=np.float32)
            new_bboxes[:bboxes.shape[0]] = bboxes
            return new_bboxes
        else:
            return bboxes

    def show_example(self, example):
        height, width = example["image"].shape[:2]
        bboxes = uf.convert_box_format_yxhw_to_2pt(example["bboxes"])
        bboxes *= np.array([[height, width, height, width, 1]])
        bboxes = bboxes.astype(np.int32)
        bboxes = bboxes[bboxes[:, 0] > 0, :]
        print("[show example] bbox 2pts:\n", bboxes)
        boxed_image = tu.draw_boxes(example["image"], bboxes, cfg.Dataset.KITTI_CATEGORIES)
        cv2.imshow("image with bboxes", boxed_image)
        cv2.waitKey(100)


class CropperAndResizer:
    def __init__(self, hw_shape):
        self.hw_shape = hw_shape

    def __call__(self, example):
        crop_yxhw, resize_ratio = self.prepare_cnr(example["image"])
        image = self.cnr_image(example["image"], crop_yxhw)
        bboxes = self.cnr_bboxes(example["bboxes"], crop_yxhw, resize_ratio)
        # return new example
        example = {"image": image, "bboxes": bboxes}
        return example

    def prepare_cnr(self, image):
        src_hw = np.array(image.shape[:2])          # [375, 1242]
        dst_hw = np.array(self.hw_shape)            # [256, 832]
        hw_ratio = dst_hw / src_hw                  # [0.68, 0.67]
        resize_ratio = np.max(hw_ratio)             # 0.68
        crop_src_hw = dst_hw / resize_ratio         # [375, 1218.75]
        crop_src_hw = crop_src_hw.astype(np.int32)  # [375, 1218]
        crop_src_yx = ((src_hw - crop_src_hw) // 2).astype(np.int32)    # [0, 12]
        crop_yxhw = np.concatenate([crop_src_yx, crop_src_hw], axis=0)  # [0, 12, 375, 1218]
        return crop_yxhw, resize_ratio

    def cnr_image(self, image, crop_yxhw):
        # crop image: image[0:375, 12:1230] (375, 1218)
        image = image[crop_yxhw[0]:crop_yxhw[0] + crop_yxhw[2], crop_yxhw[1]:crop_yxhw[1] + crop_yxhw[3]]
        image = cv2.resize(image, (self.hw_shape[1], self.hw_shape[0]))  # (256, 832)
        return image

    def cnr_bboxes(self, bboxes, crop_yxhw, resize_ratio):
        bboxes = bboxes.astype(np.float)    # [y1 x1 y2 x2 category]
        # apply crop and resize to boxes
        bboxes[:, 0] = (bboxes[:, 0] - crop_yxhw[0]) * resize_ratio
        bboxes[:, 1] = (bboxes[:, 1] - crop_yxhw[1]) * resize_ratio
        bboxes[:, 2] = (bboxes[:, 2] - crop_yxhw[0]) * resize_ratio
        bboxes[:, 3] = (bboxes[:, 3] - crop_yxhw[1]) * resize_ratio
        # filter boxes outside image
        centers_yx = np.stack([bboxes[:, 0] + bboxes[:, 2] / 2, bboxes[:, 1] + bboxes[:, 3] / 2], axis=1).astype(np.int32)
        inside = (centers_yx[:, 0] >= 0) & (centers_yx[:, 0] < self.hw_shape[0]) & \
                 (centers_yx[:, 1] >= 0) & (centers_yx[:, 1] < self.hw_shape[1])
        bboxes = bboxes[inside]
        if bboxes.size == 0:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        # clip into image range
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], self.hw_shape[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], self.hw_shape[1])
        # pixel to ratio (0~1)
        bboxes[:, 0] /= self.hw_shape[0]
        bboxes[:, 1] /= self.hw_shape[1]
        bboxes[:, 2] /= self.hw_shape[0]
        bboxes[:, 3] /= self.hw_shape[1]
        return bboxes.astype(np.float32)


def make_gt_features(bboxes, anchors, feat_sizes, feat_order):
    """
    :param bboxes: bounding boxes in image ratio (0~1) [cy, cx, h, w, category] (N, 5)
    :param anchors: anchors in image ratio (0~1) (9, 2)
    :param feat_sizes: feature map sizes for 3 feature maps {"feature_l": np.array([grid_h, grid_w]), ...}
    :param feat_order: feature map order to map index to feature map name
    :return:
    """
    boxes_hw = bboxes[:, np.newaxis, 2:4]       # (N, 1, 5)
    anchors_hw = anchors[np.newaxis, :, :]      # (1, 9, 2)
    inter_hw = np.minimum(boxes_hw, anchors_hw)      # (N, 9, 2)
    inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]  # (N, 9)
    union_area = boxes_hw[:, :, 0] * boxes_hw[:, :, 1] + anchors_hw[:, :, 0] * anchors_hw[:, :, 1] - inter_area
    iou = inter_area / union_area
    best_anchor_indices = np.argmax(iou, axis=1)
    num_scales = len(feat_order)
    bbox_dict = {feat_name: np.zeros((feat_shape[0], feat_shape[1], num_scales, 6), dtype=np.float32)
                 for feat_name, feat_shape in feat_sizes.items()}
    for anchor_index, bbox in zip(best_anchor_indices, bboxes):
        scale_index = anchor_index // num_scales
        anchor_index_in_scale = anchor_index % num_scales
        feat_name = feat_order[scale_index]
        feat_map = bbox_dict[feat_name]
        # bbox: [y, x, h, w, category] in ratio
        grid_yx = (bbox[:2] * feat_sizes[feat_name]).astype(np.int32)
        assert (grid_yx >= 0).all() and (grid_yx < feat_sizes[feat_name]).all()
        # bbox: [y, x, h, w, 1, category] in ratio
        box_at_grid = np.insert(bbox, 4, 1)
        feat_map[grid_yx[0], grid_yx[0], anchor_index_in_scale] = box_at_grid
        bbox_dict[feat_name] = feat_map
    return bbox_dict

