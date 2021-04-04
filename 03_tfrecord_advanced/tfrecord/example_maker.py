import numpy as np
import cv2

import tfrecord.tfr_util as tu
import tfrecord.preprocess as pr
from config import Config as cfg


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg):
        self.data_reader = data_reader
        self.feat_scales = cfg.Model.FEATURE_SCALES
        self.feat_order = cfg.Model.FEATURE_ORDER
        self.anchors_pixel = cfg.Model.ANCHORS_PIXEL
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       dataset_cfg=dataset_cfg,
                                                       category_names=cfg.Dataset.CATEGORY_NAMES,
                                                       max_bbox=cfg.Dataset.MAX_BBOX_PER_IMAGE,
                                                       )

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        raw_hw_shape = example["image"].shape[:2]
        example["bboxes"] = self.data_reader.get_bboxes(index)
        example = self.preprocess_example(example)
        example = self.assign_bbox_over_feature_map(example, raw_hw_shape)
        if index % 100 == 10:
            self.show_example(example)
        return example

    def assign_bbox_over_feature_map(self, example, raw_hw_shape):
        # anchors are derived from raw image shape
        # anchors_ratio: anchor sizes normalized by image size (0~1)
        anchors_ratio = self.anchors_pixel / np.array([raw_hw_shape])
        # feature map sizes are derived from tfrecord image shape
        # feat_sizes: {"feature_l": tfr_hw_shape / 32, ...}
        tfr_hw_shape = example["image"].shape[:2]
        feat_sizes = {key: np.array(tfr_hw_shape) // scale for key, scale in self.feat_scales.items()}
        gt_features = self.make_gt_feature_map(example["bboxes"], anchors_ratio, feat_sizes, self.feat_order)
        example.update(gt_features)
        return example

    def make_gt_feature_map(self, bboxes, anchors, feat_sizes, feat_order):
        """
        :param bboxes: bounding boxes in image ratio (0~1) [cy, cx, h, w, category] (N, 5)
        :param anchors: anchors in image ratio (0~1) (9, 2)
        :param feat_sizes: feature map sizes for 3 feature maps {"feature_l": [grid_h, grid_w], ...}
        :param feat_order: feature map order ["feature_s", "feature_m", "feature_l"]
        :return:
        """
        boxes_hw = bboxes[:, np.newaxis, 2:4]       # (N, 1, 5)
        anchors_hw = anchors[np.newaxis, :, :]      # (1, 9, 2)
        inter_hw = np.minimum(boxes_hw, anchors_hw) # (N, 9, 2)
        inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]  # (N, 9)
        union_area = boxes_hw[:, :, 0] * boxes_hw[:, :, 1] + anchors_hw[:, :, 0] * anchors_hw[:, :, 1] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=1)
        num_scales = len(feat_order)
        gt_features = {feat_name: np.zeros((feat_shape[0], feat_shape[1], num_scales, 6), dtype=np.float32)
                       for feat_name, feat_shape in feat_sizes.items()}

        for anchor_index, bbox in zip(best_anchor_indices, bboxes):
            scale_index = anchor_index // num_scales
            anchor_index_in_scale = anchor_index % num_scales
            feat_name = feat_order[scale_index]
            feat_map = gt_features[feat_name]
            # bbox: [y, x, h, w, category]
            grid_yx = (bbox[:2] * feat_sizes[feat_name]).astype(np.int32)
            assert (grid_yx >= 0).all() and (grid_yx < feat_sizes[feat_name]).all()
            # bbox: [y, x, h, w, 1, category]
            box_at_grid = np.insert(bbox, 4, 1)
            feat_map[grid_yx[0], grid_yx[1], anchor_index_in_scale] = box_at_grid
            gt_features[feat_name] = feat_map
        return gt_features

    def show_example(self, example):
        category_names = cfg.Dataset.CATEGORY_NAMES
        image = tu.draw_boxes(example["image"], example["bboxes"], category_names)
        cv2.imshow("image with bboxes", image)

        features = []
        for feat_name in cfg.Model.FEATURE_ORDER:
            feature = example[feat_name]
            feature = feature[feature[..., 4] > 0]      # objectness == 1
            features.append(feature)
        feat_boxes = np.concatenate(features, axis=0)
        image = tu.draw_boxes(example["image"], feat_boxes, category_names)
        cv2.imshow("image with feature bboxes", image)
        cv2.waitKey(100)

