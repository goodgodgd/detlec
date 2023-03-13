import numpy as np
import cv2

import neutr.utils.util_function as nuf
import neutr.data.preprocess as pr
import config as cfg


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 anchors_ratio=cfg.ModelOutput.ANCHORS_RATIO,
                 category_names=cfg.DataCommon.CATEGORY_NAMES,
                 max_bbox=cfg.DataCommon.MAX_BBOX_PER_IMAGE):
        self.data_reader = data_reader
        self.feat_scales = feat_scales
        self.anchors_ratio = anchors_ratio
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       dataset_cfg=dataset_cfg,
                                                       category_names=category_names,
                                                       max_bbox=max_bbox)

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        bbox = self.data_reader.get_bboxes(index)
        category = self.data_reader.get_categories(index)
        example["inst"] = np.concatenate([bbox, np.ones_like(category), category], axis=-1, dtype=np.float32)
        example = self.preprocess_example(example)
        if index % 100 == 10:
            self.show_example(example)
        return example

    def show_example(self, example):
        category_names = cfg.DataCommon.CATEGORY_NAMES
        image = nuf.draw_boxes(example["image"], example["inst"], category_names)
        cv2.imshow("image with bboxes", image)
        cv2.waitKey(100)

