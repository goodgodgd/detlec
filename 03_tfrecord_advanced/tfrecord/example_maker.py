import numpy as np
import cv2

import tfrecord.tfr_util as tu


class ExampleMaker:
    def __init__(self, data_reader, hw_shape, max_bbox):
        self.data_reader = data_reader
        self.hw_shape = hw_shape
        self.max_bbox = max_bbox

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        example["bbox"] = self.data_reader.get_bboxes(index)
        example = self.resize_example(example, self.hw_shape)
        example["bbox"] = self.fix_bbox_len(example["bbox"])
        if index % 100 == 10:
            self.show_example(example)
        return example

    def resize_example(self, example, hw_shape):
        hw_ratio = np.array(hw_shape) / np.array(example["image"].shape[:2])
        bboxes = example["bbox"].astype(np.float)    # np array [y1 x1 y2 x2 category]
        bboxes[:, 0] *= hw_ratio[0]
        bboxes[:, 1] *= hw_ratio[1]
        bboxes[:, 2] *= hw_ratio[0]
        bboxes[:, 3] *= hw_ratio[1]
        example["bbox"] = np.round(bboxes).astype(np.int32)
        example["image"] = cv2.resize(example["image"], (hw_shape[1], hw_shape[0]))
        return example

    def fix_bbox_len(self, bboxes):
        if bboxes.shape[0] < self.max_bbox:
            new_bboxes = np.zeros((self.max_bbox, 5), dtype=np.int32)
            new_bboxes[:bboxes.shape[0]] = bboxes
            return new_bboxes
        else:
            return bboxes

    def show_example(self, example):
        boxed_image = tu.draw_boxes(example["image"], example["bbox"])
        cv2.imshow("image with bboxes", boxed_image)
        cv2.waitKey(100)
