import os.path as op
import numpy as np
from glob import glob
import cv2

from tfrecord.readers.reader_base import DatasetReaderBase
import tfrecord.tfr_util as tu
import utils.util_class as uc


class KittiReader(DatasetReaderBase):
    def __init__(self, data_path, split, dataset_cfg):
        super().__init__(data_path, split, dataset_cfg)
        self.cur_frame = -1
        self.bbox = None
        self.category = None

    def init_frames(self, data_path, split):
        frame_names = glob(op.join(data_path, "*.png"))
        frame_names.sort()
        if split == "train":
            frame_names = frame_names[:-500]
        else:
            frame_names = frame_names[-500:]
        print("[KittiReader.init_frames] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index):
        """
        :return: bounding boxes in 'yxhw' format
        """
        self.read_text(index)
        return self.bbox

    def get_categories(self, index):
        self.read_text(index)
        return self.category

    def read_text(self, index):
        if self.cur_frame == index:
            return
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            bboxes = [self.read_line(line) for line in lines]
            bboxes = [bbox for bbox in bboxes if bbox is not None]
        if not bboxes:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes = np.array(bboxes)
        self.bbox = bboxes[:, :-1]
        self.category = bboxes[:, -1:]
        self.cur_frame = index
        return bboxes

    def read_line(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None
        category_index = self.dataset_cfg.CATEGORIES_TO_USE.index(category_name)
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))
        bbox = np.array([(y1+y2)/2, (x1+x2)/2, y2-y1, x2-x1, category_index], dtype=np.int32)
        return bbox

