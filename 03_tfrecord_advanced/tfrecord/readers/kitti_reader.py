import os.path as op
import numpy as np
from glob import glob
import cv2

from tfrecord.readers.reader_base import DataReaderBase, DriveManagerBase
from config import Config as cfg

KITTI_CATEGORIES = {"Pedestrian": 0, "Car": 1, "Van": 2, "Cyclist": 3}


class KittiDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        kitti_split = "training" if self.split == "train" else "testing"
        return [op.join(self.datapath, kitti_split, "image_2")]

    def get_drive_name(self, drive_index):
        raise NotImplementedError()


class KittiReader(DataReaderBase):
    def __init__(self, drive_path, split):
        super().__init__(drive_path, split)

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path, split):
        self.frame_names = glob(op.join(drive_path, "*.png"))
        self.frame_names.sort()
        print("[KittiReader.init_drive] # frames:", len(self.frame_names), "first:", self.frame_names[0])

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index):
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            bboxes = []
            for line in lines:
                bbox = self.extract_box(line)
                bboxes.append(bbox)

        bboxes = np.array(bboxes)
        return bboxes

    def extract_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category = self.map_category(raw_label[0])
        x1 = round(float(raw_label[4]))
        y1 = round(float(raw_label[5]))
        x2 = round(float(raw_label[6]))
        y2 = round(float(raw_label[7]))
        return np.array([x1, y1, x2, y2, category], dtype=np.int32)

    def map_category(self, srclabel):
        if srclabel in KITTI_CATEGORIES:
            return KITTI_CATEGORIES[srclabel]
        else:
            return cfg.Dataset.INVALID_CATEGORY
