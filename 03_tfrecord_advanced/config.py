import os.path as op
import numpy as np


class Config:
    class Paths:
        RAW_DATA = {"kitti": "/media/ian/Ian4T/dataset/kitti_detection"}
        RESULT_ROOT = "/home/ian/workspace/detlec/dataset"
        TFRECORD = op.join(RESULT_ROOT, "tfrecord")

    class Dataset:
        DATASETS_FOR_TFRECORD = {"kitti": ("train", "val")}
        KITTI_CATEGORY_MAP = {"Pedestrian": 0, "Car": 1, "Van": 2, "Cyclist": 3}
        KITTI_CATEGORIES = ["Pedestrian", "Car", "Van", "Cyclist"]
        INPUT_RESOLUTIONS = {"kitti": (256, 832)}
        INVALID_CATEGORY = 100
        MAX_BBOX_PER_IMAGE = 20

    class Model:
        FEATURE_SCALES = {"feature_l": 64, "feature_m": 32, "feature_s": 16}
        FEATURE_ORDER = ["feature_s", "feature_m", "feature_l"]
        ANCHORS_PIXEL = np.array([[13, 10], [30, 16], [23, 33],
                                  [61, 30], [45, 62], [119, 59],
                                  [90, 116], [198, 156], [326, 373]])

    class Train:
        BATCH_SIZE = 2

    @classmethod
    def get_img_shape(cls, code="HW", dataset="kitti", scale_div=1):
        imsize = cls.Dataset.INPUT_RESOLUTIONS[dataset]
        if code == "H":
            return imsize[0] // scale_div
        elif code == "W":
            return imsize[1] // scale_div
        elif code == "HW":
            return imsize[0] // scale_div, imsize[1] // scale_div
        elif code == "WH":
            return imsize[1] // scale_div, imsize[0] // scale_div
        elif code == "HWC":
            return imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code == "BHWC":
            return cls.Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
        else:
            assert 0, f"Invalid code: {code}"


