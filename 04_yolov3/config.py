import os.path as op
import numpy as np
from parameters import ParameterPool


class Config:
    class Paths:
        RESULT_ROOT = "/home/ian/workspace/detlec/dataset"
        TFRECORD = op.join(RESULT_ROOT, "tfrecord")
        CHECK_POINT = op.join(RESULT_ROOT, "ckpt")

    class Tfrdata:
        DATASETS_FOR_TFRECORD = {"kitti": ("train", "val")}
        MAX_BBOX_PER_IMAGE = 20
        CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"]
        SHARD_SIZE = 2000

    class Datasets:
        # specific dataset configs MUST have the same items
        class Kitti:
            NAME = "kitti"
            PATH = "/media/ian/Ian4T/dataset/kitti_detection"
            CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist"]
            CATEGORY_REMAP = {"Pedestrian": "Person", "Cyclist": "Bicycle"}
            INPUT_RESOLUTION = (256, 832)   # (4,13) * 64
            CROP_TLBR = [0, 0, 0, 0]        # crop [top, left, bottom, right] or [y1 x1 y2 x2]

        DATASET_CONFIGS = {"kitti": Kitti}

        @classmethod
        def get_dataset_config(cls, dataset):
            return cls.DATASET_CONFIGS[dataset]

    class Model:
        class Output:
            FEATURE_SCALES = {"feature_l": 32, "feature_m": 16, "feature_s": 8}
            FEATURE_ORDER = ["feature_s", "feature_m", "feature_l"]
            ANCHORS_PIXEL = ParameterPool.ANCHOR.COCO

        class Composition:
            BACKBONE = "darknet53"
            HEAD = "FPN"
            CONV_ARGS = {"activation": "leaky_relu", "activation_param": 0.1}

    class Train:
        CKPT_NAME = "yolo"
        MODE = ["eager", "graph"][0]
        BATCH_SIZE = 2
        TRAINING_PLAN = [
            ("kitti", 10, 0.0001, ParameterPool.LOSS.STANDARD, True),
            ("kitti", 10, 0.00001, ParameterPool.LOSS.STANDARD, True)
        ]

    @staticmethod
    def config_summary():
        # return dict of important parameters
        pass

    @classmethod
    def get_img_shape(cls, code="HW", dataset="kitti", scale_div=1):
        dataset_cfg = cls.Datasets.get_dataset_config(dataset)
        imsize = dataset_cfg.INPUT_RESOLUTION
        code = code.upper()
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


