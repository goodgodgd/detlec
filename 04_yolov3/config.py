import os.path as op
import parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/home/ri-bear/workspace/detlec/result"
    TFRECORD = op.join(RESULT_ROOT, "tfrecord")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")


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
    TARGET_DATASET = "kitti"

    @classmethod
    def get_dataset_config(cls, dataset):
        return cls.DATASET_CONFIGS[dataset]


class Tfrdata:
    DATASETS_FOR_TFRECORD = {"kitti": ("train", "val")}
    MAX_BBOX_PER_IMAGE = 20
    CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"]
    SHARD_SIZE = 2000
    ANCHORS_PIXEL = None  # assigned by set_anchors()

    @classmethod
    def set_anchors(cls):
        basic_anchor = params.Anchor.COCO_YOLOv3
        target_dataset = Datasets.TARGET_DATASET
        dataset_cfg = Datasets.get_dataset_config(target_dataset)
        input_resolution = np.array(dataset_cfg.INPUT_RESOLUTION, dtype=np.float32)
        anchor_resolution = np.array(params.Anchor.COCO_RESOLUTION, dtype=np.float32)
        scale = np.min(input_resolution / anchor_resolution)
        Tfrdata.ANCHORS_PIXEL = np.around(basic_anchor * scale, 1)
        print("[set_anchors] anchors in pixel:\n", Tfrdata.ANCHORS_PIXEL)


class Model:
    class Output:
        FEATURE_SCALES = {"feature_s": 8, "feature_m": 16, "feature_l": 32}
        FEATURE_ORDER = ["feature_s", "feature_m", "feature_l"]
        NUM_ANCHORS_PER_SCALE = 3
        GRTR_CHANNEL_COMPOSITION = {'yxhw': 4, 'object': 1, 'category': 1}
        PRED_CHANNEL_COMPOSITION = {'yxhw': 4, 'object': 1, 'category': len(Tfrdata.CATEGORY_NAMES)}
        FEATURE_CHANNELS = sum([val for key, val in PRED_CHANNEL_COMPOSITION.items()])
        GRTR_BBOX_COMPOSITION = {'yxhw': 4, 'category': 1}
        PRED_BBOX_COMPOSITION = {'yxhw': 4, 'category': 1, 'object': 1, 'ctgr_prob': 1, 'score': 1}

        @classmethod
        def get_channel_composition(cls, is_gt: bool):
            if is_gt:
                return cls.GRTR_CHANNEL_COMPOSITION
            else:
                return cls.PRED_CHANNEL_COMPOSITION

        @classmethod
        def get_bbox_composition(cls, is_gt: bool):
            if is_gt:
                return cls.GRTR_BBOX_COMPOSITION
            else:
                return cls.PRED_BBOX_COMPOSITION

    class Structure:
        BACKBONE = "Darknet53"
        HEAD = "FPN"
        BACKBONE_CONV_ARGS = {"activation": "leaky_relu", "scope": "back"}
        HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}


class Train:
    CKPT_NAME = "yolo3"
    MODE = ["eager", "graph"][1]
    BATCH_SIZE = 1
    TRAINING_PLAN = params.TrainingPlan.KITTI_SIMPLE
    DETAIL_LOG_EPOCHS = list(range(0, 100, 2))


class NMS:
    MAX_OUT = 30
    IOU_THRESH = 0.5
    SCORE_THRESH = 0.2


class Validation:
    IOU_THRESH = 0.5
    VAL_EPOCH = "latest"


def summary():
    # return dict of important parameters
    pass


def get_img_shape(code="HW", dataset="kitti", scale_div=1):
    dataset_cfg = Datasets.get_dataset_config(dataset)
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
        return Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
    else:
        assert 0, f"Invalid code: {code}"


def get_valid_category_mask(dataset="kitti"):
    """
    :param dataset: dataset name
    :return: binary mask e.g. when
        Tfrdata.CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"] and
        Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
        Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
        this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
        but "Bicycle" is not
    """
    dataset_cfg = Datasets.get_dataset_config(dataset)
    renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                          for categ in dataset_cfg.CATEGORIES_TO_USE]

    mask = np.zeros((len(Tfrdata.CATEGORY_NAMES), ), dtype=np.int32)
    for categ in renamed_categories:
        if categ in Tfrdata.CATEGORY_NAMES:
            index = Tfrdata.CATEGORY_NAMES.index(categ)
            mask[index] = 1
    return mask


Tfrdata.set_anchors()
