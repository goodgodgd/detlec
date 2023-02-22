import os.path as op
import parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "F:/work/result"
    TFRECORD = op.join(RESULT_ROOT, "tfrecord")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")


class Datasets:
    # specific dataset configs MUST have the same items
    class Kitti:
        NAME = "kitti"
        PATH = "F:/work/dataset/kitti"
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


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    NUM_ANCHORS_PER_SCALE = 3
    GRTR_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1}
    PRED_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "category": len(Tfrdata.CATEGORY_NAMES)}
    ANCHORS_RATIO = None  # assigned by update_anchors()


class Architecture:
    BACKBONE = "Darknet53"
    HEAD = "FPN"
    BACKBONE_CONV_ARGS = {"activation": "leaky_relu", "scope": "back"}
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}


class Train:
    CKPT_NAME = "yolo2"
    MODE = ["eager", "graph"][1]
    BATCH_SIZE = 2
    TRAINING_PLAN = params.TrainingPlan.KITTI_SIMPLE


class NmsInfer:
    MAX_OUT = [10, 10, 10, 10]
    IOU_THRESH = [0.3, 0.3, 0.3, 0.3]
    SCORE_THRESH = [0.2, 0.2, 0.2, 0.2]


def change_dataset(dataset_name):
    Datasets.TARGET_DATASET = dataset_name
    update_anchors(dataset_name)


def update_anchors(dataset="kitti"):
    basic_anchor = params.Anchor.COCO_YOLOv3                                        # [[13, 10], ...]
    dataset_cfg = Datasets.get_dataset_config(dataset)
    input_resolution = np.array(dataset_cfg.INPUT_RESOLUTION, dtype=np.float32)     # (256, 832)
    anchor_resolution = np.array(params.Anchor.COCO_RESOLUTION, dtype=np.float32)   # (416, 416)
    scale = np.min(input_resolution / anchor_resolution)                            # 256/416=0.615
    ModelOutput.ANCHORS_RATIO = np.around(basic_anchor * scale, 1) / input_resolution   # [[13, 10], ...] * 0.615
    print("[update_anchors] anchors in ratio:\n", ModelOutput.ANCHORS_RATIO)


update_anchors()
