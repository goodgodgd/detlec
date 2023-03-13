import os.path as op
import numpy as np


class Paths:
    RESULT_ROOT = "F:/work/result"
    TFRECORD = op.join(RESULT_ROOT, "tfrecord")


class Tfrdata:
    DATASETS_FOR_TFRECORD = {"kitti": ("train", "val"),}
    MAX_BBOX_PER_IMAGE = 20
    CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"]
    SHARD_SIZE = 2000


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

    @classmethod
    def get_dataset_config(cls, dataset):
        return cls.DATASET_CONFIGS[dataset]


class ModelOutput:
    FEATURE_SCALES = [32, 16, 8]
    NUM_ANCHORS_PER_SCALE = 3
    GRTR_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1}
    PRED_FMAP_COMPOSITION = {"yxhw": 4, "object": 1, "category":  len(Tfrdata.CATEGORY_NAMES)}
    ANCHORS_RATIO = None  # assigned by update_anchors()
    ANCHORS_PIXEL = np.array([[13, 10], [30, 16], [23, 33],
                              [61, 30], [45, 62], [119, 59],
                              [90, 116], [198, 156], [326, 373]])


class Train:
    BATCH_SIZE = 2


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


def update_anchors(dataset=None):
    YOLOv3_anchor = np.array([[13, 10], [30, 16], [23, 33],
                             [61, 30], [45, 62], [119, 59],
                             [90, 116], [198, 156], [326, 373]])                    # [[13, 10], ...]
    dataset_cfg = Datasets.get_dataset_config(dataset)
    input_resolution = np.array(dataset_cfg.INPUT_RESOLUTION, dtype=np.float32)     # (256, 832)
    YOLOv3_resolution = np.array((416, 416), dtype=np.float32)                      # (416, 416)
    scale = np.min(input_resolution / YOLOv3_resolution)                            # 256/416=0.615
    ModelOutput.ANCHORS_RATIO = np.around(YOLOv3_anchor * scale, 1) / input_resolution   # [[13, 10], ...] * 0.615
    print("[update_anchors] anchors in ratio:\n", ModelOutput.ANCHORS_RATIO)


update_anchors("kitti")
