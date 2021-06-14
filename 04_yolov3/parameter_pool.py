import numpy as np


class LossComb:
    STANDARD = {"ciou": 1., "object": 1., "category": 1.}
    INITIALIZE = {"ciou": 1., "object": 0.1, "category": 1.}
    SCALE_WEIGHT = {"ciou": 1., "object_l": 1., "object_m": 1., "object_s": 4., "category": 1.}


class Anchor:
    """
    anchor order MUST be compatible with Config.Model.Output.FEATURE_ORDER
    in the current setting, the smallest anchor comes first
    """
    COCO_YOLOv3 = np.array([[13, 10], [30, 16], [23, 33],
                            [61, 30], [45, 62], [119, 59],
                            [90, 116], [198, 156], [326, 373]], dtype=np.float32)
    COCO_RESOLUTION = (416, 416)


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 1, 0.00001, LossComb.INITIALIZE, True),
        ("kitti", 2, 0.00001, LossComb.STANDARD, True),
        ("kitti", 7, 0.0001, LossComb.STANDARD, True),
        ("kitti", 10, 0.00001, LossComb.STANDARD, True)
    ]

