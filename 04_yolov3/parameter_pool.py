import numpy as np


class LossComb:
    STANDARD = {"ciou": 1., "objectness": 1., "category": 1.}


class AnchorRatio:
    """
    anchor order MUST be compatible with Config.Model.Output.FEATURE_ORDER
    in the current setting, the smallest anchor comes first
    """
    COCO = np.array([[13, 10], [30, 16], [23, 33],
                     [61, 30], [45, 62], [119, 59],
                     [90, 116], [198, 156], [326, 373]]) / 416.


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 10, 0.0001, LossComb.STANDARD, True),
        ("kitti", 10, 0.00001, LossComb.STANDARD, True)
    ]

