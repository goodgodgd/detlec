import numpy as np


class LossComb:
    STANDARD = {"ciou": 1., "object": 1., "category": 1.}
    SCALE_WEIGHT = {"ciou": 10., "object": [4., 2., 1.], "category": 10.}
    SCALE_WEIGHT_v1 = {"ciou": 5., "object": [4., 2., 1.], "category": 5.}
    CLASSIFICATION = {"classifier": 1.}


class Anchor:
    """
    anchor order MUST be compatible with Config.ModelOutput.FEATURE_SCALE
    in the current setting, the smallest anchor comes first
    """
    COCO_YOLOv3 = np.array([[13, 10], [30, 16], [23, 33],
                            [61, 30], [45, 62], [119, 59],
                            [90, 116], [198, 156], [326, 373]], dtype=np.float32)
    COCO_RESOLUTION = (416, 416)


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 10, 0.0001, LossComb.SCALE_WEIGHT, True),
        ("kitti", 10, 0.00001, LossComb.SCALE_WEIGHT, True),
        ("kitti", 10, 0.00001, LossComb.SCALE_WEIGHT_v1, True)
    ]
    CIFAR10 = [
        ("cifar10", 10, 0.001, LossComb.CLASSIFICATION, True),
        ("cifar10", 10, 0.0001, LossComb.CLASSIFICATION, True),
    ]
