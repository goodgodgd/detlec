import os.path as op


class Config:
    class Paths:
        RAW_DATA = {"kitti": "/media/ian/Ian4T/dataset/kitti_detection"}
        RESULT_ROOT = "/home/ian/workspace/detlec/dataset"
        TFRECORD = op.join(RESULT_ROOT, "tfrecord")

    class Dataset:
        DATASETS_FOR_TFRECORD = {"kitti": ("train", "val")}
        INPUT_RESOLUTIONS = {"kitti": (302, 1000)}
        INVALID_CATEGORY = 100
        MAX_BBOX_PER_IMAGE = 20

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


