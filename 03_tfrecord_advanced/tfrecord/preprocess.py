import numpy as np
import cv2

import util_class as uc
import util_function as uf


class PreprocessBase:
    def __call__(self, example):
        raise NotImplementedError()


class ExamplePreprocess(PreprocessBase):
    def __init__(self, target_hw, crop_offset=None):
        # crop offset: dy1, dx1, dy2, dx2 (top, left, bottom, right)
        crop_offset = [0, 0, 0, 0] if crop_offset is None else crop_offset
        self.preprocess = [ExampleCropper(target_hw, crop_offset), 
                           ExampleResizer(target_hw),   # box in pixel scale
                           ExampleBoxScaler()]          # box in (0~1) scale
    
    def __call__(self, example):
        for process in self.preprocess:
            example = process(example)
        return example


class ExampleCropper(PreprocessBase):
    """
    crop image for aspect ratio to be the same with that of target
    adjust boxes to be consistent with the cropped image
    """
    def __init__(self, target_hw, crop_offset=None):
        # target image aspect ratio: width / height
        self.target_hw_ratio = target_hw[1] / target_hw[0]
        # crop offset: dy1, dx1, dy2, dx2 (top, left, bottom, right)
        self.crop_offset = [0, 0, 0, 0] if crop_offset is None else crop_offset

    def __call__(self, example: dict):
        source_hw = example["image"].shape[:2]
        crop_tlbr = self.find_crop_range(source_hw)
        image = self.crop_image(example["image"], crop_tlbr)
        cropped_hw = image.shape[:2]
        bboxes = self.crop_bboxes(example["bboxes"], crop_tlbr, cropped_hw)
        return {"image": image, "bboxes": bboxes}

    def find_crop_range(self, src_hw):                      # example:
        src_hw = np.array(src_hw, dtype=np.float32)         # [220, 540]
        offset = np.array(self.crop_offset, dtype=np.int32) # [10, 20, 10, 20]
        src_crop_hw = src_hw - (offset[:2] + offset[2:])    # [200, 500]
        src_hw_ratio = src_crop_hw[1] / src_crop_hw[0]      # 2.5
        dst_hw_ratio = self.target_hw_ratio                 # 2
        if dst_hw_ratio < src_hw_ratio:                     # crop x-axis, dst_hw=[200, 400]
            dst_hw = np.array([src_hw[0], src_hw[0] * dst_hw_ratio], dtype=np.int32)
        else:
            dst_hw = np.array([src_hw[1] / dst_hw_ratio, src_hw[1]], dtype=np.int32)
        # crop with fixed center, ([200, 500]-[200, 400])/2 = [0, 50]
        crop_yx = ((src_crop_hw - dst_hw) // 2).astype(np.int32)
        # crop top left bottom right, [10, 20, 10, 20] + [0, 50, 0, 50] = [10, 70, 10, 70]
        crop_tlbr = offset + np.concatenate([crop_yx, crop_yx], axis=0)
        return crop_tlbr

    def crop_image(self, image, crop_tlbr):
        if crop_tlbr[0] > 0:
            image = image[crop_tlbr[0]:]
        if crop_tlbr[2] > 0:
            image = image[:-crop_tlbr[2]]
        if crop_tlbr[1] > 0:
            image = image[:, crop_tlbr[1]:]
        if crop_tlbr[3] > 0:
            image = image[:, :-crop_tlbr[3]]
        return image

    def crop_bboxes(self, bboxes, crop_tlbr, cropped_hw):
        # move image origin
        bboxes[:, 0] = bboxes[:, 0] - crop_tlbr[0]
        bboxes[:, 1] = bboxes[:, 1] - crop_tlbr[1]
        # filter boxes with centers outside image
        inside = (bboxes[:, 0] >= 0) & (bboxes[:, 0] < cropped_hw[0]) & \
                 (bboxes[:, 1] >= 0) & (bboxes[:, 1] < cropped_hw[1])
        bboxes = bboxes[inside]
        if bboxes.size == 0:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        # clip into image range
        bboxes = uf.convert_box_format_yxhw_to_2pt(bboxes)
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], cropped_hw[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], cropped_hw[1])
        bboxes = uf.convert_box_format_2pt_to_yxhw(bboxes)
        return bboxes


class ExampleResizer(PreprocessBase):
    def __init__(self, target_hw):
        self.target_hw = np.array(target_hw, dtype=np.float32)
    
    def __call__(self, example):
        source_hw = np.array(example["image"].shape[:2], dtype=np.float32)
        resize_ratio = self.target_hw[0] / source_hw[0]
        assert np.isclose(self.target_hw[0] / source_hw[0], self.target_hw[1] / source_hw[1], atol=0.001)
        # resize image
        image = cv2.resize(example["image"], (self.target_hw[1], self.target_hw[0]))  # (256, 832)
        bboxes = example["bboxes"].astype(np.float32)
        # rescale yxhw
        bboxes[:, :4] *= resize_ratio
        return {"image": image, "bboxes": bboxes}


class ExampleBoxScaler(PreprocessBase):
    """
    scale bounding boxes into (0~1)
    """
    def __call__(self, example):
        height, width = example["image"].shape[:2]
        bboxes = example["bboxes"].astype(np.float32)
        bboxes[:, :4] /= np.array([height, width, height, width])
        return {"image": example["image"], "bboxes": bboxes}
