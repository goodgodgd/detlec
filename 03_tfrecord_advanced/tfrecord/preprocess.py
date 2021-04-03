import numpy as np

import util_class as uc
import util_function as uf


class ExampleCropper:
    """
    crop image and bboxes
    """
    def __init__(self, target_hw, crop_offset=None):
        self.target_hw = target_hw
        # crop offset: dy1, dx1, dy2, dx2 (top, left, bottom, right)
        self.crop_offset = [0, 0, 0, 0] if crop_offset is None else crop_offset

    def __call__(self, example: dict):
        source_hw = example["image"].shape[:2]
        crop_tlbr = self.find_crop_range(source_hw, self.target_hw, self.crop_offset)
        example["image"] = self.crop_image(example["image"], crop_tlbr)
        example["bboxes"] = self.crop_bboxes(example["bboxes"], crop_tlbr)

    def find_crop_range(self, src_hw, dst_hw, offset):      # example:
        src_hw = np.array(src_hw, dtype=np.int32)           # [220, 540]
        dst_hw = np.array(dst_hw, dtype=np.int32)           # [100, 200]
        offset = np.array(offset, dtype=np.int32)           # [10, 20, 10, 20]
        src_crop_hw = src_hw - (offset[:2] + offset[2:])    # [200, 500]
        hw_ratio = dst_hw / src_crop_hw                     # [0.5, 0.4]
        resize_ratio = np.max(hw_ratio)                     # 0.5
        dst_crop_hw = dst_hw / resize_ratio                 # [200, 400]
        dst_crop_hw = dst_crop_hw.astype(np.int32)          # [200, 400]
        # crop with fixed center, ([200, 500]-[200, 400])/2 = [0, 50]
        crop_yx = ((src_crop_hw - dst_crop_hw) // 2).astype(np.int32)
        # crop top left bottom right, [10, 20, 10, 20] + [0, 50, 0, 50] = [10, 70, 10, 70]
        crop_tlbr = offset + np.concatenate([crop_yx, crop_yx], axis=0)
        return crop_tlbr

    def crop_image(self, image, crop_tlbr):
        crop = crop_tlbr
        return image[crop[0]:-crop[2], crop[1]:-crop[3]]

    def crop_bboxes(self, bboxes, crop_tlbr):
        bboxes = bboxes.astype(np.int32)    # [y x h w category]
        # move image origin
        bboxes[:, 0] = bboxes[:, 0] - crop_tlbr[0]
        bboxes[:, 1] = bboxes[:, 1] - crop_tlbr[1]
        # filter boxes with centers outside image
        inside = (bboxes[:, 0] >= 0) & (bboxes[:, 0] < self.target_hw[0]) & \
                 (bboxes[:, 1] >= 0) & (bboxes[:, 1] < self.target_hw[1])
        bboxes = bboxes[inside]
        if bboxes.size == 0:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        # clip into image range
        bboxes = uf.convert_box_format_yxhw_to_2pt(bboxes)
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], self.target_hw[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], self.target_hw[1])
        bboxes = uf.convert_box_format_2pt_to_yxhw(bboxes)
        return bboxes


# class ExampleResizer:



class CropperAndResizer:
    def __init__(self, hw_shape):
        self.hw_shape = hw_shape

    def __call__(self, example):
        crop_yxhw, resize_ratio = self.prepare_cnr(example["image"])
        image = self.cnr_image(example["image"], crop_yxhw)
        bboxes = self.cnr_bboxes(example["bboxes"], crop_yxhw, resize_ratio)
        # return new example
        example = {"image": image, "bboxes": bboxes}
        return example

    def prepare_cnr(self, image):
        src_hw = np.array(image.shape[:2])          # [375, 1242]
        dst_hw = np.array(self.hw_shape)            # [256, 832]
        hw_ratio = dst_hw / src_hw                  # [0.68, 0.67]
        resize_ratio = np.max(hw_ratio)             # 0.68
        crop_src_hw = dst_hw / resize_ratio         # [375, 1218.75]
        crop_src_hw = crop_src_hw.astype(np.int32)  # [375, 1218]
        crop_src_yx = ((src_hw - crop_src_hw) // 2).astype(np.int32)    # [0, 12]
        crop_yxhw = np.concatenate([crop_src_yx, crop_src_hw], axis=0)  # [0, 12, 375, 1218]
        return crop_yxhw, resize_ratio

    def cnr_image(self, image, crop_yxhw):
        # crop image: image[0:375, 12:1230] (375, 1218)
        image = image[crop_yxhw[0]:crop_yxhw[0] + crop_yxhw[2], crop_yxhw[1]:crop_yxhw[1] + crop_yxhw[3]]
        image = cv2.resize(image, (self.hw_shape[1], self.hw_shape[0]))  # (256, 832)
        return image

    def cnr_bboxes(self, bboxes, crop_yxhw, resize_ratio):
        bboxes = bboxes.astype(np.float)    # [y1 x1 y2 x2 category]
        # apply crop and resize to boxes
        bboxes[:, 0] = (bboxes[:, 0] - crop_yxhw[0]) * resize_ratio
        bboxes[:, 1] = (bboxes[:, 1] - crop_yxhw[1]) * resize_ratio
        bboxes[:, 2] = (bboxes[:, 2] - crop_yxhw[0]) * resize_ratio
        bboxes[:, 3] = (bboxes[:, 3] - crop_yxhw[1]) * resize_ratio
        # filter boxes outside image
        centers_yx = np.stack([bboxes[:, 0] + bboxes[:, 2] / 2, bboxes[:, 1] + bboxes[:, 3] / 2], axis=1).astype(np.int32)
        inside = (centers_yx[:, 0] >= 0) & (centers_yx[:, 0] < self.hw_shape[0]) & \
                 (centers_yx[:, 1] >= 0) & (centers_yx[:, 1] < self.hw_shape[1])
        bboxes = bboxes[inside]
        if bboxes.size == 0:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        # clip into image range
        bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2], self.hw_shape[0])
        bboxes[:, 3] = np.minimum(bboxes[:, 3], self.hw_shape[1])
        # pixel to ratio (0~1)
        bboxes[:, 0] /= self.hw_shape[0]
        bboxes[:, 1] /= self.hw_shape[1]
        bboxes[:, 2] /= self.hw_shape[0]
        bboxes[:, 3] /= self.hw_shape[1]
        return bboxes.astype(np.float32)
