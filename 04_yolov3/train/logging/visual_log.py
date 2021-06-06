import os.path as op
import tensorflow as tf
import numpy as np
import cv2

from train.logging.metric import split_true_false


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        self.ckpt_path = ckpt_path
        self.epoch = epoch

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': (B,N,6), 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'nms': (B,M,8), 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        """
        splits = split_true_false(grtr["bboxes"], pred["bboxes"])
        batch = splits["grtr_tp"].shape[0]

        for i in range(batch):
            image_grtr = grtr["image"][i].numpy()
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"][i].numpy(), True, (0, 255, 0))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"][i].numpy(), True, (0, 0, 255))
            image_pred = grtr["image"][i].numpy()
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"][i].numpy(), False, (0, 255, 0))
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"][i].numpy(), False, (0, 0, 255))
            vlog_image = np.concatenate([image_grtr, image_pred], axis=0)
            filename = op.join(self.ckpt_path, "vlog", f"ep{self.epoch:02d}", f"{step*batch + i:05d}.jpg")
            # cv2.imwrite(filename, vlog_image)

    def draw_boxes(self, image, bboxes, is_gt, color):
        """
        :param image: numpy image (H, W, 3)
        :param bboxes: bounding boxes (N, dim)
                        if is_gt True, dim=5: yxhw, category,
                        if is_gt False, dim=8: yxhw, category, objectness, category_prob, score
        :param is_gt:
        :param color: box color
        :return: box drawn image
        """


        return image




