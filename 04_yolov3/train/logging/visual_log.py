import os
import os.path as op
import tensorflow as tf
import numpy as np
import cv2

import config as cfg
from train.logging.metric import split_true_false
import utils.util_function as uf


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        self.vlog_path = op.join(ckpt_path, "vlog", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Tfrdata.CATEGORY_NAMES)}

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        splits = split_true_false(grtr["bboxes"], pred["bboxes"])
        batch = splits["grtr_tp"]["yxhw"].shape[0]

        for i in range(batch):
            image_grtr = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], i, False, (0, 255, 0))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], i, False, (0, 0, 255))

            image_pred = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"], i, True, (0, 255, 0))
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"], i, True, (0, 0, 255))

            vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(100)
            filename = op.join(self.vlog_path, f"{step*batch + i:05d}.jpg")
            cv2.imwrite(filename, vlog_image)

    def draw_boxes(self, image, bboxes, frame_idx, is_pred, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param is_pred
        :param color: box color
        :return: box drawn image
        """
        height, width = image.shape[:2]
        box_yxhw = bboxes["yxhw"][frame_idx].numpy()        # (N, 4)
        category = bboxes["category"][frame_idx].numpy()    # (N, 1)
        objectness, ctgr_prob, score = None, None, None
        valid_mask = box_yxhw[:, 2] > 0                     # (N,) h>0

        box_yxhw = box_yxhw[valid_mask, :] * np.array([[height, width, height, width]], dtype=np.float32)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(box_yxhw)     # (N', 4)
        category = category[valid_mask, 0]                          # (N',)

        if is_pred:
            objectness = bboxes["object"][frame_idx].numpy()
            ctgr_prob = bboxes["ctgr_prob"][frame_idx].numpy()
            score = bboxes["score"][frame_idx].numpy()
            objectness = objectness[valid_mask, 0]
            ctgr_prob = ctgr_prob[valid_mask, 0]
            score = score[valid_mask, 0]

        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            annotation = f"{self.categories[category[i]]}"
            if is_pred:
                annotation += f",{objectness[i]:1.2f},{ctgr_prob[i]:1.2f},{score[i]:1.2f}"
            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image


