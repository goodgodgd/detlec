import os
import os.path as op
import numpy as np
import cv2

import config as cfg
from neutr.log.metric import split_tp_fp_fn
import neutr.utils.util_function as uf


class VisualLog:
    def __init__(self, epoch, ckpt_path):
        self.vlog_path = op.join(ckpt_path, "vlog", f"ep{epoch:02d}")
        self.heatmap_path = op.join(ckpt_path, "heatmap", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        if not op.isdir(self.heatmap_path):
            os.makedirs(self.heatmap_path)
        self.categories = cfg.Tfrdata.CATEGORY_NAMES

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr:
        :param pred:
        """
        splits = split_tp_fp_fn(grtr["inst"], pred["inst"], cfg.NmsInfer.IOU_THRESH)
        batch = splits["grtr_tp"]["yxhw"].shape[0]

        for bi in range(batch):
            frame_id = step * batch + bi
            image = (grtr["image"][bi] * 255).astype(np.uint8)
            bbox_image = self.draw_detection_result(image, splits, bi)
            heat_image = self.draw_heatmap(grtr, pred, image.shape[:2], bi)
            filename = op.join(self.vlog_path, f"{frame_id:05d}.jpg")
            cv2.imwrite(filename, bbox_image)
            filename = op.join(self.heatmap_path, f"{frame_id:05d}.jpg")
            cv2.imwrite(filename, heat_image)
            if step % 50 == 10:
                cv2.imshow("detection_result", bbox_image)
                cv2.waitKey(10)

    def draw_detection_result(self, image, splits, batch_idx):
        image_grtr = image.copy()
        image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], batch_idx, (0, 255, 0))
        image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], batch_idx, (0, 0, 255))
        image_pred = image.copy()
        image_pred = self.draw_boxes(image_pred, splits["pred_tp"], batch_idx, (0, 255, 0))
        image_pred = self.draw_boxes(image_pred, splits["pred_fp"], batch_idx, (0, 0, 255))
        bbox_image = np.concatenate([image_pred, image_grtr], axis=0)
        return bbox_image

    def draw_boxes(self, image, bboxes, batch_idx, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param batch_idx
        :param color: box color
        :return: box drawn image
        """
        height, width = image.shape[:2]
        box_yxhw = bboxes["yxhw"][batch_idx]  # (N, 4)
        category = bboxes["category"][batch_idx]  # (N, 1)
        valid_mask = box_yxhw[:, 2] > 0  # (N,) h>0

        box_yxhw = box_yxhw[valid_mask, :] * np.array([[height, width, height, width]], dtype=np.float32)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(box_yxhw)  # (N', 4)
        category = category[valid_mask, 0].astype(np.int32)  # (N',)

        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def draw_heatmap(self, grtr, pred, imshape, batch_idx):
        heatmaps = list()
        for scale, whole_feat in enumerate(pred["fmap"]["head_logit"]):
            feat_shape = whole_feat.shape[1:4]    # (height, width, anchor)
            grtr_map = grtr["fmap"]["object"][scale][batch_idx]
            pred_map = pred["fmap"]["object"][scale][batch_idx]
            grtr_scoremap = self.convert_img(grtr_map, feat_shape, imshape)
            pred_scoremap = self.convert_img(pred_map, feat_shape, imshape)
            objness_map = np.concatenate([pred_scoremap, grtr_scoremap], axis=0)
            heatmaps.append(objness_map)
        heatmaps = np.concatenate(heatmaps, axis=1)
        return heatmaps

    def convert_img(self, feature, feat_shape, imshape):
        feat_img = (feature.reshape(feat_shape) * 255)
        feat_img = np.clip(feat_img, 0, 255).astype(np.uint8)
        if feat_img.shape[-1] == 1:
            feat_img = cv2.cvtColor(feat_img, cv2.COLOR_GRAY2BGR)
        feat_img = cv2.resize(feat_img, (imshape[1]//2, imshape[0]//2), interpolation=cv2.INTER_NEAREST)
        feat_img[-1, :] = [255, 255, 255]
        feat_img[:, -1] = [255, 255, 255]
        return feat_img
