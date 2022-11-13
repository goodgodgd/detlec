import tensorflow as tf
import numpy as np

import utils.util_function as uf


class LossBase:
    def __call__(self, grtr, pred, auxi):
        dummy = [tf.reduce_mean(tf.square(fmap)) for fmap in pred["fmap"]]
        total = tf.reduce_sum(dummy)
        return total, dummy


class CiouLoss(LossBase):
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, HWA, channel), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, HWA, channel), ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1), object_count: scalar
        ciou_loss = self.compute_ciou(pred["yxhw"], grtr["yxhw"])
        # average over object-containing grid cells
        scalar_loss = tf.reduce_sum(grtr["object"][..., 0] * ciou_loss)
        return scalar_loss, ciou_loss

    def compute_ciou(self, grtr_yxhw, pred_yxhw):
        """
        :param grtr_yxhw: (batch, HWA, 4)
        :param pred_yxhw: (batch, HWA, 4)
        :return: ciou loss (batch, HWA)
        """
        grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr_yxhw)
        pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred_yxhw)
        # iou: (batch, HWA)
        iou = uf.compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr, pred_tlbr)
        cbox_tl = tf.minimum(grtr_tlbr[..., :2], pred_tlbr[..., :2])
        cbox_br = tf.maximum(grtr_tlbr[..., 2:], pred_tlbr[..., 2:])
        cbox_hw = cbox_br - cbox_tl
        c = tf.reduce_sum(cbox_hw * cbox_hw, axis=-1)
        center_diff = grtr_yxhw[..., :2] - pred_yxhw[..., :2]
        u = tf.reduce_sum(center_diff * center_diff, axis=-1)
        # NOTE: divide_no_nan results in nan gradient
        # d = tf.math.divide_no_nan(u, c)
        d = u / (c + 1.0e-5)

        # grtr_hw_ratio = tf.math.divide_no_nan(grtr_yxhw[..., 2], grtr_yxhw[..., 3])
        # pred_hw_ratio = tf.math.divide_no_nan(pred_yxhw[..., 2], pred_yxhw[..., 3])
        grtr_hw_ratio = grtr_yxhw[..., 2] / (grtr_yxhw[..., 3] + 1.0e-5)
        pred_hw_ratio = pred_yxhw[..., 2] / (pred_yxhw[..., 3] + 1.0e-5)
        coeff = tf.convert_to_tensor(4.0 / (np.pi * np.pi), dtype=tf.float32)
        v = coeff * tf.pow((tf.atan(grtr_hw_ratio) - tf.atan(pred_hw_ratio)), 2)
        alpha = v / (1 - iou + v)
        penalty = d + alpha * v
        loss = 1 - iou + penalty
        return loss


class ObjectnessLoss(LossBase):
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, HWA, channel), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, HWA, channel), ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["object"]
        pred_obj = pred["object"]
        # object_mask: (batch, HWA, 1), object_count: scalar
        object_mask = grtr["object"][..., 0]
        # objectness loss
        obj_loss = tf.keras.losses.binary_crossentropy(grtr_obj, pred_obj)
        # apply different weights on positive and negative samples
        obj_positive = tf.reduce_sum(obj_loss * object_mask)
        obj_negative = tf.reduce_sum(obj_loss * (1. - object_mask)) * 4.
        scalar_loss = obj_positive + obj_negative
        return scalar_loss, obj_loss


class CategoryLoss(LossBase):
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, HWA, channel), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, HWA, channel), ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        grtr_cate = grtr["category"][..., 0]            # (batch, HWA)
        pred_cate = pred["category"][..., tf.newaxis]   # (batch, HWA, K, 1)
        # object_mask: (batch, HWA, 1), object_count: scalar, valid_category: (K)
        object_mask, valid_category = grtr["object"], auxi["valid_category"]
        grtr_ctgr_indices = tf.cast(grtr_cate, dtype=tf.int32)
        # (batch, HWA, K, 1)
        grtr_ctgr_onehot = tf.one_hot(grtr_ctgr_indices, depth=valid_category.shape[0], axis=-1)[..., tf.newaxis]
        # category loss: binary cross entropy per category for multi-label classification (batch, HWA, K)
        ctgr_loss = tf.keras.losses.binary_crossentropy(grtr_ctgr_onehot, pred_cate, label_smoothing=0.05)
        valid_category = tf.reshape(valid_category, (1, 1, valid_category.shape[0]))
        valid_category = tf.cast(valid_category, tf.float32)
        ctgr_loss = ctgr_loss * valid_category
        # ctgr_loss_reduced: (batch, HWA, 1)
        ctgr_loss_reduced = tf.reduce_sum(ctgr_loss, axis=-1, keepdims=True)
        scalar_loss = tf.reduce_sum(ctgr_loss_reduced * object_mask)
        return scalar_loss, ctgr_loss


