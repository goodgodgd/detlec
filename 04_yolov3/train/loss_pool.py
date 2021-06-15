import tensorflow as tf
import numpy as np

import utils.util_function as uf


class LossBase:
    def __call__(self, grtr, pred, auxi):
        dummy_large = tf.reduce_mean(tf.square(pred["feature_l"]))
        dummy_medium = tf.reduce_mean(tf.square(pred["feature_m"]))
        dummy_small = tf.reduce_mean(tf.square(pred["feature_s"]))
        dummy_total = dummy_large + dummy_medium + dummy_small
        return dummy_total, {"dummy_large": dummy_large, "dummy_medium": dummy_medium, "dummy_small": dummy_small}


class CiouLoss(LossBase):
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1)
        object_mask = grtr["object"]
        ciou_loss = self.compute_ciou(grtr["yxhw"], pred["yxhw"])
        # sum over object-containing grid cells
        scalar_loss = tf.reduce_sum(object_mask[..., 0] * ciou_loss)
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
        grtr_hw_ratio = grtr_yxhw[..., 3] / (grtr_yxhw[..., 2] + 1.0e-5)
        pred_hw_ratio = pred_yxhw[..., 3] / (pred_yxhw[..., 2] + 1.0e-5)
        coeff = tf.convert_to_tensor(4.0 / (np.pi * np.pi), dtype=tf.float32)
        v = coeff * tf.pow((tf.atan(grtr_hw_ratio) - tf.atan(pred_hw_ratio)), 2)
        alpha = v / (1 - iou + v)
        penalty = d + alpha * v
        loss = 1 - iou + penalty
        return loss


class ObjectnessLoss(LossBase):
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["object"]
        pred_obj = pred["object"]
        obj_loss = tf.keras.losses.binary_crossentropy(grtr_obj, pred_obj)
        scalar_loss = tf.reduce_sum(obj_loss)
        return scalar_loss, obj_loss


class CategoryLoss(LossBase):
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        object_mask, valid_category = grtr["object"], auxi["valid_category"]
        num_cate = pred["category"].shape[-1]
        grtr_label = tf.cast(grtr["category"], dtype=tf.int32)
        grtr_onehot = tf.one_hot(grtr_label[..., 0], depth=num_cate, axis=-1)[..., tf.newaxis]
        # category_loss: (batch, HWA, K)
        category_loss = tf.losses.binary_crossentropy(grtr_onehot, pred["category"][..., tf.newaxis], label_smoothing=0.05)
        category_loss = category_loss * object_mask * valid_category
        scalar_loss = tf.reduce_sum(category_loss)
        return scalar_loss, category_loss
