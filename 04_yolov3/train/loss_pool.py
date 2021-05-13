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
        :param grtr: dict of merged GT feature map slices, {key: (batch, HWA, dim), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, HWA, dim), ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1), object_count: scalar
        object_mask, object_count = grtr["object"], auxi["object_count"]
        ciou_loss = self.compute_ciou(pred["bbox"], grtr["bbox"])
        # average over object-containing grid cells
        scalar_loss = tf.reduce_sum(object_mask[..., 0] * ciou_loss) / object_count
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
        cbox_br = tf.minimum(grtr_tlbr[..., 2:], pred_tlbr[..., 2:])
        cbox_hw = tf.maximum(cbox_br - cbox_tl, 0.0)
        c = tf.reduce_sum(cbox_hw * cbox_hw, axis=-1)
        center_diff = grtr_yxhw[..., :2] - pred_yxhw[..., :2]
        u = tf.reduce_sum(center_diff * center_diff, axis=-1)
        d = tf.math.divide_no_nan(u, c)

        grtr_hw_ratio = tf.math.divide_no_nan(grtr_yxhw[..., 2], grtr_yxhw[..., 3])
        pred_hw_ratio = tf.math.divide_no_nan(pred_yxhw[..., 2], pred_yxhw[..., 3])
        coeff = tf.convert_to_tensor(4.0 / (np.pi * np.pi), dtype=tf.float32)
        v = coeff * tf.pow((tf.atan(grtr_hw_ratio) - tf.atan(pred_hw_ratio)), 2)
        alpha = v / (1 - iou + v)
        penalty = d + alpha * v
        loss = 1 - iou + penalty
        return loss


class ObjectnessLoss:
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, HWA, dim), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, HWA, dim), ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["object"]
        pred_obj = pred["object"]
        # object_mask: (batch, HWA, 1), object_count: scalar
        object_mask, object_count = grtr["object"][..., 0], auxi["object_count"]
        # objectness loss
        obj_loss = tf.keras.losses.binary_crossentropy(grtr_obj, pred_obj)
        # to equally weight positive and negative samples, average them separately
        obj_positive = tf.reduce_sum(obj_loss * object_mask) / object_count
        negative_count = tf.cast(tf.size(object_mask), tf.float32) - object_count
        obj_negative = tf.reduce_sum(obj_loss * (1. - object_mask)) / negative_count
        scalar_loss = obj_positive + obj_negative
        return scalar_loss, obj_loss


class CategoryLoss:
    def __call__(self, grtr, pred, auxi):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, HWA, dim), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, HWA, dim), ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        grtr_cate = grtr["category"][..., 0]            # (batch, HWA)
        pred_cate = pred["category"][..., tf.newaxis]   # (batch, HWA, K, 1)
        # object_mask: (batch, HWA, 1), object_count: scalar, valid_category: (K)
        object_mask, object_count, valid_category = grtr["object"], auxi["object_count"], auxi["valid_category"]
        grtr_cate_indices = tf.cast(grtr_cate, dtype=tf.int32)
        # (batch, HWA, K, 1)
        grtr_cate_onehot = tf.one_hot(grtr_cate_indices, depth=valid_category.shape[0], axis=-1)[..., tf.newaxis]

        # category loss: binary cross entropy per category for multi-label classification (batch, HWA, K)
        cate_loss = tf.keras.losses.binary_crossentropy(grtr_cate_onehot, pred_cate, label_smoothing=0.05)
        valid_category = tf.reshape(valid_category, (1, 1, valid_category.shape[0]))
        valid_category = tf.cast(valid_category, tf.float32)
        cate_loss = cate_loss * valid_category
        # cate_loss_reduced: (batch, HWA, 1)
        cate_loss_reduced = tf.reduce_sum(cate_loss, axis=-1, keepdims=True) / tf.reduce_sum(valid_category)
        scalar_loss = tf.reduce_sum(cate_loss_reduced * object_mask) / object_count
        return scalar_loss, cate_loss


