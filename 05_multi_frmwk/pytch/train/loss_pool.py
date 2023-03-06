import torch
import torch.nn.functional as f
import numpy as np

import pytch.utils.util_function as puf


class LossBase:
    def __call__(self, grtr, pred, scale):
        dummy = [torch.mean(torch.square(fmap)) for fmap in pred["fmap"]]
        total = torch.sum(dummy)
        return total, dummy


class CiouLoss(LossBase):
    def __call__(self, grtr, pred, scale):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, channel, AHW), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, channel, AHW), ...}
        :param scale: scale index
        :return: complete-iou loss (batch, AHW)
        """
        ciou_loss = self.compute_ciou(pred["yxhw"][scale], grtr["yxhw"][scale])  # (batch, AHW)
        object_mask = torch.squeeze(grtr["object"][scale])      # (batch, AHW)
        scalar_loss = torch.sum(ciou_loss * object_mask)
        return scalar_loss, ciou_loss

    def compute_ciou(self, grtr_yxhw, pred_yxhw):
        """
        :param grtr_yxhw: (batch, 4, AHW)
        :param pred_yxhw: (batch, 4, AHW)
        :return: ciou loss (batch, AHW)
        """
        grtr_tlbr = puf.convert_box_format_yxhw_to_tlbr(grtr_yxhw)
        pred_tlbr = puf.convert_box_format_yxhw_to_tlbr(pred_yxhw)
        # iou: (batch, AHW)
        iou = puf.compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr, pred_tlbr)
        cbox_tl = torch.minimum(grtr_tlbr[:, :2], pred_tlbr[:, :2])
        cbox_br = torch.maximum(grtr_tlbr[:, 2:], pred_tlbr[:, 2:])
        cbox_hw = cbox_br - cbox_tl
        c = torch.sum(cbox_hw * cbox_hw, dim=-1)
        center_diff = grtr_yxhw[:, :2] - pred_yxhw[:, :2]
        u = torch.sum(center_diff * center_diff, dim=-1)
        # NOTE: divide_no_nan results in nan gradient
        # d = tf.math.divide_no_nan(u, c)
        d = u / (c + 1.0e-5)

        # grtr_hw_ratio = tf.math.divide_no_nan(grtr_yxhw[:, 2], grtr_yxhw[:, 3])
        # pred_hw_ratio = tf.math.divide_no_nan(pred_yxhw[:, 2], pred_yxhw[:, 3])
        grtr_hw_ratio = grtr_yxhw[:, 2] / (grtr_yxhw[:, 3] + 1.0e-5)
        pred_hw_ratio = pred_yxhw[:, 2] / (pred_yxhw[:, 3] + 1.0e-5)
        coeff = torch.tensor(4.0 / (np.pi * np.pi))
        v = coeff * torch.pow((torch.atan(grtr_hw_ratio) - torch.atan(pred_hw_ratio)), 2)
        alpha = v / (1 - iou + v)
        penalty = d + alpha * v
        loss = 1 - iou + penalty
        return loss


class ObjectnessLoss(LossBase):
    def __call__(self, grtr, pred, scale):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, channel, AHW), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, channel, AHW), ...}
        :param scale: scale index
        :return: objectness loss (batch, AHW)
        """
        grtr_obj = torch.squeeze(grtr["object"][scale])     # (batch, AHW)
        pred_obj = torch.squeeze(pred["object"][scale])
        # object_mask: (batch, 1, AHW), object_count: scalar
        object_mask = torch.squeeze(grtr["object"][scale])      # (batch, AHW)
        # objectness loss
        obj_loss = f.binary_cross_entropy(pred_obj, grtr_obj)   # (batch, AHW)
        # apply different weights on positive and negative samples
        obj_positive = torch.sum(obj_loss * object_mask)
        obj_negative = torch.sum(obj_loss * (1. - object_mask)) * 4.
        scalar_loss = obj_positive + obj_negative
        return scalar_loss, obj_loss


class CategoryLoss(LossBase):
    def __init__(self, num_ctgr):
        super().__init__()
        self.num_ctgr = num_ctgr

    def __call__(self, grtr, pred, scale):
        """
        :param grtr: dict of merged GT feature map slices, {key: (batch, channel, AHW), ...}
        :param pred: dict of merged pred. feature map slices, {key: (batch, channel, AHW), ...}
        :param scale: scale index
        :return: category loss (batch, AHW)
        """
        grtr_cate = torch.squeeze(grtr["category"][scale])  # (batch, AHW)
        pred_cate = pred["category"][scale]                 # (batch, K, AHW)
        object_mask = torch.squeeze(grtr["object"][scale])  # (batch, AHW)
        grtr_ctgr_indices = grtr_cate.type(torch.int)
        grtr_ctgr_onehot = f.one_hot(grtr_ctgr_indices, num_classes=self.num_ctgr)      # (batch, AHW, K)
        grtr_ctgr_onehot = grtr_ctgr_onehot.transpose(1, 2) # (batch, K, AHW)
        # category loss: binary cross entropy per category for multi-label classification (batch, K, AHW)
        grtr_ctgr_onehot = grtr_ctgr_onehot * (1 - 0.05) + (1 - grtr_ctgr_onehot) * 0.05
        ctgr_loss = f.binary_cross_entropy(pred_cate, grtr_ctgr_onehot)     # (batch, K, AHW)
        # ctgr_loss_reduced: (batch, AHW)
        ctgr_loss = torch.sum(ctgr_loss, dim=1)
        scalar_loss = torch.sum(ctgr_loss * object_mask)
        return scalar_loss, ctgr_loss


class CrossEntropyLoss(LossBase):
    def __init__(self, num_ctgr):
        super().__init__()
        self.num_ctgr = num_ctgr

    def __call__(self, grtr, pred, scale):
        grtr_cate = torch.squeeze(grtr["category"][scale])  # (batch, AHW)
        pred_cate = pred["category"][scale]                 # (batch, K, AHW)
        object_mask = torch.squeeze(grtr["object"][scale])  # (batch, AHW)
        celoss = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
        ctgr_loss = celoss(pred_cate, grtr_cate)            # (batch, AHW)
        scalar_loss = torch.sum(ctgr_loss * object_mask)
        return scalar_loss, ctgr_loss


class ClassifierLoss(LossBase):
    def __init__(self, num_ctgr):
        super().__init__()
        self.num_ctgr = num_ctgr

    def __call__(self, grtr, pred, scale):
        celoss = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.05)
        ctgr_loss = celoss(pred['linear2/softmax'], grtr)  # (batch, AHW)
        scalar_loss = torch.sum(ctgr_loss)
        return scalar_loss, ctgr_loss

    
    