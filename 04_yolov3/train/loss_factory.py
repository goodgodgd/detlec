import tensorflow as tf

import train.loss_pool as loss
import utils.util_function as uf


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_weights = loss_weights
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = tf.convert_to_tensor(valid_category, dtype=tf.float32)
        self.loss_objects = self.create_loss_objects(loss_weights)

    def create_loss_objects(self, loss_weights):
        loss_objects = dict()
        if "ciou" in loss_weights:
            loss_objects["ciou"] = loss.CiouLoss()
        if "object" in loss_weights:
            loss_objects["object"] = loss.ObjectnessLoss()
        if "category" in loss_weights:
            loss_objects["category"] = loss.CategoryLoss()
        return loss_objects

    def __call__(self, grtr, pred):
        grtr_slices = uf.merge_and_slice_features(grtr, True)
        pred_slices = uf.merge_and_slice_features(pred, False)
        scales = [key for key in grtr_slices if "feature_" in key]
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_objects}

        for scale_name in scales:
            scale_suffix = scale_name[-2:]  # "_l", "_m", "_s"
            auxi = self.prepare_auxiliary_data(grtr_slices[scale_name], pred_slices[scale_name])

            for loss_name, loss_object in self.loss_objects.items():
                scalar_loss, loss_map = loss_object(grtr_slices[scale_name], pred_slices[scale_name], auxi)
                sc_loss_name = loss_name + scale_suffix
                # in order to assign different weights by scale,
                # set different weights for loss names with scale suffices in LossComb class
                weight = self.loss_weights[sc_loss_name] if sc_loss_name in self.loss_weights \
                                                         else self.loss_weights[loss_name]
                total_loss += scalar_loss * weight
                loss_by_type[loss_name] += scalar_loss
                loss_by_type[sc_loss_name] = loss_map
        return total_loss, loss_by_type

    def prepare_auxiliary_data(self, grtr, pred):
        auxiliary = dict()
        # As object_count is used as a denominator, it must NOT be 0.
        auxiliary["object_count"] = tf.maximum(tf.reduce_sum(grtr["object"]), 1)
        auxiliary["valid_category"] = self.valid_category
        return auxiliary
