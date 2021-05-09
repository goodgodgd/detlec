import tensorflow as tf

import train.loss_pool as loss
import utils.util_function as uf


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_weights = loss_weights
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = tf.convert_to_tensor(valid_category)
        self.valid_category = tf.reshape(self.valid_category, (1, 1, valid_category.shape[0]))
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

    def __call__(self, features, predictions):
        scales = [key for key in features if "feature_" in key]
        total_loss = 0
        for scale_name in scales:
            suffix = scale_name[-2:]
            grtr = uf.merge_dim_hwa(features[scale_name])
            grtr = uf.slice_features(grtr)
            pred = uf.merge_dim_hwa(predictions[scale_name])
            pred = uf.slice_features(pred)
            auxi = self.prepare_auxiliary_data(grtr, pred, scale_name)

            loss_values = dict()
            for loss_name, loss_object in self.loss_objects.items():
                loss_values[loss_name + suffix] = loss_object(grtr, pred, auxi)

            for loss_name, loss_value in loss_values:
                total_loss += loss_value * self.loss_weights[loss_name]
        return total_loss

    def prepare_auxiliary_data(self, grtr, pred, scale_name):
        auxiliary = dict()
        # As obj_count is used as a denominator, it must NOT be 0.
        auxiliary["object_count"] = tf.maximum(tf.reduce_sum(grtr["object"]), 1)
        auxiliary["valid_category"] = self.valid_category
        auxiliary["scale_weight"] = {"feature_l": 1, "feature_m": 1, "feature_s": 4}[scale_name]
        return auxiliary
