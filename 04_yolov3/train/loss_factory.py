import tensorflow as tf

import train.loss_pool as loss
import config as cfg


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_weights = loss_weights
        # valid_category: binary mask of categories, (K)
        self.valid_category = tf.convert_to_tensor(valid_category, dtype=tf.int32)
        self.loss_objects = self.create_loss_objects(loss_weights)
        self.num_scale = len(cfg.ModelOutput.FEATURE_SCALES)

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
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_weights}
        loss_by_type.update({loss_name + "_map": [] for loss_name in self.loss_weights})
        for scale in range(self.num_scale):
            auxi = self.prepare_auxiliary_data(features, predictions)
            for loss_name, loss_object in self.loss_objects.items():
                loss_map_key = loss_name + "_map"
                scalar_loss, loss_map = loss_object(features["fmap"], predictions["fmap"], auxi)
                weight = self.loss_weights[loss_name]
                weight = weight[scale] if isinstance(weight, list) else weight
                total_loss += scalar_loss * weight
                loss_by_type[loss_name] += scalar_loss * weight
                loss_by_type[loss_map_key].append(loss_map)
        return total_loss, loss_by_type

    def prepare_auxiliary_data(self, grtr, pred):
        auxiliary = dict()
        auxiliary["valid_category"] = self.valid_category
        return auxiliary
