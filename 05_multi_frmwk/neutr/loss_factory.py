import config as cfg


class IntegratedLoss:
    def __init__(self, loss_weights, pool_module):
        self.loss_weights = loss_weights
        self.loss_objects = self.create_loss_objects(loss_weights, pool_module)
        self.num_scale = len(cfg.ModelOutput.FEATURE_SCALES)

    def create_loss_objects(self, loss_weights, pool_module):
        loss_objects = dict()
        if "ciou" in loss_weights:
            loss_objects["ciou"] = pool_module.CiouLoss()
        if "object" in loss_weights:
            loss_objects["object"] = pool_module.ObjectnessLoss()
        if "category" in loss_weights:
            loss_objects["category"] = pool_module.CategoryLoss(len(cfg.Tfrdata.CATEGORY_NAMES))
        return loss_objects

    def __call__(self, features, predictions):
        total_loss = 0
        loss_by_type = {}
        for loss_name, loss_object in self.loss_objects.items():
            loss_map_key = loss_name + "_map"
            loss_by_type[loss_name] = 0
            loss_by_type[loss_map_key] = []
            for scale in range(self.num_scale):
                scalar_loss, loss_map = loss_object(features["fmap"], predictions["fmap"], scale)
                weight = self.loss_weights[loss_name]
                weight = weight[scale] if isinstance(weight, list) else weight
                total_loss += scalar_loss * weight
                loss_by_type[loss_name] += scalar_loss * weight
                loss_by_type[loss_map_key].append(loss_map)
        return total_loss, loss_by_type
