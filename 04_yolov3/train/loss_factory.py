import train.loss as loss
from config import Config as cfg


class LossFactory:
    def __init__(self, loss_weights, dataset_name):
        self.loss_weight = loss_weights
        self.dataset_name = dataset_name

    def get_loss(self):
        return loss.LossBase()
