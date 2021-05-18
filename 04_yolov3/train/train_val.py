import tensorflow as tf
import numpy as np
from timeit import default_timer as timer

import utils.util_function as uf
from train.logger import ModelLog


def trainer_factory(mode, model, loss_object, optimizer, steps):
    if mode == "eager":
        return ModelEagerTrainer(model, loss_object, optimizer, steps)
    elif mode == "graph":
        return ModelGraphTrainer(model, loss_object, optimizer, steps)


def validater_factory(mode, model, loss_object, steps):
    if mode == "eager":
        return ModelEagerValidater(model, loss_object, steps)
    elif mode == "graph":
        return ModelGraphValidater(model, loss_object, steps)


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps

    def run_epoch(self, dataset):
        model_log = ModelLog()
        for step, features in enumerate(dataset):
            start = timer()
            prediction, total_loss, loss_by_type = self.run_batch(features)
            model_log.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={timer() - start:.3f}, "
                              f"feat_dl_max={np.around(np.max(np.abs(prediction['feature_l'].numpy())), 5)}, "
                              f"feat_rl_max={np.around(np.max(np.abs(prediction['feature_l_raw'].numpy())), 5)}, "
                              f"back_rl_max={np.around(np.max(np.abs(prediction['backbone_l_raw'].numpy())), 5)}, "
                              )
            if step % 200 == 10:
                print("")
            # if step > 20:
            #     break

        print("")
        model_log.finish()
        return model_log

    def run_batch(self, features):
        raise NotImplementedError()


class ModelEagerTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
    
    def run_batch(self, features):
        return self.train_step(features)

    def train_step(self, features):
        with tf.GradientTape() as tape:
            prediction = self.model(features["image"])
            total_loss, loss_by_type = self.loss_object(features, prediction)

        self.check_nan(loss_by_type, features, prediction)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type

    def check_nan(self, losses, grtr, pred):
        for name, loss in losses.items():
            loss = loss.numpy()
            if (loss.size > 1) and np.isnan(loss).any():
                print(f"[train] nan loss:", name, np.quantile(loss, np.linspace(0, 1, 11)))
            if loss.size == 1 and np.isnan(loss):
                print(f"[train] nan loss:", name, loss)
            if (loss.size > 1) and np.isinf(loss).any():
                print(f"[train] inf loss:", name, np.quantile(loss, np.linspace(0, 1, 11)))
            if loss.size == 1 and np.isinf(loss).any():
                print(f"[train] inf loss:", name, loss)
        for name, tensor in pred.items():
            if np.isnan(tensor).any():
                print(f"[train] nan pred:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
            if np.isinf(tensor).any():
                print(f"[train] inf pred:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
        for name, tensor in grtr.items():
            if np.isnan(tensor).any():
                print(f"[train] nan grtr:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
            if np.isinf(tensor).any():
                print(f"[train] inf grtr:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))


class ModelGraphTrainer(ModelEagerTrainer):
    def __init__(self, model, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
    
    @tf.function
    def run_batch(self, features):
        return self.train_step(features)


class ModelEagerValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)

    def run_batch(self, features):
        return self.validate_step(features)

    def validate_step(self, features):
        prediction = self.model(features["image"])
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type


class ModelGraphValidater(ModelEagerValidater):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, epoch_steps)

    @tf.function
    def run_batch(self, features):
        return self.validate_step(features)

