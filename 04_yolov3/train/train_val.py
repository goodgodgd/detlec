import tensorflow as tf

import time
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
        self.model_log = ModelLog()

    def run_epoch(self, dataset):
        epoch_start = time.time()
        self.model_log.clear()
        for step, features in enumerate(dataset):
            start = time.time()
            total_loss, loss_by_type, preds = self.run_batch(features)
            self.model_log.append_batch_result(total_loss, loss_by_type, preds)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={time.time() - start:.2f}... ")
            if step > 20:
                break

        print("")
        self.model_log.append_epoch_result(time=time.time() - epoch_start)
        return self.model_log

    def run_batch(self, features):
        raise NotImplementedError()


class ModelEagerTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
    
    def run_batch(self, features):
        return self.train_step(features)

    def train_step(self, features):
        with tf.GradientTape() as tape:
            preds = self.model(features["image"])
            total_loss, loss_by_type = self.loss_object(features, preds)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return total_loss, loss_by_type, preds


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
        preds = self.model(features["image"])
        total_loss, loss_by_type = self.loss_object(features, preds)
        return total_loss, loss_by_type, preds


class ModelGraphValidater(ModelEagerValidater):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, epoch_steps)

    @tf.function
    def run_batch(self, features):
        self.validate_step(features)

