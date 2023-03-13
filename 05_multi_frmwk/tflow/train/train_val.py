import tensorflow as tf
from timeit import default_timer as timer

import neutr.utils.util_function as nuf
import tflow.utils.util_function as tuf
from neutr.log.logger import Logger
from tflow.train.fmap_generator import SinglePositivePolicy


def trainer_factory(mode, model, loss_object, steps, ckpt_path, optimizer):
    if mode == "eager":
        return ModelEagerTrainer(model, loss_object, steps, ckpt_path, optimizer)
    elif mode == "graph":
        return ModelGraphTrainer(model, loss_object, steps, ckpt_path, optimizer)


def validater_factory(mode, model, loss_object, steps, ckpt_path):
    if mode == "eager":
        return ModelEagerValidater(model, loss_object, steps, ckpt_path)
    elif mode == "graph":
        return ModelGraphValidater(model, loss_object, steps, ckpt_path)


class TrainValBase:
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        self.model = model
        self.loss_object = loss_object
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.optimizer = optimizer
        self.fmap_generator = SinglePositivePolicy()
        self.is_train = True

    def run_epoch(self, dataset, epoch: int, visual_log: bool):
        logger = Logger(epoch, self.ckpt_path, visual_log, self.is_train, tuf.convert_to_numpy)
        for step, features in enumerate(dataset):
            start = timer()
            prediction, total_loss, loss_by_type = self.run_batch(features)
            logger.log_batch_result(step, features, prediction, total_loss, loss_by_type)
            nuf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, ")
            # if step > 20:
            #     break

        print("")
        logger.finalize()
        return logger

    def run_batch(self, features):
        raise NotImplementedError()


class ModelEagerTrainer(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
        self.is_train = True
    
    def run_batch(self, features):
        features = self.fmap_generator(features)
        return self.train_step(features)

    def train_step(self, features):
        with tf.GradientTape() as tape:
            prediction = self.model(features["image"])
            total_loss, loss_by_type = self.loss_object(features, prediction)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type


class ModelGraphTrainer(ModelEagerTrainer):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
    
    @tf.function
    def train_step(self, features):
        with tf.GradientTape() as tape:
            prediction = self.model(features["image"])
            total_loss, loss_by_type = self.loss_object(features, prediction)

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return prediction, total_loss, loss_by_type


class ModelEagerValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer=None):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
        self.is_train = False

    def run_batch(self, features):
        features = self.fmap_generator(features)
        return self.validate_step(features)

    def validate_step(self, features):
        prediction = self.model(features["image"])
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type


class ModelGraphValidater(ModelEagerValidater):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer=None):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)

    @tf.function
    def validate_step(self, features):
        prediction = self.model(features["image"])
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type

