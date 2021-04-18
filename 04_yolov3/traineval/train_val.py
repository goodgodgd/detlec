import tensorflow as tf

import time
import os.path as op
import utils.util_function as uf


def trainer_factory(mode, model, loss, optimizer, steps):
    if mode == "eager":
        return ModelTrainerEager(model, loss, optimizer, steps)
    elif mode == "graph":
        return ModelTrainerGraph(model, loss, optimizer, steps)


def validater_factory(mode, model, loss, steps):
    if mode == "eager":
        return ModelValidaterEager(model, loss, steps)
    elif mode == "graph":
        return ModelValidaterGraph(model, loss, steps)


class TrainValBase:
    def __init__(self, model, loss_object, optimizer=None, epoch_steps=0):
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
            outputs = self.run_batch(features)
            self.model_log.append_batch_result(outputs)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, "
                              f"time={time.time() - start:.2f}... ")

        self.model_log.append_epoch_result(time=time.time() - epoch_start)
        return self.model_log

    def run_batch(self, features):
        pass


class ModelTrainerEager(TrainValBase):
    def __init__(self, model, loss_object, optimizer=None, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
    
    def run_batch(self, features):
        pass


class ModelTrainerGraph(TrainValBase):
    def __init__(self, model, loss_object, optimizer=None, epoch_steps=0):
        super().__init__(model, loss_object, optimizer, epoch_steps)
    
    @tf.function
    def run_batch(self, features):
        pass


class ModelValidaterEager(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)

    def run_batch(self, features):
        pass


class ModelValidaterGraph(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps=0):
        super().__init__(model, loss_object, None, epoch_steps)

    @tf.function
    def run_batch(self, features):
        pass


class ModelLog:
    def __init__(self):
        self.frame = dict()
        self.batch = dict()
        self.epoch = dict()

    def append_batch_result(self, outputs):
        pass

    def append_epoch_result(self, **kwargs):
        pass

    def clear(self):
        pass


