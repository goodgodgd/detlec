import torch
from timeit import default_timer as timer

import neutr.utils.util_function as nuf
import pytch.utils.util_function as puf
# from neutr.log.logger import Logger
from pytch.train.fmap_generator import SinglePositivePolicy
from pytch.model.model_util import DetectorPostProcess


class TrainValBase:
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        self.model = model
        self.loss_object = loss_object
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.optimizer = optimizer
        self.fmap_generator = SinglePositivePolicy()
        self.post_process = DetectorPostProcess()
        self.is_train = True
        self.mode = 'training'

    def run_epoch(self, dataset, epoch: int, visual_log: bool):
        # logger = Logger(epoch, self.ckpt_path, visual_log, self.is_train, puf.convert_to_numpy)
        self.reset()
        for step, features in enumerate(dataset):
            start = timer()
            features = self.fmap_generator(features)
            prediction, total_loss, loss_by_type = self.run_batch(features)
            pred_np = puf.convert_to_numpy(prediction)
            # logger.log_batch_result(step, features, prediction, total_loss, loss_by_type)
            nuf.print_progress(f"{self.mode} {step}/{self.epoch_steps} steps, "
                               f"time={timer() - start:.3f}, "
                               f"loss={total_loss:.3f}, ")
            # if step > 20:
            #     break

        print("")
        # logger.finalize()

    def run_batch(self, features):
        raise NotImplementedError()

    def reset(self):
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
        self.is_train = True
        self.mode = 'training'

    def run_batch(self, features):
        self.optimizer.zero_grad()
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        total_loss.backward()
        self.optimizer.step()
        return prediction, total_loss, loss_by_type


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer=None):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
        self.is_train = False
        self.mode = 'evaluating'

    def run_batch(self, features):
        prediction = self.model(features)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type
