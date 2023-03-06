import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import neutr.utils.util_function as nuf
import pytch.utils.util_function as puf
from neutr.log.logger import Logger
from pytch.train.fmap_generator import SinglePositivePolicy


class TrainValBase:
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        self.model = model
        self.loss_object = loss_object
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.optimizer = optimizer
        self.fmap_generator = SinglePositivePolicy()
        self.is_train = True
        self.mode = 'training'
        self.total_count = 0
        self.correct_count = 0

    def run_epoch(self, dataset, epoch: int, visual_log: bool):
        logger = Logger(epoch, self.ckpt_path, visual_log, self.is_train, puf.convert_to_numpy)
        self.reset()
        for step, features in enumerate(dataset):
            start = timer()
            # features = self.fmap_generator(features)
            prediction, total_loss, loss_by_type = self.run_batch(features)
            # logger.log_batch_result(step, features, prediction, total_loss, loss_by_type)
            nuf.print_progress(f"{self.mode} {step}/{self.epoch_steps} steps, "
                               f"time={timer() - start:.3f}, "
                               f"loss={total_loss:.3f}, ")
            # if step > 20:
            #     break

        print("")
        # logger.finalize()
        self.evaluate()

    def run_batch(self, features):
        raise NotImplementedError()

    def reset(self):
        self.total_count = 0
        self.correct_count = 0
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

    def evaluate(self):
        correct_count = self.correct_count.detach().cpu().numpy()
        print(f"[evaluate result] correct={correct_count}, total={self.total_count}, accuracy={correct_count/self.total_count}")


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
        self.is_train = True
        self.mode = 'training'

    def run_batch(self, features):
        x_batch, y_batch = features
        self.optimizer.zero_grad()
        y_pred = self.model(x_batch)
        total_loss, loss_by_type = self.loss_object(y_batch, y_pred)
        total_loss.backward()
        self.optimizer.step()
        self.correct_count += torch.sum(torch.argmax(y_pred['linear2/softmax'], dim=1) == y_batch)
        self.total_count += y_batch.shape[0]
        return y_pred, total_loss, loss_by_type


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, ckpt_path, optimizer=None):
        super().__init__(model, loss_object, epoch_steps, ckpt_path, optimizer)
        self.is_train = False
        self.mode = 'evaluating'

    def run_batch(self, features):
        x_batch, y_batch = features
        y_pred = self.model(x_batch)
        total_loss, loss_by_type = self.loss_object(y_batch, y_pred)
        self.correct_count += torch.sum(torch.argmax(y_pred['linear2/softmax'], dim=1) == y_batch)
        self.total_count += y_batch.shape[0]
        return y_pred, total_loss, loss_by_type


class TorchClassifier:
    def __init__(self, model, batch_size=32, val_ratio=0.2):
        self.model = model
        self.loss_object = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.val_ratio = val_ratio

    def train(self, x, y, epochs):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y)
        trainlen = int(x.shape[0] * (1 - self.val_ratio))
        x_train, y_train = x[:trainlen], y[:trainlen]
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        x_val, y_val = x[trainlen:], y[trainlen:]

        with DurationTime("** training time"):
            for epoch in range(epochs):
                for x_batch, y_batch in trainloader:
                    self.train_batch(x_batch, y_batch)

                loss, accuracy = self.evaluate(x_val, y_val, verbose=False)
                print(f"[Training] epoch={epoch}, val_loss={loss:1.4f}, val_accuracy={accuracy:1.4f}")

    def train_batch(self, x_batch, y_batch):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        y_pred = self.model(x_batch)
        loss = self.loss_object(y_pred, y_batch)
        loss.backward()
        self.optimizer.step()

    def evaluate(self, x, y_true, verbose=True):
        if isinstance(x, np.ndarray):
            x, y_true = torch.from_numpy(x).float(), torch.from_numpy(y_true)

        y_pred = self.model(x)
        accuracy = torch.mean((torch.argmax(y_pred, dim=1) == y_true).float())
        loss = self.loss_object(y_pred, y_true)
        if verbose:
            np.set_printoptions(precision=4, suppress=True)
            print("  prediction shape:", y_pred.shape, y_true.shape)
            print("  first 5 predicts:\n", y_pred[:5].detach().numpy())
            print("  check probability:", torch.sum(y_pred[:5], dim=1))
            print(f"  loss={loss.detach().numpy():1.4f}, accuracy={accuracy.detach().numpy():1.4f}")
        return loss, accuracy


def torch_classifier():
    (x_train, y_train), (x_test, y_test) = load_dataset("cifar10", show_imgs=True)
    model = TorchClsfModel()
    clsf = TorchClassifier(model)
    clsf.train(x_train, y_train, 5)
    clsf.evaluate(x_test, y_test)


if __name__ == "__main__":
    torch_classifier()