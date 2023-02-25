import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


""" 
Common utils
"""
class DurationTime:
    def __init__(self, context):
        self.start = 0
        self.context = context

    def __enter__(self):        # entering 'with' context
        self.start = timer()
        return self             # pass object by 'as'

    def __exit__(self, type, value, trace_back):    # exiting 'with' context
        print(f"{self.context}: {timer() - self.start:1.2f}")


def load_dataset(dataname="cifar10", show_imgs=False):
    if dataname == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError(f"Invalid dataset name: {dataname}")

    # pytorch uses image shape like (batch, channel, height, width)
    x_train = np.transpose(np.array(trainset.data), (0, 3, 1, 2))
    y_train = np.array(trainset.targets)
    x_test = np.transpose(np.array(testset.data), (0, 3, 1, 2))
    y_test = np.array(testset.targets)
    x_train, x_test = x_train / 255.0 * 2. - 1., x_test / 255.0 * 2. - 1.
    print(f"Load {dataname} dataset:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    if show_imgs:
        show_samples(trainset.data, trainset.targets, class_names)
    return (x_train, y_train), (x_test, y_test)


def show_samples(images, labels, class_names, grid=(3,4)):
    plt.figure(figsize=grid)
    num_samples = grid[0] * grid[1]
    for i in range(num_samples):
        plt.subplot(grid[0], grid[1], i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()


"""
Classifier
"""
class TorchClsfModel(nn.Module):
    def __init__(self):
        super(TorchClsfModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class TorchClsfModelUsingSequential(nn.Module):
    def __init__(self):
        super(TorchClsfModelUsingSequential, self).__init__()
        self.conv_relu_pool1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv_relu_pool2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_relu_pool1(x)
        x = self.conv_relu_pool2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


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
