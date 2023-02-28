import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
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



class ModelFactory(nn.Module):
    def __init__(self, model_def):
        super().__init__()
        self.model_def = self.fill_modules_def(model_def)
        self.modules = self.build()

    def forward(self, x):
        mod_out = {}
        for name, (module, input_gen) in self.modules.items():
            mod_out[name] = module(input_gen(mod_out, name))        # input_gen uses self.model_Def


if __name__ == "__main__":
    pass