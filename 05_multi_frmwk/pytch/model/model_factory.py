import torch.nn
import torch.nn as nn
import torch.nn.functional as F

import pytch.model.model_definition as pmd
import config as cfg


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



class CnnModel(nn.Module):
    def __init__(self, architecture=cfg.Architecture):
        super().__init__()
        self.model_def = pmd.ModelAssembler(architecture).get_model_def()
        self.modules = self.build()
        self.input_names = [name for (name, module) in self.model_def.items() if module['type'] == 'input']
        self.output_names = [name for (name, module) in self.model_def.items() if 'output' in module and module['output'] is True]

    def build(self):
        modules = {}
        for name, module_def in self.model_def.items():
            if module_def['type'] == 'conv2d':
                args = {key: module_def[key] for key in ['in_channels', 'out_channels', 'kernel_size', 'padding']}
                modules[name] = (self.single_input, torch.nn.Conv2d(**args))
            elif module_def['type'] == 'maxpool':
                args = {key: module_def[key] for key in ['kernel_size', 'stride']}
                modules[name] = (self.single_input, torch.nn.MaxPool2d(**args))
            elif module_def['type'] == 'add':
                modules[name] = (self.multi_input, lambda x: x[0] + x[1])
            elif module_def['type'] == 'input':
                pass
            else:
                assert 0, f"No module type {module_def['type']} is defined"
        return modules

    def single_input(self, src_name, prior_outputs):
        return prior_outputs[src_name]

    def multi_input(self, src_names, prior_outputs):
        x =  [prior_outputs[src] for src in src_names]
        return x

    def forward(self, x):
        module_outputs = {self.input_names[0]: x}
        for name, (input_gen, module) in self.modules.items():
            module_outputs[name] = module(input_gen(self.model_def[name]['input'], module_outputs))
        model_output = {name: module_outputs[name] for name in self.output_names}
        return model_output


if __name__ == "__main__":
    cnn = CnnModel()
    x = torch.rand((3, 320, 320), dtype=torch.float32)
    y = cnn(x)
    print("cnn output", y.keys(), y['conv3'].size())