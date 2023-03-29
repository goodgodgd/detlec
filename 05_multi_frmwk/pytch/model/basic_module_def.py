import numpy as np
import torch

from pytch.model.module_def_base import ModuleDefBase


class Input(ModuleDefBase):
    def __init__(self, name, chw_shape, output=False):
        super().__init__()
        out_resol = np.array(chw_shape[1:], dtype=np.int32)
        self.props = {'name': name, 'out_channels': chw_shape[0],
                      'out_resol': out_resol, 'output': output, 'alias': 'input'}

    def fill_and_append(self, building_modules):
        building_modules[self['name']] = self
        return building_modules


class Conv2d(ModuleDefBase):
    def __init__(self, name=None, in_name=None, out_channels=None, kernel_size=3, padding='same', stride=1, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name, 'in_channels': None, 'out_channels': out_channels,
                      'stride': stride, 'kernel_size': kernel_size, 'padding': padding,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'conv'}
        self.args = ['in_channels', 'out_channels', 'kernel_size', 'padding', 'stride']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        assert bef_module['out_channels'] is not None
        self['in_channels'] = bef_module['out_channels']
        self['in_resol'] = bef_module['out_resol']
        self['out_resol'] = self['in_resol'] // self['stride']
        if self['stride'] > 1:
            self['out_resol'] = (self['in_resol'] - 1) // self['stride']
            self['padding'] = 0
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        args = {arg: self.props[arg] for arg in self.args}
        return self.single_input, torch.nn.Conv2d(**args)


class MaxPool2d(ModuleDefBase):
    def __init__(self, name, in_name, kernel_size=2, stride=2, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name, 'in_channels': None, 'out_channels': None,
                      'stride': stride, 'kernel_size': kernel_size,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'pool'}
        self.args = ['kernel_size', 'stride']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        self['out_resol'] = self['in_resol'] // self['stride']
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        args = {arg: self.props[arg] for arg in self.args}
        return self.single_input, torch.nn.MaxPool2d(**args)


class Activation(ModuleDefBase):
    def __init__(self, name, function, in_name, output=False, **kwargs):
        super().__init__()
        self.props = {'name': name, 'function': function, 'in_name': in_name,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'acti'}
        self.props.update(kwargs)
        self.args = list(kwargs.keys())

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        args = {arg: self.props[arg] for arg in self.args}
        if self['function'] == 'relu':
            return self.single_input, torch.nn.ReLU(**args)
        elif self['function'] == 'leakyrelu':
            return self.single_input, torch.nn.LeakyReLU(**args)
        elif self['function'] == 'swish' or self['function'] == 'silu':
            return self.single_input, torch.nn.SiLU(**args)
        elif self['function'] == 'softmax':
            return self.single_input, torch.nn.Softmax(**args)


class BatchNormalization(ModuleDefBase):
    def __init__(self, name, in_name, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'bn'}
        self.args = ['in_channels']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        return self.single_input, torch.nn.BatchNorm2d(num_features=self['in_channels'])

class Arithmetic(ModuleDefBase):
    def __init__(self, name, function, in_name, output=False):
        super().__init__()
        self.props = {'name': name, 'function': function, 'in_name': in_name,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'arth'}

    def fill_and_append(self, building_modules):
        bef_module0 = building_modules[self['in_name'][0]]
        bef_module1 = building_modules[self['in_name'][1]]
        assert np.allclose(bef_module0['out_resol'], bef_module1['out_resol'])
        assert bef_module0['out_channels'] == bef_module1['out_channels']
        self.fill_default(bef_module0)
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        if self['function'] == 'add':
            return self.multi_input, lambda x: x[0] + x[1]
        elif self['function'] == 'subtract':
            return self.multi_input, lambda x: x[0] - x[1]
        elif self['function'] == 'multiply':
            return self.multi_input, lambda x: x[0] * x[1]
        elif self['function'] == 'divide':
            return self.multi_input, lambda x: x[0] / x[1]


class Flatten(ModuleDefBase):
    def __init__(self, name, in_name, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name,
                      'in_channels': None, 'out_features': None,
                      'in_resol': None, 'output': output, 'alias': 'flat'}

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self['in_channels'] = bef_module['out_channels']
        self['out_features'] = self['in_channels'] * bef_module['out_resol'][0] * bef_module['out_resol'][1]
        self['in_resol'] = bef_module['out_resol']
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        return self.single_input, lambda x: x.view(-1, self['out_features'])


class Linear(ModuleDefBase):
    def __init__(self, name, in_name, out_features=None, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name,
                      'in_features': None, 'out_features': out_features, 'output': output, 'alias': 'linear'}
        self.args = ['in_features', 'out_features']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        assert bef_module['out_features'] is not None
        self['in_features'] = bef_module['out_features']
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        args = {arg: self.props[arg] for arg in self.args}
        return self.single_input, torch.nn.Linear(**args)


class Upsample2d(ModuleDefBase):
    def __init__(self, name, in_name, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'upsamp'}

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        self['out_resol'] = self['in_resol'] * 2
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        return self.single_input, torch.nn.Upsample(scale_factor=2, mode='bilinear')


class Concat(ModuleDefBase):
    def __init__(self, name, in_name=None, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'concat'}

    def fill_and_append(self, building_modules):
        bef_module0 = building_modules[self['in_name'][0]]
        bef_module1 = building_modules[self['in_name'][1]]
        assert np.allclose(bef_module0['out_resol'], bef_module1['out_resol'])
        self.fill_default(bef_module0)
        self['in_channels'] = [bef_module0['out_channels'], bef_module1['out_channels']]
        self['out_channels'] = bef_module0['out_channels'] + bef_module1['out_channels']
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        return self.multi_input, lambda x: torch.cat(x, dim=1)


class Padding(ModuleDefBase):
    def __init__(self, name=None, in_name=None, padding=(0, 1, 0, 1), value=0, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name, 'padding': padding, 'value': value,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'pad'}
        self.args = ['padding', 'value']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        padding = self['padding']
        self['out_resol'] = self['in_resol'] + np.array((padding[2] + padding[3],
                                                         padding[0] + padding[1]), dtype=np.int32)
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        args = {arg: self.props[arg] for arg in self.args}
        return self.single_input, torch.nn.ConstantPad2d(**args)


class Reshape5D(ModuleDefBase):
    def __init__(self, name, in_name=None, anchors=None, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name, 'anchors': anchors,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output, 'alias': 'bn'}
        self.args = ['in_channels']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        return self.single_input, Reshape5DModule(self['anchors'])


class Reshape5DModule(torch.nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def forward(self, x):
        batch, channel, height, width = x.shape
        out_channel = channel // self.anchors
        x = torch.reshape(x, (batch, out_channel, self.anchors, height, width))
        return x