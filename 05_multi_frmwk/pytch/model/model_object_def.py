import numpy as np
from pprint import PrettyPrinter
import torch

import config as cfg


class ModelTemplate(torch.nn.Module):
    def __init__(self, architecture, input_shape=(3, 320, 320)):
        super().__init__()
        model_def = ModelDefFactory(architecture, input_shape).get_model_def()
        self.input_names = [name for (name, module) in model_def.items() if isinstance(module, Input)]
        self.output_names = [name for (name, module) in model_def.items() if module['output']]
        self.model_def = model_def
        self.modules = self.build(self.model_def)

    def build(self, model_def):
        modules = {}
        for name, module_def in model_def.items():
            if isinstance(module_def, Input):
                continue
            modules[name] = module_def.get_module()
            setattr(self, name[1:], modules[name][1])
        return modules

    def forward(self, x):
        module_outputs = {self.input_names[0]: x}
        for name, (input_gen, module) in self.modules.items():
            input_tensor = input_gen(self.model_def[name]['in_name'], module_outputs)
            module_outputs[name] = module(input_tensor)
        model_output = {name: module_outputs[name] for name in self.output_names}
        return model_output



class ModuleDefBase:
    INPUT_NAMES = ['input0', 'input1', 'input2']
    
    def __init__(self):
        self.args = []
        self.props = {}

    def __getitem__(self, key):
        return self.props[key]

    def __setitem__(self, key, value):
        self.props[key] = value

    def __str__(self):
        return self.__class__.__name__ + str(self.props)

    def __contains__(self, key):
        return key in self.props

    def fill_and_append(self, building_modules):
        pass

    def fill_default(self, bef_module):
        if 'out_channels' in bef_module:
            self['in_channels'] = bef_module['out_channels']
            self['out_channels'] = self['in_channels']
        elif 'in_channels' in self:
            del self.props['in_channels']
            del self.props['out_channels']
        if 'out_features' in bef_module:
            self['in_features'] = bef_module['out_features']
            self['out_features'] = self['in_features']
        elif 'in_features' in self:
            del self.props['in_features']
            del self.props['out_features']
        if 'out_resol' in bef_module:
            self['in_resol'] = bef_module['out_resol']
            self['out_resol'] = self['in_resol']
        elif 'in_resol' in self:
            del self.props['in_resol']
            del self.props['out_resol']

    def append_name_prefix(self, prefix):
        if prefix is None:
            return
        self['name'] = prefix + '/' + self['name']
        if 'in_name' not in self:
            return

        def append_prefix(prefix, in_name):
            if not in_name.startswith('/') and not in_name.startswith('input'):
                return prefix + '/' + in_name
            else:
                return in_name

        if isinstance(self['in_name'], list):
            self['in_name'] = [append_prefix(prefix, in_name) for in_name in self['in_name']]
        else:
            self['in_name'] = append_prefix(prefix, self['in_name'])

    def get_module(self):
        pass

    def single_input(self, src_name, prior_outputs):
        return prior_outputs[src_name]

    def multi_input(self, src_names, prior_outputs):
        x =  [prior_outputs[src] for src in src_names]
        return x

    def propagate_in_name(self, new_in_name):
        if isinstance(new_in_name, list):
            replacement = dict(zip(ModuleDefBase.INPUT_NAMES, new_in_name))
        else:
            replacement = {ModuleDefBase.INPUT_NAMES[0]: new_in_name}
        if isinstance(self['in_name'], list):
            for i, in_name in enumerate(self['in_name']):
                if in_name in replacement:
                    self['in_name'][i] = replacement[in_name]
        else:
            if self['in_name'] in replacement:
                self['in_name'] = replacement[self['in_name']]

class Input(ModuleDefBase):
    def __init__(self, name, chw_shape, output=False):
        super().__init__()
        out_resol = np.array(chw_shape[1:], dtype=np.int32)
        self.props = {'name': name, 'out_channels': chw_shape[0], 'out_resol': out_resol, 'output': output}

    def fill_and_append(self, building_modules):
        building_modules[self['name']] = self
        return building_modules


class Conv2d(ModuleDefBase):
    def __init__(self, name, in_name, out_channels=None, kernel_size=3, padding='same', stride=1, output=False):
        super().__init__()
        self.props = {'name': name, 'in_name': in_name, 'in_channels': None, 'out_channels': out_channels,
                      'stride': stride, 'kernel_size': kernel_size, 'padding': padding,
                      'in_resol': None, 'out_resol': None, 'output': output}
        self.args = ['in_channels', 'out_channels', 'kernel_size', 'padding', 'stride']

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        assert bef_module['out_channels'] is not None
        self['in_channels'] = bef_module['out_channels']
        self['in_resol'] = bef_module['out_resol']
        self['out_resol'] = self['in_resol'] // self['stride']
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
                      'in_resol': None, 'out_resol': None, 'output': output}
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
                      'in_resol': None, 'out_resol': None, 'output': output}
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


class Arithmetic(ModuleDefBase):
    def __init__(self, name, function, in_name, output=False):
        super().__init__()
        self.props = {'name': name, 'function': function, 'in_name': in_name,
                      'in_channels': None, 'out_channels': None,
                      'in_resol': None, 'out_resol': None, 'output': output}

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
                      'in_resol': None, 'output': output}

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
                      'in_features': None, 'out_features': out_features, 'output': output}
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


class BlockModuleDefBase(ModuleDefBase):
    def __init__(self, name, in_name=None):
        super().__init__()
        if name is not None:
            self.props['name'] = name
        if in_name is not None:
            self.props['in_name'] = in_name
        self.block_def = []

    def fill_and_append(self, building_modules):
        for module_def in self.block_def:
            module_def.append_name_prefix(self['name'])
            if 'in_name' in self:
                module_def.propagate_in_name(self['in_name'])
            building_modules = module_def.fill_and_append(building_modules)
        return building_modules

    def __str__(self):
        text = self.__class__.__name__ + '={\n'
        text += f"\tname: {self['name']}\n"
        if 'in_name' in self:
            text += f"\tin_name: {self['in_name']}\n"
        for index, (module_def) in enumerate(self.block_def):
            text += f"\t{index:02}, {module_def}\n"
        text += '}'
        return text


class ModelDefFactory(BlockModuleDefBase):
    def __init__(self, architecture, src_shape):
        super().__init__('')    # top name must be ''(empty) or start with '/' e.g. '/model'
        self.block_def = [
            Input('image', chw_shape=src_shape),
            ResNet1('bkbn', in_name='image', out_channels=64),
            Classifier('clsf', in_name='bkbn/conv2/relu', num_class=10)
        ]
        self.model_def = self.fill_and_append({})
        print(self)

    def get_model_def(self):
        return self.model_def

    def __str__(self):
        text = "[ModelDefFactory] Model{\n"
        for index, (name, module_def) in enumerate(self.model_def.items()):
            text += f"\t{index:02}: {module_def}\n"
        text += '}'
        return text


class ResNet1(BlockModuleDefBase):
    def __init__(self, name, in_name, out_channels):
        super().__init__(name, in_name)
        self.block_def = [
            Conv2d('conv1', in_name=ModuleDefBase.INPUT_NAMES[0], out_channels=32),
            Activation('conv1/relu', function='relu', in_name='conv1'),
            MaxPool2d('pool1', in_name='conv1/relu', kernel_size=2, stride=2),
            ResBlock('resblock1', in_name='conv1/relu'),
            ResBlock('resblock2', in_name='conv1/relu'),
            MaxPool2d('pool2', in_name='resblock2/add', kernel_size=2, stride=2),
            Conv2d('conv2', in_name='pool2', out_channels=out_channels),
            Activation('conv2/relu', function='relu', in_name='conv2'),
        ]


class ResBlock(BlockModuleDefBase):
    def __init__(self, name, in_name):
        super().__init__(name, in_name)
        self.block_def = [
            Conv2d('conv1', in_name=ModuleDefBase.INPUT_NAMES[0]),
            Activation('conv1/relu', function='relu', in_name='conv1'),
            Conv2d('conv2', in_name='conv1/relu'),
            Activation('conv2/relu', function='relu', in_name='conv2'),
            Arithmetic('add', function='add', in_name=[ModuleDefBase.INPUT_NAMES[0], 'conv2/relu'])
        ]

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        out_channels = bef_module['out_channels']
        self.block_def[0]['out_channels'] = out_channels // 2
        self.block_def[2]['out_channels'] = out_channels
        for module_def in self.block_def:
            module_def.append_name_prefix(self['name'])
            if 'in_name' in self:
                module_def.propagate_in_name(self['in_name'])
            building_modules = module_def.fill_and_append(building_modules)
        return building_modules


class Classifier(BlockModuleDefBase):
    def __init__(self, name, in_name, num_class):
        super().__init__(name, in_name)
        self.block_def = [
            Flatten('flatten', in_name=ModuleDefBase.INPUT_NAMES[0]),
            Linear('linear1', in_name='flatten', out_features=100),
            Activation('linear1/relu', function='relu', in_name='linear1'),
            Linear('linear2', in_name='linear1/relu', out_features=num_class, output=True),
            Activation('linear2/softmax', function='softmax', in_name='linear2', dim=-1, output=True),
        ]


if __name__ == "__main__":
    # model_def = ModelDefFactory(None, (3, 320, 320)).get_model_def()
    model = ModelTemplate(cfg.Architecture)
    x = torch.rand((2, 3, 320, 320), dtype=torch.float32)
    y = model(x)
    print("output shape", y)
