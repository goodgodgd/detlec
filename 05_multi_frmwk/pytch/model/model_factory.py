from pprint import PrettyPrinter
import torch

import config as cfg
from pytch.model.basic_module_def import *

IN_NAMES = ModuleDefBase.INPUT_NAMES


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
            Conv2d('conv1', in_name=IN_NAMES[0], out_channels=32),
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
            Conv2d('conv1', in_name=IN_NAMES[0]),
            Activation('conv1/relu', function='relu', in_name='conv1'),
            Conv2d('conv2', in_name='conv1/relu'),
            Activation('conv2/relu', function='relu', in_name='conv2'),
            Arithmetic('add', function='add', in_name=[IN_NAMES[0], 'conv2/relu'])
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
            Flatten('flatten', in_name=IN_NAMES[0]),
            Linear('linear1', in_name='flatten', out_features=100),
            Activation('linear1/relu', function='relu', in_name='linear1'),
            Linear('linear2', in_name='linear1/relu', out_features=num_class, output=True),
            Activation('linear2/softmax', function='softmax', in_name='linear2', dim=-1, output=True),
        ]


if __name__ == "__main__":
    model = ModelTemplate(cfg.Architecture)
    x = torch.rand((2, 3, 320, 320), dtype=torch.float32)
    y = model(x)
    print("model outputs:")
    for k, v in y.items():
        print(f"\t{k}: {v.shape}")
