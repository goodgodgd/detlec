import numpy as np
from copy import deepcopy
from pprint import PrettyPrinter
import torch


class NamePool:
    count = 0
    names = {}

    @staticmethod
    def new_key(name: str):
        NamePool.count += 1
        NamePool.names[NamePool.count] = name
        return NamePool.count

    @staticmethod
    def get_name(key: int):
        return NamePool.names[key]


class ModelTemplate:
    def __init__(self, architecture, input_shape=(3, 320, 320)):
        model_def = ModelDefFactory(architecture, input_shape).get_model_def()
        self.input_names = [name for (name, module) in model_def.items() if isinstance(module, Input)]
        self.output_names = [name for (name, module) in model_def.items() if module['output']]
        self.model_def = model_def
        self.modules = self.build(self.model_def)
        pp = PrettyPrinter(sort_dicts=False)
        pp.pprint(self.modules)

    def build(self, model_def):
        modules = {}
        for name, part in model_def.items():
            if isinstance(part, Input):
                continue
            modules[name] = part.get_module()
        return modules


class ModuleDefBase:
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
        self['in_channels'] = bef_module['out_channels']
        self['out_channels'] = self['in_channels']
        self['in_resol'] = bef_module['out_resol']
        self['out_resol'] = self['in_resol']

    def append_name_prefix(self, prefix):
        self.append_name_prefix_to_self(prefix)

    def append_name_prefix_to_self(self, prefix):
        if prefix is None:
            return
        self['name'] = prefix + '/' + self['name']
        print("== append name", self['name'], prefix)
        if 'in_name' not in self:
            return
        print("-- append in_name1", self['in_name'])
        if isinstance(self['in_name'], list):
            self['in_name'] = [prefix + '/' + in_name if not in_name.startswith('/') else in_name
                               for in_name in  self['in_name']]
        else:
            self['in_name'] = prefix + '/' + self['in_name'] if not self['in_name'].startswith('/') else self['in_name']
        print("-- append in_name2", self['in_name'])

    def get_module(self):
        pass

    def single_input(self, src_name, prior_outputs):
        return prior_outputs[src_name]

    def multi_input(self, src_names, prior_outputs):
        x =  [prior_outputs[src] for src in src_names]
        return x


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
        self.args = ['name', 'in_channels', 'out_channels', 'kernel_size', 'padding', 'stride']

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
        self.args = ['name', 'kernel_size', 'stride']

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
        self.args = ['name'] + list(kwargs.keys())

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
        bef_module = building_modules[self['in_name']]
        self.fill_default(bef_module)
        building_modules[self['name']] = self
        return building_modules

    def get_module(self):
        if self['function'] == 'add':
            return self.single_input, lambda x: x[0] + x[1]
        elif self['function'] == 'subtract':
            return self.single_input, lambda x: x[0] - x[1]
        elif self['function'] == 'multiply':
            return self.single_input, lambda x: x[0] * x[1]
        elif self['function'] == 'divide':
            return self.single_input, lambda x: x[0] / x[1]


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
        self.args = ['name', 'in_features', 'out_features']

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


class BlockModuleDefBase(ModuleDefBase):
    def __init__(self, name, in_name=None):
        super().__init__()
        self.props['name'] = name
        if in_name is not None:
            self.props['in_name'] = in_name
        self.block_def = []

    def fill_and_append(self, building_modules):
        building_modules = {} if building_modules is None else building_modules
        print("==block module names", [block['name'] for block in self.block_def])
        print("==block module input names", [block['in_name'] for block in self.block_def if 'in_name' in block])
        for module_def in self.block_def:
            module_def.append_name_prefix(self['name'])
            print("[block fill] module def", module_def)
            print("[block fill] building", building_modules)
            building_modules = module_def.fill_and_append(building_modules)
        return building_modules


    def append_block_name(self, block_name, block_mod_def):
        new_block_mod_def = []
        block_mod_names = [module['name'] for module in block_mod_def]
        for sub_mod_def in block_mod_def:
            if 'in_name' in sub_mod_def:
                sub_input = sub_mod_def['in_name']
                if isinstance(sub_input, list):   # when input is a list of strs
                    new_sub_input = []
                    for src_name in sub_input:
                        new_name = block_name + '/' + src_name if src_name in block_mod_names else src_name
                        new_sub_input.append(new_name)
                    sub_mod_def['in_name'] = new_sub_input
                elif isinstance(sub_input, str):       # when input is a single str
                    sub_mod_def['in_name'] = block_name + '/' + sub_input if sub_input in block_mod_names else sub_input
            sub_mod_def['name'] = block_name + '/' + sub_mod_def['name']
            new_block_mod_def.append(sub_mod_def)
        return new_block_mod_def

    def __str__(self):
        text = self.__class__.__name__ + '={\n'
        text += f"\tname: {self['name']}\n"
        if 'in_name' in self:
            text += f"\tin_name: {self['in_name']}\n"
        for index, (module_def) in enumerate(self.block_def):
            text += f"\t{module_def['name']}: {module_def}\n"
        text += '}'
        return text

    def update_names(self):
        print("==before update names", self['name'], [block['name'] for block in self.block_def])
        print("==before update input names", [block['in_name'] for block in self.block_def if 'in_name' in block])
        for module_def in self.block_def:
            module_def.append_name_prefix(self['name'])
        print("==after update names", self['name'], [block['name'] for block in self.block_def])
        print("==after update input names", [block['in_name'] for block in self.block_def if 'in_name' in block])

    def append_name_prefix(self, prefix):
        src_in_name = self['in_name']
        self.append_name_prefix_to_self(prefix)
        dst_in_name = self['in_name']
        print("appending", src_in_name, dst_in_name)
        for module_def in self.block_def:
            if module_def['in_name'] == src_in_name:
                pass


class ModelDefFactory(BlockModuleDefBase):
    def __init__(self, architecture, src_shape):
        super().__init__('')    # top name must be ''(empty) or start with '/' e.g. '/model'
        self.block_def = [
            Input('image', chw_shape=src_shape),
            ResBlock('resblock', 'image')
            # ResNet1('bkbn', in_name='image', out_channels=64),
            # Classifier('clsf', in_name='conv2/relu', num_class=10)
        ]
        self.update_names()
        self.model_def = self.fill_and_append({})
        print("== final model def", self.model_def)

    def get_model_def(self):
        return self.model_def

    def __str__(self):
        text = "Model{\n"
        for index, (name, module_def) in enumerate(self.model_def.items()):
            text += f"\t{name}: {module_def}\n"
        text += '}'
        return text


class ResNet1(BlockModuleDefBase):
    def __init__(self, name, in_name, out_channels):
        super().__init__(name, in_name)
        self.block_def = [
            # 'image1': Input(out_channels=channel, out_resol=np.array((height, width), dtype=np.int32)),
            Conv2d('conv1', in_name=in_name, out_channels=32),
            Activation('conv1/relu', function='relu', in_name='conv1'),
            ResBlock('resblock1', in_name='conv1/relu'),
            ResBlock('resblock2', in_name='conv1/relu'),
            MaxPool2d('pool1', in_name='resblock2/add', kernel_size=2, stride=2),
            Conv2d('conv2', in_name='pool1', out_channels=out_channels),
            Activation('conv2/relu', function='relu', in_name='conv2'),
        ]


class ResBlock(BlockModuleDefBase):
    def __init__(self, name, in_name):
        super().__init__(name, in_name)
        self.block_def = [
            Conv2d('conv1', in_name=in_name),
            Activation('conv1/relu', function='relu', in_name='conv1'),
            Conv2d('conv2', in_name='conv1/relu'),
            Activation('conv2/relu', function='relu', in_name='conv2'),
            Arithmetic('add', function='add', in_name=[in_name, 'conv2/relu'])
        ]


class Classifier(BlockModuleDefBase):
    def __init__(self, name, in_name, num_class):
        super().__init__(name, in_name)
        self.block_def = [
            Flatten('flatten', in_name=in_name),
            Linear('linear1', in_name='flatten', out_features=100),
            Activation('linear1/relu', function='relu', in_name='linear1'),
            Linear('linear2', in_name='linear1/relu', out_features=num_class, output=True),
            Activation('linear2/softmax', function='softmax', in_name='linear2', dim=-1, output=True),
        ]


if __name__ == "__main__":
    model_def = ModelDefFactory(None, (3, 320, 320)).get_model_def()
    pp = PrettyPrinter(sort_dicts=False)
    pp.pprint(model_def)
