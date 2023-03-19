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
        self.fill_names()
        self.out_name_to_in_name()
        for module_def in self.block_def:
            module_def.append_name_prefix(self['name'])
            if 'in_name' in self:
                module_def.propagate_in_name(self['in_name'])
            building_modules = module_def.fill_and_append(building_modules)
        return building_modules

    def fill_names(self):
        """
        fill empty name and in_name
        """
        for i, module_def in enumerate(self.block_def):
            module_def.make_sure_have_name()
            if i > 0 and (('in_name' not in module_def) or module_def['in_name'] is None):
                module_def['in_name'] = self.block_def[i-1]['name']

    def out_name_to_in_name(self):
        """
        when in_name is a name of a block, replace in_name with out_name of the block
        """
        block_def_dict = {}
        for i, module_def in enumerate(self.block_def):
            block_def_dict[module_def['name']] = module_def
            if isinstance(module_def, Input):
                continue

            cur_in_names = module_def['in_name']
            if isinstance(cur_in_names, str):
                cur_in_names = [cur_in_names]

            new_in_names = []
            for in_name in cur_in_names:
                if in_name in block_def_dict:
                    bef_block_def = block_def_dict[in_name]
                    if 'out_name' in bef_block_def:
                        bef_out_name = bef_block_def.get_out_name()
                        if isinstance(bef_out_name, list):
                            new_in_names += bef_out_name
                        else:
                            new_in_names.append(bef_out_name)
                    else:
                        new_in_names.append(in_name)
                else:
                    new_in_names.append(in_name)

            if len(new_in_names) == 1:
                new_in_names = new_in_names[0]
            module_def['in_name'] = new_in_names

    def get_out_name(self):
        if isinstance(self['out_name'], list):
            return [self['name'] + '/' + out_name for out_name in self['out_name']]
        else:
            return self['name'] + '/' + self['out_name']

    def __str__(self):
        text = self.__class__.__name__ + '={\n'
        if 'name' in self:
            text += f"\tname: {self['name']}\n"
        if 'in_name' in self:
            text += f"\tin_name: {self['in_name']}\n"
        for index, (module_def) in enumerate(self.block_def):
            text += f"\t{index:02}, {module_def}\n"
        text += '}'
        return text


class ModelDefFactory(BlockModuleDefBase):
    def __init__(self, architecture, chw_shape):
        super().__init__('')    # top name must be ''(empty) or start with '/' e.g. '/model'
        self.block_def = self.define_model(architecture, chw_shape)
        self.model_def = self.fill_and_append({})
        print(self)

    def define_model(self, architecture, chw_shape):
        block_def = [Input('image', chw_shape)]

        if architecture.BACKBONE.lower() == 'darknet53':
            block_def.append(Darknet53('dark53'))
        elif architecture.BACKBONE.lower() == 'resnet1':
            block_def.append(ResNet1('resnet', out_channels=64))
        else:
            raise ValueError(f"[define_model] {architecture.BACKBONE} is NOT implemented")

        if architecture.HEAD.lower() == 'fpn':
            block_def.append(FPN('FPN'))
        elif architecture.HEAD.lower() == 'classifier':
            block_def.append(Classifier('classifier', num_class=10))
        else:
            raise ValueError(f"[define_model] {architecture.HEAD} is NOT implemented")

        return block_def

    def get_model_def(self):
        return self.model_def

    def __str__(self):
        text = "[ModelDefFactory] Model{\n"
        for index, (name, module_def) in enumerate(self.model_def.items()):
            text += f"\t{index:02}: {module_def}\n"
        text += '}'
        return text


class ResNet1(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None, out_channels=None):
        super().__init__(name, in_name)
        self.block_def = [
            Conv2d('conv1', in_name=IN_NAMES[0], out_channels=32),
            Activation('conv1/relu', function='relu', in_name='conv1'),
            MaxPool2d('pool1', in_name='conv1/relu', kernel_size=2, stride=2),
            Residual('resblock1', in_name='conv1/relu'),
            Residual('resblock2', in_name='conv1/relu'),
            MaxPool2d('pool2', in_name='resblock2/add', kernel_size=2, stride=2),
            Conv2d('conv2', in_name='pool2', out_channels=out_channels),
            Activation('conv2/relu', function='relu', in_name='conv2'),
        ]
        self['out_name'] = 'conv2/relu'
        self['alias'] = 'resnet1'


class Darknet53(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None):
        super().__init__(name, in_name)
        self.block_def = [
            ConvActiBN(in_name=IN_NAMES[0], out_channels=32),
        ]
        self.append_resi_blocks(64, 1)
        self.append_resi_blocks(128, 2)
        self.append_resi_blocks(256, 8)
        self.append_resi_blocks(512, 8)
        self.append_resi_blocks(1024, 4)
        self['alias'] = 'dark53'
        self['out_name'] = ['resd11/add1', 'resd19/add1', 'resd23/add1']

    def append_resi_blocks(self, out_channels, residual_iter):
        self.block_def.append(Padding())
        self.block_def.append(ConvActiBN(out_channels=out_channels, stride=2))
        for i in range(residual_iter):
            self.block_def.append(Residual())

    def fill_and_append(self, building_modules):
        ModuleDefBase.NAMESPACE = 'dark/'
        building_modules = super(Darknet53, self).fill_and_append(building_modules)
        ModuleDefBase.NAMESPACE = ''
        return building_modules


class Residual(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None):
        super().__init__(name, in_name)
        self.block_def = [
            ConvActiBN('conv1', in_name=IN_NAMES[0]),
            ConvActiBN('conv2', in_name='conv1/relu'),
            Arithmetic('add1', function='add', in_name=[IN_NAMES[0], 'conv2'])
        ]
        self['out_name'] = 'add1'
        self['alias'] = 'resd'

    def fill_and_append(self, building_modules):
        bef_module = building_modules[self['in_name']]
        out_channels = bef_module['out_channels']
        self.block_def[0]['out_channels'] = out_channels // 2
        self.block_def[1]['out_channels'] = out_channels
        return super(Residual, self).fill_and_append(building_modules)


class ConvActiBN(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None, out_channels=None, kernel_size=3,
                 padding='same', stride=1, function='relu', output=False):
        super().__init__(name, in_name)
        self.block_def = [
            Conv2d('conv', IN_NAMES[0], out_channels, kernel_size, padding, stride),
            Activation('relu', function=function, in_name='conv'),
            BatchNormalization('bn', in_name='relu', output=output),
        ]
        self['out_name'] = 'relu'
        self['out_name'] = 'bn'
        self['alias'] = 'cvab'

    def fill_and_append(self, building_modules):
        if 'out_channels' in self and self['out_channels'] is not None:
            self.block_def[0]['out_channels'] = self['out_channels']
        return super(ConvActiBN, self).fill_and_append(building_modules)


class Classifier(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None, num_class=0):
        super().__init__(name, in_name)
        self.block_def = [
            Flatten('flatten', in_name=IN_NAMES[0]),
            Linear('linear1', in_name='flatten', out_features=100),
            Activation('linear1/relu', function='relu', in_name='linear1'),
            Linear('linear2', in_name='linear1/relu', out_features=num_class, output=True),
            Activation('linear2/softmax', function='softmax', in_name='linear2', dim=-1, output=True),
        ]
        self['out_name'] = 'linear2/softmax'
        self['alias'] = 'clsf'


class FPN(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None):
        super().__init__(name, in_name)
        self.num_anchors_per_scale = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        self.pred_channels_per_anchor = sum(list(cfg.ModelOutput.PRED_FMAP_COMPOSITION.values()))
        last_channels = self.num_anchors_per_scale * self.pred_channels_per_anchor
        self.block_def = []
        self.append_conv_5x(IN_NAMES[2], 512, 'pred_raw5', last_channels)
        self.upsample_concat(IN_NAMES[2], IN_NAMES[1], 256, 4)
        self.append_conv_5x('cat_4', 256, 'pred_raw4', last_channels)
        self.upsample_concat('cat_4', IN_NAMES[0], 256, 3)
        self.append_conv_5x('cat_3', 128, 'pred_raw3', last_channels)
        self['out_name'] = ['pred_raw5', 'pred_raw4', 'pred_raw3']
        self['alias'] = 'fpn'

    def append_conv_5x(self, in_name, out_channels, last_name, last_channels):
        self.block_def += [
            ConvActiBN(in_name=in_name, out_channels=out_channels, kernel_size=1),
            ConvActiBN(out_channels=out_channels * 2),
            ConvActiBN(out_channels=out_channels, kernel_size=1),
            ConvActiBN(out_channels=out_channels * 2),
            ConvActiBN(out_channels=out_channels, kernel_size=1),
            ConvActiBN(out_channels=out_channels * 2),
            Conv2d(name=last_name, out_channels=last_channels, kernel_size=1),
        ]

    def upsample_concat(self, upper, lower, out_channels, scale_num):
        scale_str = f'{scale_num}'
        self.block_def += [
            ConvActiBN('cvab_' + scale_str, in_name=upper, out_channels=out_channels, kernel_size=1),
            Upsample2d('upsamp_' + scale_str, in_name='cvab_' + scale_str),
            Concat('cat_' + scale_str, in_name=['upsamp_' + scale_str, lower]),
        ]

    def fill_and_append(self, building_modules):
        ModuleDefBase.NAMESPACE = 'fpn/'
        building_modules = super(FPN, self).fill_and_append(building_modules)
        for local_name in self['out_name']:
            out_name = self['name'] + '/' + local_name
            building_modules[out_name]['output'] = True
        ModuleDefBase.NAMESPACE = ''
        return building_modules


if __name__ == "__main__":
    model = ModelTemplate(cfg.Architecture)
    x = torch.rand((2, 3, 320, 320), dtype=torch.float32)
    y = model(x)
    print("model outputs:")
    for k, v in y.items():
        print(f"\t{k}: {v.shape}")
