import torch

import config as cfg
from pytch.model.module_def_base import ModuleDefBase, BlockModuleDefBase
import pytch.model.basic_module_def as bmd
import pytch.model.model_util as mu
import pytch.utils.util_function as puf

IN_NAMES = ModuleDefBase.INPUT_NAMES


class ModelTemplate(torch.nn.Module):
    def __init__(self, architecture, input_shape=(3, 320, 320)):
        super().__init__()
        print("===== model template init")
        model_def = ModelDefFactory(architecture, input_shape).get_model_def()
        self.post_proc = None
        if architecture.POSTPROCESS == 'detector':
            self.post_proc = mu.DetectorPostProcess()
        self.input_names = [name for (name, module) in model_def.items() if isinstance(module, bmd.Input)]
        self.output_names = [name for (name, module) in model_def.items() if module['output']]
        self.model_def = model_def
        self.modules = self.build(self.model_def)

    def build(self, model_def):
        modules = {}
        for name, module_def in model_def.items():
            if isinstance(module_def, bmd.Input):
                continue
            modules[name] = module_def.get_module()
            setattr(self, name[1:], modules[name][1])
        return modules

    def forward(self, x):
        module_outputs = {self.input_names[0]: x['image']}
        for name, (input_gen, module) in self.modules.items():
            input_tensor = input_gen(self.model_def[name]['in_name'], module_outputs)
            module_outputs[name] = module(input_tensor)
        model_output = {name: module_outputs[name] for name in self.output_names}
        if self.post_proc:
            model_output = self.post_proc(model_output)
        return model_output


class ModelDefFactory(BlockModuleDefBase):
    def __init__(self, architecture, chw_shape):
        super().__init__('')    # top name must be ''(empty) or start with '/' e.g. '/model'
        ModuleDefBase.CLASS_COUNT = {}
        self.block_def = self.define_model(architecture, chw_shape)
        self.model_def = self.fill_and_append({})
        print(self)

    def define_model(self, architecture, chw_shape):
        block_def = [bmd.Input('image', chw_shape)]

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
            line = f"\t{index:02}: {module_def}\n"
            if 'Activation' not in line and 'BatchNormalization' not in line:
                text += line
        text += '}'
        return text


class ResNet1(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None, out_channels=None):
        super().__init__(name, in_name)
        self.block_def = [
            bmd.Conv2d('conv1', in_name=IN_NAMES[0], out_channels=32),
            bmd.Activation('conv1/relu', function='relu', in_name='conv1'),
            bmd.MaxPool2d('pool1', in_name='conv1/relu', kernel_size=2, stride=2),
            Residual('resblock1', in_name='conv1/relu'),
            Residual('resblock2', in_name='conv1/relu'),
            bmd.MaxPool2d('pool2', in_name='resblock2/add', kernel_size=2, stride=2),
            bmd.Conv2d('conv2', in_name='pool2', out_channels=out_channels),
            bmd.Activation('conv2/relu', function='relu', in_name='conv2'),
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
        self.block_def.append(bmd.Padding())
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
            bmd.Arithmetic('add1', function='add', in_name=[IN_NAMES[0], 'conv2'])
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
            bmd.Conv2d('conv', IN_NAMES[0], out_channels, kernel_size, padding, stride),
            bmd.Activation('relu', function=function, in_name='conv'),
            bmd.BatchNormalization('bn', in_name='relu', output=output),
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
            bmd.Flatten('flatten', in_name=IN_NAMES[0]),
            bmd.Linear('linear1', in_name='flatten', out_features=100),
            bmd.Activation('linear1/relu', function='relu', in_name='linear1'),
            bmd.Linear('linear2', in_name='linear1/relu', out_features=num_class, output=True),
            bmd.Activation('linear2/softmax', function='softmax', in_name='linear2', dim=-1, output=True),
        ]
        self['out_name'] = 'linear2/softmax'
        self['alias'] = 'clsf'


class FPN(BlockModuleDefBase):
    def __init__(self, name=None, in_name=None):
        super().__init__(name, in_name)
        anchors_per_scale = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        channels_per_anchor = sum(list(cfg.ModelOutput.PRED_FMAP_COMPOSITION.values()))
        last_channels = anchors_per_scale * channels_per_anchor
        self.block_def = []
        self.append_conv_5x(IN_NAMES[2], 512, 'head_logit5', last_channels)
        self.upsample_concat(IN_NAMES[2], IN_NAMES[1], 256, 4)
        self.append_conv_5x('cat_4', 256, 'head_logit4', last_channels)
        self.upsample_concat('cat_4', IN_NAMES[0], 256, 3)
        self.append_conv_5x('cat_3', 128, 'head_logit3', last_channels)
        self['out_name'] = ['head_logit5', 'head_logit4', 'head_logit3']
        self['alias'] = 'fpn'

    def append_conv_5x(self, in_name, out_channels, last_name, last_channels):
        self.block_def += [
            ConvActiBN(in_name=in_name, out_channels=out_channels, kernel_size=1),
            ConvActiBN(out_channels=out_channels * 2),
            ConvActiBN(out_channels=out_channels, kernel_size=1),
            ConvActiBN(out_channels=out_channels * 2),
            ConvActiBN(out_channels=out_channels, kernel_size=1),
            ConvActiBN(out_channels=out_channels * 2),
            bmd.Conv2d(name=last_name, out_channels=last_channels, kernel_size=1),
        ]

    def upsample_concat(self, upper, lower, out_channels, scale_num):
        scale_str = f'{scale_num}'
        self.block_def += [
            ConvActiBN('cvab_' + scale_str, in_name=upper, out_channels=out_channels, kernel_size=1),
            bmd.Upsample2d('upsamp_' + scale_str, in_name='cvab_' + scale_str),
            bmd.Concat('cat_' + scale_str, in_name=['upsamp_' + scale_str, lower]),
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
    puf.print_structure("model output", y)
