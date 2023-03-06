import numpy as np
from pprint import PrettyPrinter


class ModuleParts:
    @staticmethod
    def get_cnn(src_shape):
        channel, height, width = src_shape
        module = {
            'image1': {'type': 'input', 'out_channels': channel, 'out_resol': np.array((height, width), dtype=np.int32)},
            'conv1': {'type': 'conv2d', 'input': 'image1', 'out_channels': 32, 'kernel_size': 3},
            'conv1/relu': {'type': 'relu', 'input': 'conv1'},
            'resblock1': {'type': 'resblock', 'input': 'conv1/relu'},
            'pool1': {'type': 'maxpool', 'input': 'resblock1/add', 'kernel_size': 2, 'stride': 2},
            'conv2': {'type': 'conv2d', 'input': 'pool1', 'out_channels': 64, 'kernel_size': 3, 'output': True},
            'conv2/relu': {'type': 'relu', 'input': 'conv2'},
            'flatten': {'type': 'flatten', 'input': 'conv2/relu'},
            'linear1': {'type': 'linear', 'input': 'flatten', 'out_channels': 100},
            'linear1/relu': {'type': 'relu', 'input': 'linear1'},
            'linear2': {'type': 'linear', 'input': 'linear1/relu', 'out_channels': 10, 'output': True},
            'linear2/softmax': {'type': 'softmax', 'input': 'linear2', 'dim': -1, 'output': True},
        }
        return module

    @staticmethod
    def get_resblock(block_name, src_name, in_channels):
        module = {
            'conv1': {'type': 'conv2d', 'input': 'src', 'out_channels': in_channels // 2, 'kernel_size': 3},
            'conv1/relu': {'type': 'relu', 'input': 'conv1'},
            'conv2': {'type': 'conv2d', 'input': 'conv1/relu', 'out_channels': in_channels, 'kernel_size': 3},
            'conv2/relu': {'type': 'relu', 'input': 'conv2'},
            'add': {'type': 'add', 'input': ['src', 'conv2/relu']},
        }
        module = ModuleParts.append_block_name(block_name, module)
        module = ModuleParts.replace_input_name(module, {'src': src_name})
        return module

    @staticmethod
    def append_block_name(block_name, block_module):
        new_block_module = {}
        for sub_name, sub_module in block_module.items():
            sub_input = sub_module['input']
            if isinstance(sub_input, list):   # when input is a list of strs
                new_sub_input = []
                for src_name in sub_input:
                    new_name = block_name + '/' + src_name if src_name in block_module else src_name
                    new_sub_input.append(new_name)
                sub_module['input'] = new_sub_input
            elif isinstance(sub_input, str):       # when input is a single str
                sub_module['input'] = block_name + '/' + sub_input if sub_input in block_module else sub_input

            new_block_module[block_name + '/' + sub_name] = sub_module
        return new_block_module

    @staticmethod
    def replace_input_name(block_module, name_map):
        for sub_name, sub_module in block_module.items():
            sub_input = sub_module['input']
            for old_name, new_name in name_map.items():
                if isinstance(sub_input, list):   # when input is a list of strs
                    # replace old_name with new_name in sub_input
                    sub_module['input'] = [new_name if cur_name == old_name else cur_name for cur_name in sub_input]
                elif isinstance(sub_input, str):       # when input is a single str
                    sub_module['input'] = new_name if sub_input == old_name else sub_input
        return block_module


class ModelAssembler:
    def __init__(self, architecture, input_shape=(3, 320, 320)):
        model_def = ModuleParts.get_cnn(input_shape)
        self.model_def = self.fill_modules_def(model_def)
        pp = PrettyPrinter(sort_dicts=False)
        pp.pprint(self.model_def)

    def get_model_def(self):
        return self.model_def

    def fill_modules_def(self, adding_modules, new_modules=None):
        new_modules = {} if new_modules is None else new_modules
        module_filler = {'input': self.fill_input, 'conv2d': self.fill_conv2d, 'resblock': self.fill_resblock,
                         'maxpool': self.fill_maxpool, 'add': self.fill_add, 'relu': self.fill_default,
                         'flatten': self.fill_flatten, 'linear': self.fill_linear, 'softmax': self.fill_softmax
                         }
        for name, module_def in adding_modules.items():
            new_modules = module_filler[module_def['type']](module_def, new_modules, name)
        return new_modules

    def fill_input(self, cur_module, building_modules, cur_name):
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_resblock(self, cur_module, building_modules, cur_name):
        bef_module = building_modules[cur_module['input']]
        in_channels = bef_module['out_channels']
        block_modules = ModuleParts.get_resblock(cur_name, cur_module['input'], in_channels)
        block_modules = self.fill_modules_def(block_modules, building_modules)
        return block_modules

    def fill_conv2d(self, cur_module, building_modules, cur_name):
        """
        fill in new attributes: ['in_channels', 'in_resol', 'out_resol', 'padding']
        """
        bef_module = building_modules[cur_module['input']]
        cur_module['in_channels'] = bef_module['out_channels']
        cur_module['in_resol'] = bef_module['out_resol']
        cur_module['out_resol'] = cur_module['in_resol']
        cur_module['padding'] = 'same'
        if 'stride' not in cur_module:
            cur_module['stride'] = 1
        cur_module['out_resol'] = cur_module['in_resol'] // cur_module['stride']
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_maxpool(self, cur_module, building_modules, cur_name):
        """
        fill in new attributes: ['in_channels', 'out_channels', 'in_resol', 'out_resol']
        """
        cur_module = self.set_default(cur_module, building_modules)
        cur_module['out_resol'] = cur_module['in_resol'] // cur_module['stride']
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_add(self, cur_module, building_modules, cur_name):
        """
        fill in new attributes: ['in_channels', 'out_channels', 'in_resol', 'out_resol']
        """
        bef_module0 = building_modules[cur_module['input'][0]]
        bef_module1 = building_modules[cur_module['input'][0]]
        assert bef_module0['out_channels'] == bef_module1['out_channels']
        assert np.allclose(bef_module0['out_resol'], bef_module1['out_resol'])
        cur_module['in_channels'] = bef_module0['out_channels']
        cur_module['out_channels'] = cur_module['in_channels']
        cur_module['in_resol'] = bef_module0['out_resol']
        cur_module['out_resol'] = cur_module['in_resol']
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_flatten(self, cur_module, building_modules, cur_name):
        bef_module = building_modules[cur_module['input']]
        cur_module['in_channels'] = bef_module['out_channels']
        cur_module['in_resol'] = bef_module['out_resol']
        cur_module['out_channels'] = cur_module['in_channels'] * cur_module['in_resol'][0] * cur_module['in_resol'][1]
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_linear(self, cur_module, building_modules, cur_name):
        bef_module = building_modules[cur_module['input']]
        cur_module['in_channels'] = bef_module['out_channels']
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_softmax(self, cur_module, building_modules, cur_name):
        cur_module = self.set_default(cur_module, building_modules)
        cur_module['dim'] = 1
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_default(self, cur_module, building_modules, cur_name):
        building_modules[cur_name] = self.set_default(cur_module, building_modules)
        return building_modules

    def set_default(self, cur_module, building_modules):
        """
        fill in attributes by the same value with bef_module: ['in_channels', 'out_channels', 'in_resol', 'out_resol']
        """
        bef_module = building_modules[cur_module['input']]
        cur_module['in_channels'] = bef_module['out_channels']
        cur_module['out_channels'] = cur_module['in_channels']
        if 'out_resol' in bef_module:
            cur_module['in_resol'] = bef_module['out_resol']
            cur_module['out_resol'] = cur_module['in_resol']
        return cur_module


import config as cfg

if __name__ == "__main__":
    model_def = ModelAssembler(cfg.Architecture).get_model_def()
    pp = PrettyPrinter(sort_dicts=False)
    pp.pprint(model_def)
