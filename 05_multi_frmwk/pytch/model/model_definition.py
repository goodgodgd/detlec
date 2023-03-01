import numpy as np
from pprint import PrettyPrinter


class ModuleParts:
    @staticmethod
    def get_cnn(src_shape):
        channel, height, width = src_shape
        module = {
            'image1': {'type': 'input', 'out_channels': channel, 'out_resol': np.array((height, width), dtype=np.int32)},
            'conv1': {'type': 'conv2d', 'input': 'image1', 'in_channels': None, 'out_channels': 32, 'kernel_size': 3, 'padding': None, 'activation': 'relu', 'in_resol': None, 'out_resol': None},
            'resblock1': {'type': 'resblock', 'input': 'conv1'},
            'pool1': {'type': 'maxpool', 'input': 'resblock1/add', 'kernel_size': 2, 'stride': 2, 'in_channels': None, 'out_channels': None, 'in_resol': None, 'out_resol': None},
            'conv3': {'type': 'conv2d', 'input': 'pool1', 'out_channels': 64, 'kernel_size': 3, 'activation': 'relu', 'output': True},
        }
        return module

    @staticmethod
    def get_resblock(block_name, src_name, in_channels):
        module = {
            'conv1': {'type': 'conv2d', 'input': 'src', 'out_channels': in_channels // 2, 'kernel_size': 3, 'activation': 'relu'},
            'conv2': {'type': 'conv2d', 'input': 'conv1', 'out_channels': in_channels, 'kernel_size': 3, 'activation': 'relu'},
            'add': {'type': 'add', 'input': ['src', 'conv2']},
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
                         'maxpool': self.fill_maxpool, 'add': self.fill_add,
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
        cur_module['padding'] = 'same'
        if 'stride' in cur_module:
            cur_module['out_resol'] = cur_module['in_resol'] // cur_module['stride']
        else:
            cur_module['out_resol'] = cur_module['in_resol']
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_maxpool(self, cur_module, building_modules, cur_name):
        """
        fill in new attributes: ['in_channels', 'out_channels', 'in_resol', 'out_resol']
        """
        bef_module = building_modules[cur_module['input']]
        cur_module['in_channels'] = bef_module['out_channels']
        cur_module['out_channels'] = cur_module['in_channels']
        cur_module['in_resol'] = bef_module['out_resol']
        cur_module['out_resol'] = cur_module['in_resol'] // cur_module['stride']
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_add(self, cur_module, building_modules, cur_name):
        """
        fill in new attributes: ['in_channels', 'out_channels', 'in_resol', 'out_resol']
        """
        bef_module0 = building_modules[cur_module['input'][0]]
        bef_module1 = building_modules[cur_module['input'][0]]
        print("fild add", bef_module0['out_channels'], bef_module1['out_channels'])
        assert bef_module0['out_channels'] == bef_module1['out_channels']
        assert np.allclose(bef_module0['out_resol'], bef_module1['out_resol'])
        cur_module['in_channels'] = bef_module0['out_channels']
        cur_module['out_channels'] = cur_module['in_channels']
        cur_module['in_resol'] = bef_module0['out_resol']
        cur_module['out_resol'] = cur_module['in_resol']
        building_modules[cur_name] = cur_module
        return building_modules


import config as cfg

if __name__ == "__main__":
    model_def = ModelAssembler(cfg.Architecture).get_model_def()
    pp = PrettyPrinter(sort_dicts=False)
    pp.pprint(model_def)
