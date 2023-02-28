import numpy as np
from pprint import PrettyPrinter


class ModuleParts:
    @staticmethod
    def get_cnn(src_shape):
        channel, height, width = src_shape
        module = {
            'image1': {'module': 'image', 'out_channels': channel, 'out_resol': np.array((height, width), dtype=np.int32)},
            'conv1': {'module': 'conv2d', 'input': 'image1', 'in_channels': None, 'out_channels': 32, 'kernel_size': 3, 'activation': 'relu', 'in_resol': None, 'out_resol': None},
            'resblock1': {'module': 'resblock', 'input': 'conv1'},
            'pool1': {'module': 'maxpool', 'input': 'resblock1/add', 'stride': 2, 'in_channels': None, 'out_channels': None, 'in_resol': None, 'out_resol': None},
            'conv3': {'module': 'conv2d', 'input': 'pool1', 'out_channels': 64, 'kernel_size': 3, 'activation': 'relu'},
        }
        return module

    @staticmethod
    def get_resblock(block_name, src_name, in_channels):
        module = {
            'conv1': {'module': 'conv2d', 'input': None, 'out_channels': in_channels // 2, 'kernel_size': 3, 'activation': 'relu'},
            'conv2': {'module': 'conv2d', 'input': 'conv1', 'out_channels': in_channels, 'kernel_size': 3, 'activation': 'relu'},
            'add': {'module': 'add', 'input': ['conv1', 'conv2']},
        }
        module = ModuleParts.append_block_name(block_name, module)
        module[block_name + '/' + 'conv1']['input'] = src_name
        return module

    @staticmethod
    def append_block_name(block_name, block_module):
        new_block_module = {}
        for sub_name, sub_module in block_module.items():
            sub_input = sub_module['input']
            if sub_input is None:
                pass
            elif isinstance(sub_input, list):   # when input is a list of strs
                new_sub_input = []
                for src_name in sub_input:
                    if src_name in block_module:
                        new_sub_input.append(block_name + '/' + src_name)
                    else:
                        new_sub_input.append(src_name)
                sub_module['input'] = new_sub_input
            else:       # when input is a single str
                if sub_input in block_module:
                    sub_module['input'] = block_name + '/' + sub_input
                else:
                    sub_module['input'] = sub_input

            new_block_module[block_name + '/' + sub_name] = sub_module
        return new_block_module



class PartAssembler:
    def create_model_def(self):
        model_def = ModuleParts.get_cnn((3, 320, 320))
        model_def = self.fill_modules_def(model_def)
        return model_def

    def fill_modules_def(self, adding_modules, new_modules=None):
        new_modules = {} if new_modules is None else new_modules
        module_filler = {'image': self.fill_image, 'conv2d': self.fill_conv2d, 'resblock': self.fill_resblock,
                         'maxpool': self.fill_maxpool, 'add': self.fill_add,
                         }
        for name, module_def in adding_modules.items():
            new_modules = module_filler[module_def['module']](module_def, new_modules, name)
        return new_modules

    def fill_image(self, cur_module, building_modules, cur_name):
        building_modules[cur_name] = cur_module
        return building_modules

    def fill_resblock(self, cur_module, building_modules, cur_name):
        bef_module = building_modules[cur_module['input']]
        in_channels = bef_module['out_channels']
        block_modules = ModuleParts.get_resblock(cur_name, cur_module['input'], in_channels)
        print("block modules", block_modules)
        block_modules = self.fill_modules_def(block_modules, building_modules)
        return block_modules

    def fill_conv2d(self, cur_module, building_modules, cur_name):
        """
        fill in new attributes: ['in_channels', 'in_resol', 'out_resol', 'padding', 'padding_mode']
        """
        bef_module = building_modules[cur_module['input']]
        cur_module['in_channels'] = bef_module['out_channels']
        cur_module['in_resol'] = bef_module['out_resol']
        cur_module['padding'] = cur_module['kernel_size'] // 2
        cur_module['padding_mode'] = 'zeros'
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
        bef_module = building_modules[cur_module['input'][0]]
        cur_module['in_channels'] = bef_module['out_channels']
        cur_module['out_channels'] = cur_module['in_channels']
        cur_module['in_resol'] = bef_module['out_resol']
        cur_module['out_resol'] = cur_module['in_resol']
        building_modules[cur_name] = cur_module
        return building_modules


if __name__ == "__main__":
    model_def = PartAssembler().create_model_def()
    pp = PrettyPrinter(sort_dicts=False)
    pp.pprint(model_def)
    print(list(model_def))
