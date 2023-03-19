
class ModuleDefBase:
    INPUT_NAMES = ['input0', 'input1', 'input2']
    CLASS_COUNT = {}
    NAMESPACE = ''

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
        x = [prior_outputs[src] for src in src_names]
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

    def make_sure_have_name(self):
        class_name = self.NAMESPACE + self.__class__.__name__
        if class_name not in self.CLASS_COUNT:
            self.CLASS_COUNT[class_name] = 0
        self.CLASS_COUNT[class_name] += 1
        if 'name' in self and self['name'] is not None:
            return
        name_prefix = self['alias'] if 'alias' in self else class_name
        self['name'] = f"{name_prefix}{self.CLASS_COUNT[class_name]}"


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
            if module_def.__class__.__name__ == 'Input':
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

