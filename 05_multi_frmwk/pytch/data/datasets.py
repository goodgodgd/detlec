import os.path as op
import torch
import numpy as np

import neutr.data.readers.kitti_reader as kitti


class Kitti2dDetectDataset:
    def __init__(self, dset_cfg, split, transform):
        self.transform = transform
        drive_path = op.join(dset_cfg.PATH, 'training', 'image_2')
        self.data_reader = kitti.KittiReader(drive_path, split, dset_cfg)

    def __len__(self):
        return len(self.data_reader)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        example = dict()
        example["image"] = self.data_reader.get_image(index)
        bbox = self.data_reader.get_bboxes(index)
        category = self.data_reader.get_categories(index)
        example["inst"] = np.concatenate([bbox, np.ones_like(category), category], axis=-1, dtype=np.float32)

        if self.transform:
            example = self.transform(example)
        return example
