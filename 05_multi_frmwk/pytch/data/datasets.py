import os.path as op
import torch
import torchvision
import numpy as np

import neutr.data.readers.kitti_reader as kitti
import neutr.utils.util_function as nuf


class Kitti2dDetectDataset:
    def __init__(self, dset_cfg, split, transform=None):
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



class Cifar10Dataset:
    def __init__(self, split, batch_size, shuffle):
        self.split = split
        self.batch_size = batch_size
        self.dataset = self.get_dataset(split)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def get_dataset(self, split):
        train = True if split == 'train' else False
        cifar = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # pytorch uses image shape like (batch, channel, height, width)
        x = np.transpose(np.array(cifar.data), (0, 3, 1, 2))
        y = np.array(cifar.targets)
        x = nuf.to_float_image(x)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y)
        dataset = torch.utils.data.TensorDataset(x, y)
        return dataset

    def get_dataloader(self):
        steps = len(self.dataset // self.batch_size)
        x, y = self.dataset[0]
        imshape = x.size()[1:]
        return self.dataloader, steps, imshape
