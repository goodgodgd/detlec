import os.path as op
import torch
import numpy as np
import torch.utils.data as td
import torchvision as tv
import cv2

import neutr.data.readers.kitti_reader as kitti
import neutr.data.preprocess as pp
import config as cfg


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


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, example):
        for key, val in example.items():
            if isinstance(val, np.ndarray):
                example[key] = torch.from_numpy(val)
        # numpy image: H x W x C
        # torch image: C x H x W
        example['image'] = example['image'].permute(2, 0, 1)
        return example

def make_dataloader(dset_cfg, split, batch_size=4, shuffle=False, num_workers=0):
    transform = get_transform(dset_cfg)
    dataset = get_dataset(dset_cfg, split, transform)
    dataloader = td.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def get_transform(dset_cfg):
    target_hw = dset_cfg.INPUT_RESOLUTION
    trfm = tv.transforms.Compose([pp.ExampleCropper(target_hw, dset_cfg.CROP_TLBR),
                                  pp.ExampleResizer(target_hw),   # box in pixel scale
                                  pp.ExampleBoxScaler(),          # box in (0~1) scale
                                  pp.ExampleCategoryRemapper(dset_cfg.CATEGORIES_TO_USE,
                                                   dset_cfg.CATEGORY_REMAP,
                                                   cfg.Tfrdata.CATEGORY_NAMES),
                                  pp.ExampleZeroPadBbox(cfg.Tfrdata.MAX_BBOX_PER_IMAGE),
                                  ToTensor()
                                 ])
    return trfm


def get_dataset(dset_cfg, split, transform):
    if dset_cfg.NAME == 'kitti':
        dataset = Kitti2dDetectDataset(dset_cfg, split, transform)
    else:
        raise ValueError(f"No dataset named {dset_cfg.NAME} is prepared")
    return dataset


# ==========
import neutr.utils.util_function as nuf


def test_dataset():
    transform = get_transform(cfg.Datasets.Kitti)
    dataset = get_dataset(cfg.Datasets.Kitti, "train", transform)
    for i in range(len(dataset)):
        data = dataset[i]
        print("frame", i, "inst", data['inst'].size())
        image = data['image'].numpy()
        image = image.transpose((1, 2, 0))
        inst = data['inst'].numpy()
        image = nuf.draw_boxes(image, inst, cfg.Datasets.Kitti.CATEGORIES_TO_USE)
        cv2.imshow("image", image)
        key = cv2.waitKey()
        if key == ord('q'):
            break


def test_dataloader():
    dataloader = make_dataloader(cfg.Datasets.Kitti, "train", batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        print("frame", i, "inst", data['inst'].size())
        image = data['image'][0].numpy()
        image = image.transpose((1, 2, 0))
        inst = data['inst'][0].numpy()
        image = nuf.draw_boxes(image, inst, cfg.Datasets.Kitti.CATEGORIES_TO_USE)
        cv2.imshow("image", image)
        key = cv2.waitKey()
        if key == ord('q'):
            break


if __name__ == "__main__":
    # test_dataset()
    test_dataloader()
