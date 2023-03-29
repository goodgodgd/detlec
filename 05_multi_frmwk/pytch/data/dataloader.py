import torch
import numpy as np
import torch.utils.data as td
import torchvision as tv

import neutr.data.preprocess as pp
import pytch.data.datasets as pds
import pytch.utils.util_function as puf
import config as cfg


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, example):
        for key, val in example.items():
            if isinstance(val, np.ndarray):
                device = puf.device()
                if key == 'category':
                    val = torch.from_numpy(val).long().to(device)
                else:
                    val = torch.from_numpy(val).float().to(device)
                example[key] = val
        # image: H x W x C -> C x H x W
        example['image'] = example['image'].permute(2, 0, 1)
        return example


def make_dataloader(data_cfg, split, batch_size=4, shuffle=False, num_workers=0):
    simple_datasets = {'cifar10': pds.Cifar10Dataset}
    if data_cfg.NAME in simple_datasets:
        dataset_class = simple_datasets[data_cfg.NAME]
        return dataset_class(data_cfg.PATH, split, batch_size, shuffle, num_workers).get_dataloader()

    transform = get_transform(data_cfg)
    dataset = get_dataset(data_cfg, split, transform)
    dataloader = td.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=torch.Generator(device=puf.device()))

    steps = len(dataset) // batch_size
    x = dataset[0]
    imshape = x['image'].shape
    return dataloader, steps, imshape


def get_transform(dset_cfg):
    if dset_cfg.NAME == 'kitti':
        target_hw = dset_cfg.INPUT_RESOLUTION
        trfm = tv.transforms.Compose([pp.ExampleCropper(target_hw, dset_cfg.CROP_TLBR),
                                      pp.ExampleResizer(target_hw),   # box in pixel scale
                                      pp.ExampleBoxScaler(),          # box in (0~1) scale
                                      pp.ExampleCategoryRemapper(dset_cfg.CATEGORIES_TO_USE,
                                                       dset_cfg.CATEGORY_REMAP,
                                                       cfg.DataCommon.CATEGORY_NAMES),
                                      pp.ExampleZeroPadBbox(cfg.DataCommon.MAX_BBOX_PER_IMAGE),
                                      ToTensor()
                                     ])
    else:
        raise ValueError(f"No dataset named {dset_cfg.NAME} is prepared")
    return trfm


def get_dataset(dset_cfg, split, transform):
    if dset_cfg.NAME == 'kitti':
        dataset = pds.Kitti2dDetectDataset(dset_cfg, split, transform)
    else:
        raise ValueError(f"No dataset named {dset_cfg.NAME} is prepared")
    return dataset


# ==========
import cv2
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
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataloader, steps, imshape = make_dataloader(cfg.Datasets.Kitti, "train", batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        data = puf.convert_to_numpy(data)
        image = data['image'][0]
        inst = data['inst'][0]
        # image = image.astype(np.uint8)
        image = nuf.draw_boxes(image, inst, cfg.Datasets.Kitti.CATEGORIES_TO_USE)
        print("frame", i, image.shape, image.dtype, "inst", inst.shape)
        cv2.imshow("image", image)
        key = cv2.waitKey()
        if key == ord('q'):
            break


if __name__ == "__main__":
    # test_dataset()
    test_dataloader()
