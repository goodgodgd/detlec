import os.path as op
import numpy as np
from glob import glob
import cv2

from neutr.data.readers.reader_base import DatasetReaderBase, DriveManagerBase
import neutr.utils.util_function as nuf
import neutr.utils.util_class as uc


class KittiDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        kitti_split = "training"    # if self.split == "train" else "testing"
        return [op.join(self.datapath, kitti_split, "image_2")]

    def get_drive_name(self, drive_index):
        return f"kitti{drive_index:02d}"


class KittiReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)
        self.cur_frame = -1
        self.bbox = None
        self.category = None

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.png"))
        frame_names.sort()
        if split == "train":
            frame_names = frame_names[:-500]
        else:
            frame_names = frame_names[-500:]
        print("[KittiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index):
        """
        :return: bounding boxes in 'yxhw' format
        """
        self.read_text(index)
        return self.bbox

    def get_categories(self, index):
        self.read_text(index)
        return self.category

    def read_text(self, index):
        if self.cur_frame == index:
            return
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            bboxes = [self.read_line(line) for line in lines]
            bboxes = [bbox for bbox in bboxes if bbox is not None]
        if not bboxes:
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes = np.array(bboxes)
        self.bbox = bboxes[:, :-1]
        self.category = bboxes[:, -1:]
        self.cur_frame = index
        return bboxes

    def read_line(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None
        category_index = self.dataset_cfg.CATEGORIES_TO_USE.index(category_name)
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))
        bbox = np.array([(y1+y2)/2, (x1+x2)/2, y2-y1, x2-x1, category_index], dtype=np.int32)
        return bbox


# ==================================================
import config as cfg


def test_kitti_reader():
    print("===== start test_kitti_reader")
    dataset_cfg = cfg.Datasets.Kitti
    drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i)
        ctgrs = reader.get_categories(i)
        inst = np.concatenate([bboxes, np.ones_like(ctgrs), ctgrs], axis=1)
        print(f"frame {i}, instances:\n", inst)
        boxed_image = nuf.draw_boxes(image, inst, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("kitti", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_kitti_reader passed")


if __name__ == "__main__":
    test_kitti_reader()
