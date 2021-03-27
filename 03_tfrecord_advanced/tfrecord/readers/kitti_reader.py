import os.path as op
import numpy as np
from glob import glob
import cv2

from tfrecord.readers.reader_base import DataReaderBase, DriveManagerBase
import tfrecord.tfr_util as tu

KITTI_CATEGORIES = {"Pedestrian": 0, "Car": 1, "Van": 2, "Cyclist": 3}


class KittiDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        kitti_split = "training" if self.split == "train" else "testing"
        return [op.join(self.datapath, kitti_split, "image_2")]

    def get_drive_name(self, drive_index):
        return f"train"


class KittiReader(DataReaderBase):
    def __init__(self, drive_path, split):
        super().__init__(drive_path, split)

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.png"))
        frame_names.sort()
        print("[KittiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_bboxes(self, index):
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            bboxes = [self.extract_box(line) for line in lines]
            bboxes = [bbox for bbox in bboxes if bbox is not None]
        bboxes = np.array(bboxes)
        return bboxes

    def extract_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]
        if category_name not in KITTI_CATEGORIES:
            return None
        category_index = KITTI_CATEGORIES[category_name]
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))
        return np.array([y1, x1, y2, x2, category_index], dtype=np.int32)


# ==================================================

def test_kitti_reader():
    print("===== start test_kitti_reader")
    kitti_path = "/media/ian/Ian4T/dataset/kitti_detection"
    drive_mngr = KittiDriveManager(kitti_path, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train")
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i)
        print(f"frame {i}, bboxes:\n", bboxes)
        boxed_image = tu.draw_boxes(image, bboxes)
        cv2.imshow("kitti", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_kitti_reader passed")


if __name__ == "__main__":
    test_kitti_reader()
