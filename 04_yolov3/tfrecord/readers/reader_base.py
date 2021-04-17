class DriveManagerBase:
    def __init__(self, datapath, split):
        self.datapath = datapath
        self.split = split
        self.drive_paths = self.list_drive_paths()

    def list_drive_paths(self):
        raise NotImplementedError()

    def get_drive_paths(self):
        return self.drive_paths

    def get_drive_name(self, drive_index):
        raise NotImplementedError()


class DatasetReaderBase:
    def __init__(self, drive_path, split, dataset_cfg):
        self.frame_names = self.init_drive(drive_path, split)
        self.dataset_cfg = dataset_cfg

    def init_drive(self, drive_path, split):
        """
        :param drive_path: path to the specific drive folder
        :param split: train/val/test
        reset variables and list frame files in the drive
        """
        raise NotImplementedError()

    def num_frames(self):
        return len(self.frame_names)

    def get_image(self, index):
        """
        :param index: image index in self.frame_names
        :return: image, np.uint8
        """
        raise NotImplementedError()

    def get_bboxes(self, index):
        """
        :param index: image index in self.frame_names
        :return: bounding box in the indexed image (y, x, h, w, category_index), np.int32
        """
        raise NotImplementedError()
