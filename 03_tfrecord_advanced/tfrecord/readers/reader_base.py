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


class DataReaderBase:
    def __init__(self, drive_path, split, dataset_cfg):
        self.frame_names = self.init_drive(drive_path, split)
        self.dataset_cfg = dataset_cfg

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path, split):
        """
        :param drive_path: path to data of a drive
        :param split: train/val/test
        reset variables for a new sequence like intrinsic, extrinsic, and last index
        """
        raise NotImplementedError()

    def num_frames(self):
        return len(self.frame_names)

    def get_image(self, index):
        """
        :return: 'undistorted' indexed image in the current sequence
        """
        raise NotImplementedError()

    def get_bboxes(self, index):
        """
        :return: bounding box (x1, y1, x2, y2, category_index)
        """
        raise NotImplementedError()

    def map_category(self, srclabel):
        """
        :return: convert srclabel to category index
        """
        raise NotImplementedError()

