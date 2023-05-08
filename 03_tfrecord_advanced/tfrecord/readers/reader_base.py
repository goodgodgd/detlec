
class DatasetReaderBase:
    def __init__(self, data_path, split, dataset_cfg):
        self.frame_names = self.init_frames(data_path, split)
        self.dataset_cfg = dataset_cfg

    def init_frames(self, data_path, split):
        """
        :param data_path: path to the dataset folder
        :param split: train/val/test
        """
        raise NotImplementedError()

    def get_frame_names(self):
        return self.frame_names

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

    def get_categories(self, index):
        """
        :param index: image index in self.frame_names
        :return: category ids
        """
        raise NotImplementedError()
