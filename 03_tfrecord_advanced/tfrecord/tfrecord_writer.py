import os
import os.path as op
import tensorflow as tf
import numpy as np
import json
import copy
from glob import glob
import shutil
from timeit import default_timer as timer

import util_class as uc
import util_function as uf
from tfrecord.example_maker import ExampleMaker
import tfrecord.tfr_util as tu


def drive_manager_factory(dataset, split, srcpath):
    if dataset == "kitti_raw":
        from tfrecord.readers.kitti_reader import KittiDriveManager
        return KittiDriveManager(srcpath, split)
    else:
        assert 0, f"[drive_manager_factory] invalid dataset name: {dataset}"


def drive_reader_factory(dataset, split, drive_path):
    if dataset == "kitti_raw":
        from tfrecord.readers.kitti_reader import KittiReader
        return KittiReader(drive_path, split)
    else:
        assert 0, f"[drive_reader_factory] invalid dataset name: {dataset}"


class TfrecordMaker:
    """
    create tfrecord files divided in shards for a single dataset and a single split
    get raw examples from ExampleMaker and convert them into tf.data.Example
    serialize examples and write serialized data into tfrecord files
    """
    def __init__(self, dataset, split, srcpath, tfrpath, hwc_shape, shard_size,
                 drive_example_limit, total_example_limit):
        self.dataset = dataset              # dataset name e.g. "kitti"
        self.split = split                  # split name e.g. "train", "val", "test
        self.srcpath = srcpath              # raw data path
        self.tfrpath__ = tfrpath + "__"     # temporary path to write tfrecord
        self.tfrpath = tfrpath              # path to save final tfrecord
        self.tfr_drive_path = ""            # path to write current drive's tfrecord
        self.hwc_shape = hwc_shape
        self.shard_size = shard_size        # max number of examples in a shard
        self.drive_example_limit = drive_example_limit
        self.total_example_limit = total_example_limit
        self.shard_count = 0                # number of shards written in this drive
        self.shard_example_count = 0        # number of examples in this shard
        self.drive_example_count = 0        # number of examples in this drive
        self.total_example_count = 0        # number of examples in this dataset
        self.drive_manger = drive_manager_factory(dataset, split, srcpath)
        self.serializer = tu.TfrSerializer()
        self.writer = None
        self.path_manager = uc.PathManager([""])
        self.error_count = 0

    def make(self):
        print("\n\n========== Start a new dataset:", op.basename(self.tfrpath))
        drive_paths = self.drive_manger.get_drive_paths()
        with uc.PathManager(self.tfrpath__, closer_func=self.on_exit) as path_manager:
            self.path_manager = path_manager
            for drive_index, drive_path in enumerate(drive_paths):
                # skip if drive_path has been converted to tfrecord
                if self.init_drive_tfrecord(drive_index):
                    continue
                # stop if number of total frame exceeds the limit
                if (self.total_example_limit > 0) and (self.total_example_count >= self.total_example_limit):
                    break

                print("\n==== Start a new drive:", drive_path)
                drive_example = self.write_drive(drive_index, self.hwc_shape)
                self.write_tfrecord_config(drive_example)

            path_manager.set_ok()
        self.wrap_up()

    def init_drive_tfrecord(self, drive_index=0):
        drive_name = self.drive_manger.get_drive_name(drive_index)
        outpath = op.join(self.tfrpath__, drive_name)
        print("[init_drive_tfrecord] tfrecord drive path:", outpath)
        if op.isdir(outpath):
            print(f"[init_drive_tfrecord] {op.basename(outpath)} exists. move onto the next")
            return True

        # change path to check date integrity
        self.path_manager.reopen(outpath, closer_func=self.on_exit)
        self.tfr_drive_path = outpath
        self.shard_count = 0
        self.shard_example_count = 0
        self.drive_example_count = 0
        self.open_new_writer(drive_index)
        return False

    def open_new_writer(self, drive_index):
        drive_name = self.drive_manger.get_drive_name(drive_index)
        outfile = f"{self.tfr_drive_path}/{drive_name}_shard_{self.shard_count:03d}.tfrecord"
        self.writer = tf.io.TFRecordWriter(outfile)

    def write_drive(self, drive_index, hwc_shape):
        drive_paths = self.drive_manger.get_drive_paths()
        num_drives = len(drive_paths)
        data_reader = drive_reader_factory(self.dataset, self.split, drive_paths[drive_index])
        example_maker = ExampleMaker(data_reader, hwc_shape)
        loop_range = example_maker.get_range()
        num_frames = len(loop_range)
        drive_example = ()

        for ex_index, ex_id in enumerate(loop_range):
            time1 = timer()
            if (self.drive_example_limit > 0) and (self.drive_example_count >= self.drive_example_limit):
                break
            if (self.total_example_limit > 0) and (self.total_example_count >= self.total_example_limit):
                break

            try:
                example = example_maker.get_example(ex_id)
                drive_example = self.verify_example(drive_example, example)
            except StopIteration as si:  # raised from xxx_reader._get_frame()
                print("\n==[write_drive][StopIteration] stop this drive", si)
                break
            except uc.MyExceptionToCatch as me:  # raised from xxx_reader._get_frame()
                uf.print_progress(f"==[write_drive][MyException] {ex_index}/{num_frames}, {me}")
                continue

            serialized_example = self.serializer(example)
            self.write_example(serialized_example, drive_index)
            uf.print_progress(f"==[write_drive] shard/drive/ndrives: {self.shard_count}/{drive_index}/{num_drives} | "
                              f"in-drive: {ex_index}/{self.drive_example_count}/{num_frames} | "
                              f"in-shard: {self.shard_example_count}/{self.shard_size} | "
                              f"total: {self.total_example_count} | "
                              f"time: {timer() - time1:1.4f}")
        print("")
        return drive_example

    def verify_example(self, drive_example, example):
        if (not example) or ("image" not in example):
            raise uc.MyExceptionToCatch(f"[verify_example] EMPTY example")
        # initialize drive example (representative sample among drive examples)
        if not drive_example:
            drive_example = copy.deepcopy(example)
            print("[verify_example] initialize drive_example:", list(drive_example.keys()))
            return drive_example
        # check key change
        if list(drive_example.keys()) != list(example.keys()):
            self.error_count += 1
            assert self.error_count < 10, "too frequent errors"
            raise uc.MyExceptionToCatch(f"[verify_example] error count: {self.error_count}, different keys:\n"
                                        f"{list(drive_example.keys())} != {list(example.keys())}")
        # check shape change
        for key in drive_example:
            if not isinstance(drive_example[key], np.ndarray):
                continue
            if drive_example[key].shape != example[key].shape:
                self.error_count += 1
                assert self.error_count < 10, "too frequent errors"
                raise uc.MyExceptionToCatch(f"[verify_example] error count: {self.error_count}, "
                      f"different shape of {key}: {drive_example[key].get_shape()} != {example[key].get_shape()}")
        return drive_example

    def write_example(self, example_serial, drive_index):
        self.writer.write(example_serial)
        self.shard_example_count += 1
        self.drive_example_count += 1
        self.total_example_count += 1
        # reset and create a new tfrecord file
        if self.shard_example_count > self.shard_size:
            self.shard_count += 1
            self.shard_example_count = 0
            self.open_new_writer(drive_index)

    def write_tfrecord_config(self, example):
        if self.drive_example_count == 0:
            return
        assert ('image' in example) and (example['image'] is not None)
        config = tu.inspect_properties(example)
        config["length"] = self.drive_example_count
        config["imshape"] = self.hwc_shape
        print("## save config", config)
        with open(op.join(self.tfr_drive_path, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

    def on_exit(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def wrap_up(self):
        tfrpath__ = self.tfrpath__
        files = glob(f"{tfrpath__}/*/*.tfrecord")
        print("[wrap_up] move tfrecords:", files[0:-1:5])
        for file in files:
            shutil.move(file, op.join(tfrpath__, op.basename(file)))

        # merge config files of all drives and save only one in tfrpath
        files = glob(f"{tfrpath__}/*/tfr_config.txt")
        print("[wrap_up] config files:", files[:5])
        total_length = 0
        config = dict()
        for file in files:
            with open(file, 'r') as fp:
                config = json.load(fp)
                total_length += config["length"]
        config["length"] = total_length
        print("[wrap_up] final config:", config)
        with open(op.join(tfrpath__, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

        os.rename(tfrpath__, self.tfrpath)
