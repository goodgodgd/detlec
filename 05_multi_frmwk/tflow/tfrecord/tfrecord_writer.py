import os
import os.path as op
import tensorflow as tf
import numpy as np
import json
import copy
from glob import glob
import shutil
import cv2
from timeit import default_timer as timer

import neutr.utils.util_class as nuc
import neutr.utils.util_function as nuf
from neutr.data.example_maker import ExampleMaker
import tflow.tfrecord.tfr_util as tu


def drive_manager_factory(dataset_name, split, srcpath):
    if dataset_name == "kitti":
        from neutr.data.readers.kitti_reader import KittiDriveManager
        return KittiDriveManager(srcpath, split)
    else:
        assert 0, f"[drive_manager_factory] invalid dataset name: {dataset_name}"


def drive_reader_factory(dataset_cfg, split, drive_path):
    if dataset_cfg.NAME == "kitti":
        from neutr.data.readers.kitti_reader import KittiReader
        return KittiReader(drive_path, split, dataset_cfg)
    else:
        assert 0, f"[drive_reader_factory] invalid dataset name: {dataset_cfg.NAME}"


class TfrecordMaker:
    """
    create tfrecord files divided in shards for a single dataset and a single split.
    get raw examples from ExampleMaker and convert them into tf.data.Example.
    serialize examples and write serialized data into tfrecord files.
    """
    def __init__(self, dataset_cfg, split, tfrpath, shard_size,
                 drive_example_limit=0, total_example_limit=0):
        self.dataset_cfg = dataset_cfg
        self.split = split                  # split name e.g. "train", "val", "test
        self.tfrpath__ = tfrpath + "__"     # temporary path to write tfrecord
        self.tfrpath = tfrpath              # path to save final tfrecord
        self.tfr_drive_path = ""            # path to write current drive's tfrecord
        self.shard_size = shard_size        # max number of examples in a shard
        self.drive_example_limit = drive_example_limit
        self.total_example_limit = total_example_limit
        # indices and counts
        self.drive_index = 0                # frame index in current drive
        self.shard_index = 0                # frame index in current shard
        self.shard_example_count = 0        # number of examples in this shard
        self.drive_example_count = 0        # number of examples in this drive
        self.total_example_count = 0        # number of examples in this dataset
        self.error_count = 0
        # objects
        self.drive_manger = drive_manager_factory(dataset_cfg.NAME, split, dataset_cfg.PATH)
        self.serializer = tu.TfrSerializer()
        self.writer = None
        self.path_manager = nuc.PathManager([""])

    def make(self):
        print("\n\n========== Start dataset:", self.dataset_cfg.NAME)
        drive_paths = self.drive_manger.get_drive_paths()
        with nuc.PathManager(self.tfrpath__, closer_func=self.on_exit) as path_manager:
            self.path_manager = path_manager
            for self.drive_index, drive_path in enumerate(drive_paths):
                print(f"\n==== Start drive-{self.drive_index}:", drive_path)
                # skip if drive_path has been completed
                if self.init_drive_tfrecord():
                    continue
                # stop if number of total frame exceeds the limit
                if (self.total_example_limit > 0) and (self.total_example_count >= self.total_example_limit):
                    break

                drive_example = self.write_drive(drive_path)
                self.write_tfrecord_config(drive_example)

            path_manager.set_ok()
        self.wrap_up()
        cv2.destroyAllWindows()

    def init_drive_tfrecord(self):
        drive_name = self.drive_manger.get_drive_name(self.drive_index)
        tfr_drive_path = op.join(self.tfrpath__, drive_name)
        print("[init_drive_tfrecord] tfrecord drive path:", tfr_drive_path)
        if op.isdir(tfr_drive_path):
            print(f"[init_drive_tfrecord] {op.basename(tfr_drive_path)} exists. move onto the next")
            return True

        self.path_manager.reopen(tfr_drive_path, closer_func=self.on_exit)
        self.tfr_drive_path = tfr_drive_path
        self.shard_index = 0
        self.shard_example_count = 0
        self.drive_example_count = 0
        self.open_new_writer(self.shard_index)
        return False

    def open_new_writer(self, shard_index):
        drive_name = op.basename(self.tfr_drive_path)
        outfile = f"{self.tfr_drive_path}/{drive_name}_shard_{shard_index:03d}.tfrecord"
        print(f"\n==== Start shard-{shard_index}:", outfile)
        self.writer = tf.io.TFRecordWriter(outfile)

    def write_drive(self, drive_path):
        data_reader = drive_reader_factory(self.dataset_cfg, self.split, drive_path)
        example_maker = ExampleMaker(data_reader, self.dataset_cfg)
        num_drive_frames = data_reader.num_frames()
        drive_example = {}

        for index in range(num_drive_frames):
            time1 = timer()
            if (self.drive_example_limit > 0) and (self.drive_example_count >= self.drive_example_limit):
                break
            if (self.total_example_limit > 0) and (self.total_example_count >= self.total_example_limit):
                break

            try:
                example = example_maker.get_example(index)
                drive_example = self.verify_example(drive_example, example)
            except StopIteration as si:     # raised from next()
                print("\n==[write_drive][StopIteration] stop this drive\n", si)
                break
            except nuc.MyExceptionToCatch as me:  # raised from xxx_reader._get_frame()
                nuf.print_progress(f"\n==[write_drive][MyException] {index}/{num_drive_frames}, {me}\n")
                continue

            serialized_example = self.serializer(example)
            self.write_example(serialized_example)
            nuf.print_progress(f"==[write_drive]: shard: {self.shard_index}/{self.shard_example_count}/{self.shard_size} | "
                              f"drive: {self.drive_example_count}/{index}/{num_drive_frames} | "
                              f"total: {self.total_example_count} | time: {timer() - time1:1.4f}")
        print("")
        return drive_example

    def verify_example(self, drive_example, example):
        if (not example) or ("image" not in example):
            raise nuc.MyExceptionToCatch(f"[verify_example] EMPTY example")
        # initialize drive example (representative sample among drive examples)
        if not drive_example:
            drive_example = copy.deepcopy(example)
            print("[verify_example] initialize drive_example:", list(drive_example.keys()))
            self.write_tfrecord_config(drive_example)
            return drive_example
        # check key change
        if list(drive_example.keys()) != list(example.keys()):
            self.error_count += 1
            assert self.error_count < 10, "too frequent errors"
            raise nuc.MyExceptionToCatch(f"[verify_example] error count: {self.error_count}, different keys:\n"
                                        f"{list(drive_example.keys())} != {list(example.keys())}")
        # check shape change
        for key in drive_example:
            if not isinstance(drive_example[key], np.ndarray):
                continue
            if drive_example[key].shape != example[key].shape:
                self.error_count += 1
                assert self.error_count < 10, "too frequent errors"
                raise nuc.MyExceptionToCatch(f"[verify_example] error count: {self.error_count}, "
                      f"different shape of {key}: {drive_example[key].shape} != {example[key].shape}")
        return drive_example

    def write_example(self, example_serial):
        self.writer.write(example_serial)
        self.shard_example_count += 1
        self.drive_example_count += 1
        self.total_example_count += 1
        # reset and create a new tfrecord file
        if self.shard_example_count >= self.shard_size:
            self.shard_index += 1
            self.shard_example_count = 0
            self.open_new_writer(self.shard_index)

    def write_tfrecord_config(self, example):
        assert ('image' in example) and (example['image'] is not None)
        config = tu.inspect_properties(example)
        config["length"] = self.drive_example_count
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
