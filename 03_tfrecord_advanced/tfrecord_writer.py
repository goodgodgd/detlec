import os
import os.path as op
from glob import glob
import tensorflow as tf
import shutil
import json
import copy
from timeit import default_timer as timer

import utils as ut
from tfrecords.example_maker import ExampleMaker
from tfrecords.tfr_util import Serializer, inspect_properties
from utils.util_class import MyExceptionToCatch


def tfrecord_maker_factory(dataset, split, srcpath, tfrpath, dstshape):
    if dataset == "kitti_raw":
        return KittiTfrecordMaker(dataset, split, srcpath, tfrpath, 2000, dstshape)
    else:
        assert 0, f"Invalid dataset: {dataset}"

