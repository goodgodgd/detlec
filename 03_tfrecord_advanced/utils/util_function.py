import sys
import tensorflow as tf
import numpy as np


def print_progress(status_msg):
    # Note the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32) * 2 - 1


def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype((im_tensor + 1.) / 2., dtype=tf.uint8)


def convert_box_format_2pt_to_yxhw(boxes):
    # boxes: [y1, x1, y2, x2, category] -> [cy, cx, h, w, category]
    new_boxes = boxes.copy()
    new_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.  # center y
    new_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.  # center x
    new_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]         # height
    new_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]         # width
    return new_boxes.astype(boxes.dtype)


def convert_box_format_yxhw_to_2pt(boxes):
    # boxes: [cy, cx, h, w, category] -> [y1, x1, y2, x2, category]
    new_boxes = boxes.copy()
    new_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.    # y1 = cy - h/2
    new_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.    # x1 = cx - w/2
    new_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.    # y2 = cy + h/2
    new_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.    # x2 = cx + w/2
    return new_boxes.astype(boxes.dtype)





