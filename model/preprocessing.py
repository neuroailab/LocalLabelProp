from __future__ import division, print_function
import os, sys
import numpy as np
import tensorflow as tf

from .resnet_th_preprocessing import preprocessing_inst

# This file contains various preprocessing ops for images (typically
# used for data augmentation).

def resnet_train(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=True)


def resnet_validate(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=False)


def _get_resize_scale(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(
            tf.greater(height, width),
            lambda: smallest_side / width,
            lambda: smallest_side / height)
    return scale


def center_crop(img_str, out_height, out_width):
    shape = tf.image.extract_jpeg_shape(image_string)
    # the scaling factor needed to make the smaller side 256
    scale = _get_resize_scale(shape[0], shape[1], 256)
    cp_height = tf.cast(out_height / scale, tf.int32)
    cp_width = tf.cast(out_width / scale, tf.int32)
    cp_begin_x = tf.cast((shape[0] - cp_height) / 2, tf.int32)
    cp_begin_y = tf.cast((shape[1] - cp_width) / 2, tf.int32)
    bbox = tf.stack([cp_begin_x, cp_begin_y,
                     cp_height, cp_width])
    crop_image = tf.image.decode_and_crop_jpeg(
        image_string, bbox, channels=3)
    image = image_resize(crop_image, out_height, out_width)

    image.set_shape([out_height, out_width, 3])
    return image


def rgb_to_gray(flt_image):
    flt_image = tf.cast(flt_image, tf.float32)
    gry_image = flt_image[:,:,0] * 0.299 \
            + flt_image[:,:,1] * 0.587 \
            + flt_image[:,:,2] * 0.114
    gry_image = tf.expand_dims(gry_image, axis=2)
    gry_image = tf.cast(gry_image + EPS, tf.uint8)
    gry_image = tf.cast(gry_image, tf.float32)
    return gry_image
