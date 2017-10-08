"""
This module contains data preprocessor and pipelined data feeder
"""

from __future__ import print_function

import io
import bson
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
import multiprocessing as mp

IMAGE_SHAPE = (180, 180, 3)

_img_dir = None
_sess = None
_coord = tf.train.Coordinator()
_id_queue = tf.RandomShuffleQueue(
    10000,                                  # capacity
    0,                                      # min after dequeue
    dtypes=[tf.string],                     # only string
    shapes=[()])                            # 1d elements
_data_queue = tf.FIFOQueue(
    500,                                    # capacity of 500
    dtypes=(
        tf.float32,                         # type of data(image)
        tf.int32                            # type of label(index)
    ),
    shapes=(
        IMAGE_SHAPE,
        ()                                  # single integer
    ))

def _preprocess(bson_path, image_dir):
    pass

def check_preprocess(bson_path, image_dir):
    """
    initializes the package level variables
    performs preprocessing if needed

    Return: None
    """
    pass

def _load_worker():
    pass

def start(sess):
    """
    start the daemon of workers

    Return: None
    """
    pass

def end():
    """
    Shutdown all the loaders

    Return: None
    """
    pass
