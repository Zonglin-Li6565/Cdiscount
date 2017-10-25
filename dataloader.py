"""
This module contains data preprocessor and pipelined data feeder
"""

from __future__ import print_function

import io
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
import multiprocessing as mp
import os.path
import struct
import bson
from bson.errors import InvalidBSON

IMAGE_SHAPE = (180, 180, 3)
NCORE = 8

_bson_path = None
_sess = None
_coord = tf.train.Coordinator()

_file_offset_queue = tf.RandomShuffleQueue(
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

_threads = []
_data_starts = None

def _preprocess(bson_path):
    """
    adapted from
    https://github.com/mongodb/mongo-python-driver/blob/0b34f9702ca8bed45792a53287d33a2292b99152/bson/__init__.py#L855

    Returns: a list of integers, specifies the start of each data point in the
             file
    """
    data_starts = []
    current_offset = 0
    with open(bson_path, "rb") as f:
        while True:
            size_data = f.read(4)
            data_starts.append(current_offset)
            if len(size_data) is 0:
                break
            elif len(size_data) != 4:
                raise InvalidBSON("cut off in middle of objsize")
            size = struct.Struct("<i").unpack(size_data)[0]
            f.seek(size - 4, os.SEEK_CUR)
            current_offset += size
    _bson_path = bson_path
    return data_starts

def init(bson_path, cache_file):
    """
    initializes the package level variables
    performs preprocessing if needed

    Return: None
    """
    global _data_starts
    if os.path.isfile(cache_file) is False:
        _data_starts = _preprocess(bson_path)
        with open(cache_file, "wb") as f:
            pickle.dump(_data_starts, f)
    else:
        with open(cache_file, "rb") as f:
            _data_starts = pickle.load(f)

def _load_worker(offset_dequeue, data_enqueue):
    while _coord.should_stop() is False:
        # dequeue one from offset queue
        offset = _sess.run(offset_dequeue)
        with open(_bson_path, "rb") as f:
            f.seek(offset, os.SEEK_SET)
            _, data = bson.decode_file_iter(f).next()
            product_id = data["_id"]
            category_id = data["category_id"]
            for e, pic in enumerate(data["imgs"]):
                picture = imread(io.BytesIO(pic["picture"]))
                ## TODO: enqueue the image

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
    if _sess is None:
        return
    _coord.request_stop()
    dq_stop = _data_queue.close(cancel_pending_enqueues=True)
    _sess.run(dq_stop)
    _coord.join(_threads)

