"""
This module contains data preprocessor and pipelined data feeder
"""

from __future__ import print_function

import io
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
import multiprocessing as mp
import os.path

IMAGE_SHAPE = (180, 180, 3)
NCORE = 8

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

_threads = []

def _preprocess(bson_path, image_dir):

    def worker(queue, iolock):
        while True:
            d = q.get()
            if d is None:
                break
            product_id = d["_id"]
            category_id = d["category_id"]
            product = [category_id]
            for e, pic in enumerate(d["imgs"]):
                picture = imread(io.BytesIO(pic["picture"]))
                product.append(picture)
            path = os.path.join(image_dir, str(product_id) + ".p")
            with open(path, "wb") as f:
                pickle.dump(product, f)

    q = mp.Queue(maxsize=NCORE)
    iolock = mp.Lock()
    pool = mp.Pool(NCORE, initializer=worker, initargs=(q, iolock))
    data = bson.decode_file_iter(open(bson_path, "rb"))
    for c, d in enumerate(data):
        q.put(d)

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
    if _sess is None:
        return
    _coord.request_stop()
    dq_stop = _data_queue.close(cancel_pending_enqueues=True)
    _sess.run(dq_stop)
    _coord.join(_threads)

