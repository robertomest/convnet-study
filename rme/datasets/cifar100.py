from __future__ import absolute_import

import os
import glob
import pickle
import numpy as np

from .preprocessing import one_hotify

def load(data_dir, valid_ratio=0.0, shuffle=False, dtype='float32'):
    assert valid_ratio < 1 and valid_ratio >= 0, 'valid_ratio must be in [0, 1)'
    with open(os.path.join(data_dir, 'train'), 'rb') as f:
        train = pickle.load(f)

    data = train['data']
    labels = one_hotify(train['fine_labels'], nb_classes=100)

    N = data.shape[0]

    if shuffle:
        new_order = np.random.permutation(np.arange(N))
        data = data[new_order]
        labels = labels[new_order]

    # Samples on the training set
    M = int((1 - valid_ratio)*N)
    train_set, valid_set = {}, {}

    train_set['data'] = data[:M].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(dtype)
    train_set['labels'] = labels[:M]

    valid_set['data'] = data[M:].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(dtype)
    valid_set['labels'] = labels[M:]

    with open(os.path.join(data_dir, 'test'), 'rb') as f:
        test = pickle.load(f)

    test_set = {}
    test_set['data'] = test['data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(dtype)
    test_set['labels'] = one_hotify(test['fine_labels'], nb_classes=100)

    return train_set, valid_set, test_set


def preprocess(dataset):
    mean = np.array([129.3, 124.1, 112.4])
    std = np.array([68.2, 65.4, 70.4])

    dataset -= mean
    dataset /= std

    return dataset
