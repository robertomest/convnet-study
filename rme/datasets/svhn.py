from __future__ import absolute_import

import os
import scipy.io as sio
import numpy as np

from .preprocessing import one_hotify

def load(data_dir, shuffle=True):

    train_set, valid_set, test_set = {}, {}, {}

    # Loading training set
    Tr = sio.loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    Te = sio.loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    Ex = sio.loadmat(os.path.join(data_dir, 'extra_32x32.mat'))

    # Transpose data to TF format, adjust the label values to range (0,9)
    # and get the labels as a 1D vector.
    for dataset in [Tr, Te, Ex]:
        dataset['X'] = dataset['X'].transpose((3, 0 , 1, 2))
        dataset['y'] = np.squeeze(dataset['y'])
        # 0 is represented as 10, make it 0.
        dataset['y'][dataset['y'] == 10] = 0
    # How many samples we'll get from training and extra sets from each class
    valid_train = 400
    valid_extra = 200


    N_tr = Tr['X'].shape[0]
    N_ex = Ex['X'].shape[0]

    if shuffle:
        idx_tr = np.random.permutation(np.arange(N_tr))
        idx_ex = np.random.permutation(np.arange(N_ex))

        Tr['X'] = Tr['X'][idx_tr]
        Tr['y'] = Tr['y'][idx_tr]
        Ex['X'] = Ex['X'][idx_ex]
        Ex['y'] = Ex['y'][idx_ex]

    for i in range(10): # Go through every class
        for dataset, n in zip([Tr, Ex], [valid_train, valid_extra]):
            # Get indices of that class
            idx = np.where(dataset['y'] == i)[0]
            if valid_set.get('labels') is None: # First time
                valid_set['labels'] = dataset['y'][idx][:n]
                valid_set['data'] = dataset['X'][idx][:n]
                train_set['labels'] = dataset['y'][idx][n:]
                train_set['data'] = dataset['X'][idx][n:]
            else:
                # First n goes to validation set
                valid_set['labels'] = np.concatenate((valid_set['labels'],
                                                      dataset['y'][idx][:n]))
                valid_set['data'] = np.vstack((valid_set['data'],
                                               dataset['X'][idx][:n]))
                # Rest goes to training set
                train_set['labels'] = np.concatenate((train_set['labels'],
                                                      dataset['y'][idx][n:]))
                train_set['data'] = np.vstack((train_set['data'],
                                               dataset['X'][idx][n:]))

    test_set['data'] = Te['X']
    test_set['labels'] = Te['y']

    for dataset in [train_set, valid_set, test_set]:
        perm = np.random.permutation(np.arange(dataset['data'].shape[0]))
        dataset['data'] = (dataset['data'][perm]).astype('float32')
        dataset['labels'] = one_hotify(dataset['labels'][perm], nb_classes=10)

    return train_set, valid_set, test_set

def preprocess(dataset):
    mean = np.array([109.9, 109.7, 113.8])
    std = np.array([50.1, 50.6, 50.9])

    dataset -= mean
    dataset /= std

    return dataset
