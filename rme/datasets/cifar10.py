from __future__ import absolute_import

import os
import glob
import pickle
import numpy as np

from .utils import global_contrast_normalization, zca_whitening, one_hotify

def load(data_dir, valid_ratio=0.0, one_hot=True, shuffle=False, gcn=True,
         zca=True, dtype='float32'):
  """
  Loads CIFAR-10 pickled batch files, given the files' directory.
  Optionally shuffles samples before dividing training and validation sets.
  Can also apply global contrast normalization and ZCA whitening.

  Arguments:
    data_dir: pickled batch files directory.
    valid_ratio: how much of the training data to hold for validation.
      Default: 0.
    one_hot: if True returns one-hot encoded labels, otherwise, returns
    integers. Default: True.
    shuffle: if True shuffles the data before splitting validation data.
      Default: False.
    gcn: if True applies global constrast normalization. Default: False.
    zca: if True applies ZCA whitening. Default: False.
    dtype: data type of image ndarray. Default: `float32`.

  Returns:
    train_set: dict containing training data with keys `data` and `labels`.
    valid_set: dict containing validation data with keys `data` and `labels`.
    test_set: dict containing test data with keys `data` and `labels`.
    If zca == True, also returns
      mean: the computed mean values for each input dimension.
      whitening: the computed ZCA whitening matrix.
      For more information please see datasets.utils.zca_whitening.
   """
  assert valid_ratio < 1 and valid_ratio >= 0, 'valid_ratio must be in [0, 1)'
  files = glob.glob(os.path.join(data_dir, 'data_batch_*'))
  assert len(files) == 5, 'Could not find files!'
  files = [os.path.join(data_dir, 'data_batch_%d' %(i+1)) for i in range(5)]
  data_set = None
  labels = None
  # Iterate over the batches
  for f_name in files:
    with open(f_name, 'rb') as f:
      # Get batch data
      batch_dict = pickle.load(f)
    if data_set is None:
      # Initialize the dataset
      data_set = batch_dict['data'].astype(dtype)
    else:
      # Stack all batches together
      data_set = np.vstack((data_set, batch_dict['data'].astype(dtype)))

    # Get the labels
    # If one_hot, transform all integer labels to one hot vectors
    if one_hot:
      batch_labels = one_hotify(batch_dict['labels'])
    else:
      # If not, just return the labels as integers
      batch_labels = np.array(batch_dict['labels'])
    if labels is None:
      # Initalize labels
      labels = batch_labels
    else:
      # Stack labels together
      labels = np.vstack((labels, batch_labels))

  N = data_set.shape[0]
  if shuffle:
    # Shuffle and separate between training and validation set
    new_order = np.random.permutation(np.arange(N))
    data_set = data_set[new_order]
    labels = labels[new_order]

  # Get the number of samples on the training set
  M = int((1 - valid_ratio)*N)
  # Divide the samples
  train_set, valid_set = {}, {}
  # Reassing the data and reshape it as images
  train_set['data'] = data_set[:M].reshape(
                      (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
  #train_set['data'] = data_set[:M]
  train_set['labels'] = labels[:M]
  valid_set['data'] = data_set[M:].reshape(
                      (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
  valid_set['labels'] = labels[M:]

  test_set = {}
  # Get the test set
  f_name = os.path.join(data_dir, 'test_batch')
  with open(f_name, 'rb') as f:
    batch_dict = pickle.load(f)
    test_set['data'] = batch_dict['data'].astype(dtype).reshape(
                       (-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    if one_hot:
      test_set['labels'] = one_hotify(batch_dict['labels'])
    else:
      test_set['labels'] = np.array(batch_dict['labels'])

  # Do some postprocessing
  if gcn:
    # Do global contrast normalization
    train_set['data'] = global_contrast_normalization(train_set['data'])
    valid_set['data'] = global_contrast_normalization(valid_set['data'])
    test_set['data'] = global_contrast_normalization(test_set['data'])

  if zca:
    train_set['data'], mean, whitening = zca_whitening(train_set['data'])
    valid_set['data'], _, _ = zca_whitening(valid_set['data'], mean=mean,
                                            whitening=whitening)
    test_set['data'], _, _ = zca_whitening(test_set['data'], mean=mean,
                                           whitening=whitening)
    return train_set, valid_set, test_set, mean, whitening

  return train_set, valid_set, test_set
