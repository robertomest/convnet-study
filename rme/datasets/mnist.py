from __future__ import absolute_import

import os
import numpy as np
import gzip
import struct

from .utils import one_hotify, normalization

def load(data_dir, valid_ratio=0.0, one_hot=True, shuffle=False,
         normalize=True, dtype='float32'):

  train_set, valid_set, test_set = {}, {}, {}
  # Get data from binary files
  for img_set, file_name in zip((train_set, test_set), ('train', 't10k')):
    # Load images
    img_path = os.path.join(data_dir, file_name + '-images-idx3-ubyte.gz')
    with gzip.open(img_path, 'rb') as f:
      magic_num, num_imgs, num_rows, num_cols = struct.unpack('>iiii',
                                                              f.read(16))
      shape = (num_imgs, num_rows, num_cols, 1)
      img_set['data'] = np.fromstring(f.read(),
                        dtype='uint8').astype(dtype).reshape(shape)

    # Load labels
    label_path = os.path.join(data_dir, file_name + '-labels-idx1-ubyte.gz')
    with gzip.open(label_path, 'rb') as f:
      magic_num, num_labels = struct.unpack('>ii', f.read(8))
      img_set['labels'] = np.fromstring(f.read(),
                          dtype='uint8').astype('int')
      if one_hot:
        img_set['labels'] = one_hotify(img_set['labels'])

  N = train_set['data'].shape[0]
  if shuffle:
    # Shuffle and separate between training and validation set
    new_order = np.random.permutation(np.arange(N))
    train_set['data'] = train_set['data'][new_order]
    train_set['labels'] = train_set['labels'][new_order]

  # Get the number of samples on the training set
  M = int((1 - valid_ratio)*N)
  # Separate validation set
  valid_set['data'] = train_set['data'][M:]
  valid_set['labels'] = train_set['labels'][M:]
  train_set['data'] = train_set['data'][:M]
  train_set['labels'] = train_set['labels'][:M]

  if normalize:
    # Normalize data
    train_set['data'], mean, std = normalization(train_set['data'])
    valid_set['data'], _, _ = normalization(valid_set['data'], mean, std)
    test_set['data'], _, _ = normalization(test_set['data'], mean, std)

    return train_set, valid_set, test_set, mean, std

  return train_set, valid_set, test_set
