import os
import glob
import pickle
import numpy as np
import gzip
import struct

def one_hotify(labels, nb_classes=None):
    '''
    Converts integer labels to one-hot vectors.

    Arguments:
        labels: numpy array containing integer labels. The labels must be in
        range [0, num_labels - 1].

    Returns:
        one_hot_labels: numpy array with shape (batch_size, num_labels).
    '''
    size = len(labels)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1

    one_hot_labels = np.zeros((size, nb_classes))
    one_hot_labels[np.arange(size), labels] = 1
    return one_hot_labels

def ops_in_batches(data_set, oplist, session, input_placeholder, labels_placeholder, num_per_batch=1000,
                   feed_dict=None):
    ''' Function that evaluates an operation in the graph in batches. '''
    num_samples = float(data_set['labels'].shape[0])
    num_batches = int(np.ceil(num_samples/num_per_batch))
    # Store the results in a list
    results = []
    batch_num_samples = []
    if feed_dict is None:
        feed_dict = {}

    for i in range(num_batches):
        batch_data = data_set['data'][i*num_per_batch:(i+1)*num_per_batch]
        batch_label = data_set['labels'][i*num_per_batch:(i+1)*num_per_batch]
        batch_num_samples.append(batch_label.shape[0])
        feed_dict[input_placeholder] = batch_data
        feed_dict[labels_placeholder] = batch_label
        result = session.run(oplist, feed_dict)
        results.append(result)

    return results, batch_num_samples
