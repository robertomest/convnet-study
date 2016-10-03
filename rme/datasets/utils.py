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

def normalization(data_set, mean=None, std=None, eps=1e-6):
    '''
    Normalizes data across each dimension by removing it's mean and dividing
    by it's standard deviation.

    Arguments:
        data_set: numpy array of shape(batch_size, ...).
        mean: numpy array with the same shape as the input, excluding the
            batch axis, that will be used as the mean. If None (Default),
            the mean will be computed from the input data.
        std: numpy array with the same shape as the input, excluding the
            batch axis, that will be used as the standard deviation. If None
            (Default), the mean will be computed from the input data.
        eps: small constant to avoid division by very small numbers during
            normalization. If the a divisor is smaller than eps, no division
            will be carried out on that dimension.
    '''
    if mean is None:
        mean = np.mean(data_set, axis=0)
    if std is None:
        std = np.std(data_set, axis=0, ddof=1)
    std[std < eps] = 1
    data_set -= mean
    data_set /= std
    return data_set, mean, std

def global_contrast_normalization(data_set, eps=1e-6):
    '''
    Applies global contrast normalization to the input image data.

    Arguments:
        data_set: numpy array of shape (batch_size, dim). If the input has
            more than 2 dimensions (such as images), it will be flatten the
            data.
        eps: small constant to avoid division by very small numbers during
            normalization. If the a divisor is smaller than eps, no division
            will be carried out on that dimension.
    Returns:
        norm_data: numpy array with normalized data. Has the same shape
            as the input.
    '''
    if not data_set.size:
        # Simply return if data_set is empty
        return data_set
    data_shape = data_set.shape
    # If data has more than 2 dims, normalize along all axis > 0
    if len(data_shape) > 2:
        size = data_shape[0]
        norm_data = data_set.reshape((size, -1))
    else:
        norm_data = data_set
    mean = norm_data.mean(axis=1)
    norm_data -= mean[:, np.newaxis]
    std = norm_data.std(axis=1, ddof=1)
    std[std < eps] = 1
    norm_data /= std[:, np.newaxis]
    return norm_data.reshape(data_shape)

def zca_whitening(data_set, mean=None, whitening=None):
    '''
    Applies ZCA whitening the the input data.

    Arguments:
        data_set: numpy array of shape (batch_size, dim). If the input has
            more than 2 dimensions (such as images), it will be flatten the
            data.
        mean: numpy array of shape (dim) that will be used as the mean.
            If None (Default), the mean will be computed from the input data.
        whitening: numpy array shaped (dim, dim) that will be used as the
            whitening matrix. If None (Default), the whitening matrix will be
            computed from the input data.

    Returns:
        white_data: numpy array with whitened data. Has the same shape as
            the input.
        mean: numpy array of shape (dim) that contains the mean of each input
            dimension. If mean was provided as input, this is a copy of it.
        whitening:  numpy array of shape (dim, dim) that contains the whitening
            matrix. If whitening was provided as input, this is a copy of it.
    '''
    if not data_set.size:
        # Simply return if data_set is empty
        return data_set, mean, whitening
    data_shape = data_set.shape
    if len(data_shape) > 2:
        size = data_shape[0]
        white_data = data_set.reshape((size, -1))
    else:
        white_data = data_set

    if mean is None:
        # No mean matrix, we must compute it
        mean = white_data.mean(axis=0)
    # Remove mean
    white_data -= mean

    # If no whitening matrix, we must compute it
    if whitening is None:
        cov = np.dot(white_data.T, white_data)/size
        U, S, V = np.linalg.svd(cov)
        whitening = np.dot(np.dot(U, np.diag(1./np.sqrt(S + 1e-6))), U.T)

    white_data = np.dot(white_data, whitening)
    return white_data.reshape(data_shape), mean, whitening

def per_channel_normalization(data_set, mean=None, std=None):
    '''
    Applies channel-wise mean and standard deviation normalization.

    Arguments:
        data_set: numpy array of shape (samples, height, width, channels).
        mean: numpy array of shape (channels,) that contains the mean values
            of the channels. If None (Default), the mean will be computed
            from the input data.
        std: numpy array of shape (channels,) that contains the standard
            deviation values of the channels. If None (Default), the mean
            will be computed from the input data.

    Returns:
        normalized_set: numpy array with normalized data. Has same shape as the
            input.
        mean: numpy array of shape (channels,) that contains the values by which
            the mean of each channel was subtracted by. If a mean was provided
            as input, this is it.
        std: numpy array of shape (channels,) that contains the values by which
            the standard deviation of each channel was divided by. If a mean was
            provided as input, this is it.
    '''
    if len(data_set.shape) < 4:
        raise Exception('Expected 4 dim tensor, found shape: %s'
                        %str(data_set.shape))
    if mean is None:
        mean = np.mean(data_set, axis=(0, 1, 2))
    if std is None:
        std = np.std(data_set, axis=(0, 1, 2))

    normalized_set = data_set - mean
    normalized_set /= std

    return normalized_set, mean, std

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
