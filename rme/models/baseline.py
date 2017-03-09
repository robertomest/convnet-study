import keras
from keras.models import Sequential
from keras.layers import (Input, Convolution2D, Activation, Dropout,
                          AveragePooling2D, MaxPooling2D, Flatten, Dense,
                          BatchNormalization)
from keras.regularizers import l2
from rme.datasets import preprocessing, mnist
from rme.callbacks import Step

def model(dataset, l2_reg=1e-5, drop_p=0.5, init='he_normal'):
    if dataset != 'mnist':
        raise ValueError('Model is not defined for dataset: %s.' %dataset)
    # model = Sequential()
    # model.add(Convolution2D(32, 3, 3, border_mode='same',
    #                       input_shape=(28, 28, 1), W_regularizer=l2(l2_reg),
    #                       init=init))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3, border_mode='same',
    #                         W_regularizer=l2(l2_reg), init=init))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, W_regularizer=l2(l2_reg), init=init))
    # model.add(Activation('relu'))
    # model.add(Dropout(drop_p))
    # model.add(Dense(10, init=init))
    # model.add(Activation('softmax'))
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='same',
                          input_shape=(28, 28, 1), W_regularizer=l2(l2_reg),
                          init=init))
    model.add(Activation('relu'))
    model.add(AveragePooling2D())
    model.add(Convolution2D(64, 5, 5, border_mode='same',
                            W_regularizer=l2(l2_reg), init=init))
    model.add(Activation('relu'))
    model.add(AveragePooling2D())
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(200, W_regularizer=l2(l2_reg), init=init))
    model.add(Activation('relu'))
    if drop_p > 0:
        model.add(Dropout(drop_p))
    model.add(Dense(10, init=init))
    model.add(Activation('softmax'))
    return model

def preprocess_data(train_set, valid_set, test_set, dataset):
    if dataset == 'mnist':
        train_set, mean, std = preprocessing.normalization(train_set)
        valid_set, _, _ = preprocessing.normalization(valid_set, mean, std)
        test_set, _, _ = preprocessing.normalization(test_set, mean, std)
        # train_set = mnist.preprocess(train_set)
        # valid_set = mnist.preprocess(valid_set)
        # test_set = mnist.preprocess(test_set)
        # train_set = train_set/255
        # valid_set = valid_set/255
        # test_set = test_set/255
    else:
        raise ValueError('Preprocessing not defined for dataset: %s.' %dataset)

    return train_set, valid_set, test_set

def default_args(dataset):
    training_args = {}
    if dataset =='mnist':
        training_args['lr'] = 0.1
        training_args['epochs'] = 30
        training_args['batch_size'] = 128

    return training_args

def schedule(dataset, lr):
    if dataset == 'mnist':
        steps = [20]
        lrs = [lr, lr/10]
    else:
        raise ValueError('Schedule not defined for dataset: %s.' %dataset)
    return Step(steps, lrs)
