import keras
from keras.models import Model
from keras.layers import (Input, Convolution2D, Activation, BatchNormalization,
                          Dropout, MaxPooling2D, AveragePooling2D,
                          GlobalAveragePooling2D)
from keras.regularizers import l2
from rme.datasets import cifar10, cifar100, svhn, mnist, preprocessing
from rme.callbacks import Step

# Functions
def nin_block(x, filters_list, filter_size, l2_reg, stride=1, bn=True,
              init='he_normal', prefix=''):
    o = x
    name = prefix + '_conv'
    idx = 1
    use_bias = not bn
    for num_filters in filters_list:

        o = Convolution2D(num_filters, filter_size, filter_size,
                          border_mode='same', subsample=(stride, stride),
                          W_regularizer=l2(l2_reg), bias=use_bias,
                          init=init, name=name)(o)
        if bn:
            o = BatchNormalization(name=name+'_bn')(o)
        o = Activation('relu', name=name+'_relu')(o)
        filter_size = 1 # Only the first convoution has size > 1
        name = prefix + '_nin%d' %(idx)
        idx += 1
    return o

def model(dataset, l2_reg=1e-4, drop_p=0.5, bn=True, init='he_normal'):
    if dataset in ['cifar10', 'cifar100']:
        x = Input((32, 32, 3))
        filters = [[192, 160, 96], [192, 192, 192], [192, 192, 10]]
        if dataset == 'cifar100':
            filters[-1][-1] = 100
    elif dataset == 'svhn':
        x = Input((32, 32, 3))
        filters = [[128, 96, 64], [320, 256, 128], [384, 256, 10]]
    elif dataset == 'mnist':
        x = Input((28, 28, 1))
        # filters = [[96, 64, 48], [128, 96, 48], [128, 96, 10]]
        filters = [[128, 96, 48], [128, 96, 48], [128, 96, 10]]
    else:
        raise ValueError('Model is not defined for dataset: %s' %dataset)

    # Define the network
    o = nin_block(x, filters[0], 5, l2_reg, bn=bn, init=init, prefix='block1')
    o = MaxPooling2D(pool_size=(3,3), strides=(2, 2), border_mode='same')(o)
    if drop_p > 0:
        o = Dropout(drop_p)(o)
    o = nin_block(o, filters[1], 5, l2_reg, bn=bn, init=init, prefix='block2')
    o = AveragePooling2D(pool_size=(3,3), strides=(2, 2), border_mode='same')(o)
    if drop_p > 0:
        o = Dropout(drop_p)(o)
    o = nin_block(o, filters[2], 5, l2_reg, bn=bn, init=init, prefix='block3')
    o = GlobalAveragePooling2D()(o)
    o = Activation('softmax')(o)

    return Model(input=x, output=o)

def preprocess_data(train_set, valid_set, test_set, dataset):
    if dataset == 'mnist':
        # train_set, mean, std = preprocessing.normalization(train_set)
        # valid_set, _, _ = preprocessing.normalization(valid_set, mean, std)
        # test_set, _, _ = preprocessing.normalization(test_set, mean, std)
        train_set = mnist.preprocess(train_set)
        valid_set = mnist.preprocess(valid_set)
        test_set = mnist.preprocess(test_set)
    elif dataset == 'cifar10':
        train_set = cifar10.preprocess(train_set)
        valid_set = cifar10.preprocess(valid_set)
        test_set = cifar10.preprocess(test_set)
    elif dataset == 'cifar100':
        train_set = cifar100.preprocess(train_set)
        valid_set = cifar100.preprocess(valid_set)
        test_set = cifar100.preprocess(test_set)
    elif dataset == 'svhn':
        train_set = svhn.preprocess(train_set)
        valid_set = svhn.preprocess(valid_set)
        test_set = svhn.preprocess(test_set)
    else:
        raise ValueError('Preprocessing not defined for dataset: %s' %dataset)

    return train_set, valid_set, test_set

def default_args(dataset):
    training_args = {}
    if dataset =='mnist':
        training_args['lr'] = 0.1
        training_args['epochs'] = 30
        training_args['batch_size'] = 128
    elif dataset in ['cifar10', 'cifar100']:
        training_args['lr'] = 0.1
        training_args['epochs'] = 250
        training_args['batch_size'] = 64
    elif dataset == 'svhn':
        training_args['lr'] = 0.1
        training_args['epochs'] = 40
        training_args['batch_size'] = 64
    else:
        print('Default args not defined for dataset: %s' %dataset)
    return training_args

def schedule(dataset, lr):
    if dataset == 'mnist':
        steps = [20]
        lrs = [lr, lr/10]
    elif dataset in ['cifar10', 'cifar100']:
        steps = [25*i for i in range(1, 10)]
        lrs = [lr/ 2**i for i in range(10)]
    elif dataset == 'svhn':
        steps = [20, 30]
        lrs = [lr, lr/10, lr/100]
    else:
        raise ValueError('Schedule not defined for dataset: %s' %dataset)
    return Step(steps, lrs)
