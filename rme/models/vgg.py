import keras
from keras.models import Model
from keras.layers import (Input, Convolution2D, Activation, BatchNormalization,
                          Dropout, MaxPooling2D, ZeroPadding2D, Flatten, Dense)
from keras.regularizers import l2
from rme.datasets import cifar10, cifar100, svhn, mnist, preprocessing
from rme.callbacks import Step

def conv_bn_relu(x, num_filters, l2_reg, init='he_normal', border_mode='same',
                 name=None):
    o = Convolution2D(num_filters, 3, 3, border_mode=border_mode,
                      W_regularizer=l2(l2_reg), bias=False,
                      init=init, name=name)(x)
    o = BatchNormalization(name=name+'_bn')(o)
    o = Activation('relu', name=name+'_relu')(o)
    return o

def model(dataset, l2_reg=5e-4, init='he_normal'):
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        x = Input((32, 32, 3))
    else:
        raise ValueError('Model is not defined for dataset: %s' %dataset)

    # Input size is 32x32
    o = conv_bn_relu(x, 64, l2_reg, init=init, name='block1_conv1')
    o = Dropout(0.3)(o)
    o = conv_bn_relu(o, 64, l2_reg, init=init, name='block1_conv2')
    o = MaxPooling2D()(o)

    # Input size is 16x16
    o = conv_bn_relu(o, 128, l2_reg, init=init, name='block2_conv1')
    o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 128, l2_reg, init=init, name='block2_conv2')
    o = MaxPooling2D()(o)

    # Input size is 8x8
    o = conv_bn_relu(o, 256, l2_reg, init=init, name='block3_conv1')
    o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 256, l2_reg, init=init, name='block3_conv2')
    o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 256, l2_reg, init=init, name='block3_conv3')
    o = MaxPooling2D()(o)

    # Input size is 4x4
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block4_conv1')
    o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block4_conv2')
    o = Dropout(0.4)(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block4_conv3')
    o = MaxPooling2D()(o)

    # Input size is 2x2
    # Manually pad the image to 4x4 and use VALID padding to get it back to 2x2
    o = ZeroPadding2D(padding=(1,1))(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv1',
                     border_mode='valid')
    o = Dropout(0.4)(o)
    o = ZeroPadding2D(padding=(1,1))(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv2',
                     border_mode='valid')
    o = Dropout(0.4)(o)
    o = ZeroPadding2D(padding=(1,1))(o)
    o = conv_bn_relu(o, 512, l2_reg, init=init, name='block5_conv3',
                     border_mode='valid')
    o = MaxPooling2D()(o)

    # Input size is 1x1
    o = Flatten()(o)

    # Classifier
    o = Dropout(0.5)(o)
    o = Dense(512)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Dropout(0.5)(o)
    if dataset in ['cifar10', 'svhn']:
        output_size = 10
    elif dataset == 'cifar100':
        output_size = 100
    o = Dense(output_size)(o)
    o = Activation('softmax')(o)

    return Model(input=x, output=o)

def preprocess_data(train_set, valid_set, test_set, dataset):
    if dataset == 'cifar10':
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
    if dataset in ['cifar10', 'cifar100']:
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
    if dataset in ['cifar10', 'cifar100']:
        steps = [25*i for i in range(1, 10)]
        lrs = [lr/ 2**i for i in range(10)]
    elif dataset == 'svhn':
        steps = [20, 30]
        lrs = [lr, lr/10, lr/100]
    else:
        raise ValueError('Schedule not defined for dataset: %s' %dataset)
    return Step(steps, lrs)
