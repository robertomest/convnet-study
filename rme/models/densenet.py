import keras
from keras.models import Model
from keras.layers import (Input, Convolution2D, Activation, BatchNormalization,
                          merge, AveragePooling2D, GlobalAveragePooling2D,
                          Dense, Dropout)
from keras.regularizers import l2
from rme.datasets import cifar10, cifar100, svhn, mnist, preprocessing
from rme.callbacks import Step

def preact_layer(x, num_channels, l2_reg, dropout, kernel_size=3):
    '''
    Adds a preactivation layer for the densenet. This also includes l2
    reagularization on BatchNorm learnable parameters as in the original
    implementation.
    '''
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    out = Activation('relu')(out)
    out = Convolution2D(num_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out

def dense_block(x, num_layers, growth_rate, l2_reg, dropout):
    '''
    Adds a dense block for the densenet.
    '''
    for i in range(num_layers):
        # Get layer output
        out = preact_layer(x, growth_rate, l2_reg, dropout)
        # Merge them on the channel axis
        merge_axis = -1
        # Concatenate input with layer ouput
        x = merge([x, out], mode='concat', concat_axis=merge_axis)
    return x

def transition_block(x, num_channels, l2_reg, dropout):
    '''
    Adds a transition block for the densenet. This halves the spatial
    dimensions.
    '''
    x = preact_layer(x, num_channels, l2_reg, dropout, kernel_size=1)
    # x = Convolution2D(n_channels, 1, 1, border_mode='same',
    #                   init='he_normal', W_regularizer=l2(l2_reg))(x)
    x = AveragePooling2D()(x)
    return x

def model(dataset, num_blocks=3, num_layers=12, growth_rate=12, dropout=0.,
          l2_reg=1e-4, init_channels=16):
    '''
    Implementation of Densenet[1] model which concatenate all previous layers'
    outputs as the current layer's input.

    [1] Huang, Liu and Weinberger. `Densely Connected Convolutional
        Networks`: https://arxiv.org/abs/1608.06993

    '''
    num_channels = init_channels
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        x = Input(shape=(32, 32, 3))
    else:
        raise ValueError('Model is not defined for dataset: %s' %dataset)

    # Initial convolution
    o = Convolution2D(init_channels, 3, 3, border_mode='same',
                      init='he_normal', W_regularizer=l2(l2_reg),
                      bias=False)(x)
    for i in range(num_blocks - 1):
        # Create a dense block
        o = dense_block(o, num_layers, growth_rate, l2_reg, dropout)
        # Update the number of channels
        num_channels += num_layers*growth_rate
        # Transition layer
        o = transition_block(o, num_channels, l2_reg, dropout)

    # Add last dense_block
    o = dense_block(o, num_layers, growth_rate, l2_reg, dropout)
    # Add final BN-Relu
    o = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(o)
    o = Activation('relu')(o)
    # Global average pooling
    o = GlobalAveragePooling2D()(o)
    if dataset in ['cifar10', 'svhn']:
        output_size = 10
    elif dataset == 'cifar100':
        output_size = 100
    o = Dense(output_size, W_regularizer=l2(l2_reg))(o)
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

    training_args['lr'] = 0.1
    training_args['batch_size'] = 64

    if dataset in ['cifar10', 'cifar100']:
        training_args['epochs'] = 300
    elif dataset == 'svhn':
        training_args['epochs'] = 40
    else:
        print('Default args not defined for dataset: %s' %dataset)

    return training_args


def schedule(dataset, lr):
    if dataset in ['cifar10', 'cifar100']:
        steps = [150, 225]
        lrs = [lr, lr/10, lr/100]
    elif dataset == 'svhn':
        steps = [20, 30]
        lrs = [lr, lr/10, lr/100]
    else:
        raise ValueError('Schedule not defined for dataset: %s' %dataset)
    return Step(steps, lrs)
