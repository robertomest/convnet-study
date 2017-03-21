import keras
from keras.models import Model
from keras.layers import (Input, Convolution2D, Activation, BatchNormalization,
                          merge, AveragePooling2D, GlobalAveragePooling2D,
                          Dense, Dropout)
from keras.regularizers import l2
from rme.datasets import cifar10, cifar100, svhn, mnist, preprocessing
from rme.callbacks import Step

def preact_layer(x, num_channels, l2_reg, dropout, kernel_size=3,
                 name='layer'):
    '''
    Adds a preactivation layer for the densenet. This also includes l2
    reagularization on BatchNorm learnable parameters as in the original
    implementation.
    '''

    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg),
                             name=name + '_bn')(x)
    out = Activation('relu', name=name + '_relu')(out)
    out = Convolution2D(num_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False,
                        name=name + '_conv')(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out

def bottleneck_layer(x, num_channels, l2_reg, dropout, kernel_size=3,
                     name='layer'):
    '''
    DenseNet-B: 1x1 conv bottleneck before 3x3 conv
    '''
    o = preact_layer(x, num_channels*4, l2_reg, dropout, kernel_size=1,
                     name=name+'_bottleneck')
    o = preact_layer(o, num_channels, l2_reg, dropout, kernel_size=kernel_size,
                     name=name+'_main')

    return o

def dense_block(x, num_layers, growth_rate, l2_reg, dropout, bottleneck,
                name='block'):
    '''
    Adds a dense block for the densenet.
    '''
    for i in range(num_layers):
        # Get layer output
        if bottleneck:
            out = bottleneck_layer(x, growth_rate, l2_reg, dropout,
                                   name=name+'_layer%d' %(i + 1))
        else:
            out = preact_layer(x, growth_rate, l2_reg, dropout,
                               name=name+'_layer%d' %(i + 1))
        # Merge them on the channel axis
        merge_axis = -1
        # Concatenate input with layer ouput
        x = merge([x, out], mode='concat', concat_axis=merge_axis,
                  name=name + '_layer%d_concat' %(i + 1))
    return x

def transition_block(x, num_channels, l2_reg, dropout, name='transition'):
    '''
    Adds a transition block for the densenet. This halves the spatial
    dimensions.
    '''
    x = preact_layer(x, num_channels, l2_reg, dropout, kernel_size=1,
                     name=name + '_conv')
    # x = Convolution2D(n_channels, 1, 1, border_mode='same',
    #                   init='he_normal', W_regularizer=l2(l2_reg))(x)
    x = AveragePooling2D(name=name + '_pool')(x)
    return x

def model(dataset, num_blocks=3, num_layers=12, growth_rate=12,
          bottleneck=False, compression=1.,
          dropout=0., l2_reg=1e-4, init_channels=16):
    '''
    Implementation of Densenet[1] model which concatenate all previous layers'
    outputs as the current layer's input.

    If bottleneck is True, each layer is preceded by a 1x1 conv bottleneck
        (DenseNet-B)

    If compression < 1, each transition block will output
        output_channels = compression*input_channels (DenseNet-C)

    If both, DenseNet-BC

    [1] Huang, Liu and Weinberger. `Densely Connected Convolutional
        Networks`: https://arxiv.org/abs/1608.06993

    '''
    if compression > 1:
        raise ValueError('Compression rate should be <= one.' +
                         'Found: compression = %g' %compression)
    num_channels = init_channels
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        x = Input(shape=(32, 32, 3))
    else:
        raise ValueError('Model is not defined for dataset: %s' %dataset)

    # Initial convolution
    o = Convolution2D(init_channels, 3, 3, border_mode='same',
                      init='he_normal', W_regularizer=l2(l2_reg),
                      bias=False, name='first_conv')(x)
    for i in range(num_blocks - 1):
        # Create a dense block
        o = dense_block(o, num_layers, growth_rate, l2_reg, dropout, bottleneck,
                        name='block%d' %(i + 1))
        # Update the number of channels
        num_channels += num_layers*growth_rate

        num_channels = int(compression * num_channels)
        # Transition layer
        o = transition_block(o, num_channels, l2_reg, dropout,
                             name='transition%d' %(i + 1))

    i += 1
    # Add last dense_block
    o = dense_block(o, num_layers, growth_rate, l2_reg, dropout, bottleneck,
                    name='block%d' %(i + 1))
    # Add final BN-Relu
    o = BatchNormalization(gamma_regularizer=l2(l2_reg),
                           beta_regularizer=l2(l2_reg),
                           name='last_bn')(o)
    o = Activation('relu', name='last_relu')(o)
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
