import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, merge, Activation, Dropout, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from rme.datasets import cifar10
from rme.callbacks import Step, MetaCheckpoint

import os
import yaml
import numpy as np

def bottleneck_layer(x, nb_channels, kernel_size=3, stride=1, l2_reg=1e-4,
                     first=False):
    if first: # Skip BN-Relu
        out = x
    else:
        out = BatchNormalization()(x)
        out = Activation('relu')(out)
    # Apply the bottleneck convolution
    out = Convolution2D(nb_channels, 1, 1,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    # 3x3 conv with bottlenecked channels
    # We stride it on 3x3 conv as done on Facebook's implementation
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    # 1x1 conv that expands the number of channels
    out = Convolution2D(nb_channels*4, 1, 1,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    return out

def two_conv_layer(x, nb_channels, kernel_size=3, stride=1, l2_reg=1e-4,
                   first=False):
    if first: # Skip BN-Relu
        out = x
    else:
        out = BatchNormalization()(x)
        out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    return out

def residual_block(x, nb_channels, kernel_size=3, bottleneck=True, l2_reg=1e-4,
                   first=False):
    if bottleneck:
        out = bottleneck_layer(x, nb_channels, kernel_size=kernel_size,
                               l2_reg=l2_reg, first=first)
        if first:
            x = Convolution2D(nb_channels * 4, 1, 1,
                              border_mode='valid', init='he_normal',
                              W_regularizer=l2(l2_reg), bias=False)(x)
    else:
        out = two_conv_layer(x, nb_channels, kernel_size=kernel_size,
                             l2_reg=l2_reg, first=first)
    out = merge([x, out], mode='sum')
    return out

def downsample_block(x, nb_channels, kernel_size=3, bottleneck=True,
                     l2_reg=1e-4):
    if bottleneck:
        out = bottleneck_layer(x, nb_channels, kernel_size=kernel_size,
                               stride=2, l2_reg=l2_reg)
        # The output channels is 4x bigger on this case
        nb_channels = nb_channels * 4
    else:
        out = two_conv_layer(x, nb_channels, kernel_size=kernel_size,
                             stride=2, l2_reg=l2_reg)
    # Projection on the shortcut
    proj = Convolution2D(nb_channels, 1, 1, subsample=(2, 2),
                         border_mode='valid', init='he_normal',
                         W_regularizer=l2(l2_reg), bias=False)(x)
    # proj = AveragePooling2D((1, 1), (2, 2))(x)
    out = merge([proj, out], mode='sum')
    return out

def block_stack(x, nb_channels, nb_blocks, kernel_size=3, bottleneck=True,
                l2_reg=1e-4, first=False):
    if first:
        x = residual_block(x, nb_channels, kernel_size=kernel_size,
                           bottleneck=bottleneck, l2_reg=l2_reg,
                           first=True)
    else:
        x = downsample_block(x, nb_channels, kernel_size=kernel_size,
                           bottleneck=bottleneck, l2_reg=l2_reg)
    for _ in range(nb_blocks-1):
        x = residual_block(x, nb_channels, kernel_size=kernel_size,
                           bottleneck=bottleneck, l2_reg=l2_reg)
    return x

def resnet_model(nb_blocks, bottleneck=True, l2_reg=1e-4):
    nb_channels = [16, 32, 64]
    inputs = Input((32, 32, 3))
    x = Convolution2D(16, 3, 3, border_mode='same', init='he_normal',
                      W_regularizer=l2(l2_reg), bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for n, f in zip(nb_channels, [True, False, False]):
        x = block_stack(x, n, nb_blocks, bottleneck=bottleneck, l2_reg=l2_reg,
                        first=f)
    # Last BN-Relu
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    model = Model(input=inputs, output=x)
    return model

def preprocess_data(data_set):
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])

    data_set -= mean
    data_set /= std
    return data_set

if __name__ == '__main__':
    augmented = True

    # Number of blocks per stack
    nb_blocks = 9
    bottleneck = True
    if bottleneck:
        depth = nb_blocks*9 + 2
    else:
        depth = nb_blocks*6 + 2
    file_name = 'resnet%d' %depth
    if augmented:
        file_name = file_name + '_augmented'
    print 'Checkpoint name: %s' %file_name

    save_path = os.path.join('weights', 'cifar10', file_name)

    # Define optimizer
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

    model = resnet_model(nb_blocks, bottleneck=bottleneck)
    from keras.utils.visualize_util import plot
    plot(model, show_shapes=True)
    # from rme.models.cifar10 import nin_bn_model
    # model = nin_bn_model()
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print 'Model compiled.'

    print 'Loading data.'
    train_set, _, test_set = cifar10.load('data/cifar10', valid_ratio=0.,
                                          gcn=False, zca=False)
    train_set['data'] = preprocess_data(train_set['data'])
    test_set['data'] = preprocess_data(test_set['data'])
    print 'Data loaded.'

    # nb_epoch = 82
    nb_epoch = 100
    offset = 0
    # Define callbacks for checkpointing and scheduling
    callbacks = []
    callbacks.append(ModelCheckpoint(save_path + '.h5'))
    steps = [1, nb_epoch/2 - offset, 3*nb_epoch/4 - offset]
    schedule = Step(steps, [0.01, 0.1, 0.01, 0.001], verbose=1)
    callbacks.append(schedule)
    schedule = None
    # Custom meta checkpoint that saves training information at each epoch.
    callbacks.append(MetaCheckpoint(save_path + '.meta', schedule=schedule))

    if augmented:
        data_gen = ImageDataGenerator(horizontal_flip=True,
                                      width_shift_range=0.125,
                                      height_shift_range=0.125,
                                      fill_mode='constant')
        data_iter = data_gen.flow(train_set['data'], train_set['labels'],
                                  batch_size=64, shuffle=True)
        print 'Starting fit_generator.'
        model.fit_generator(data_iter,
                            samples_per_epoch=train_set['data'].shape[0],
                            nb_epoch=(nb_epoch - offset),
                            verbose = 2,
                            validation_data=(test_set['data'],
                                             test_set['labels']),
                            callbacks=callbacks)
    else:
        model.fit(train_set['data'], train_set['labels'], batch_size=64,
                  nb_epoch=(nb_epoch - offset), verbose=2,
                  validation_data=(test_set['data'], test_set['labels']),
                  callbacks=callbacks, shuffle=True)
