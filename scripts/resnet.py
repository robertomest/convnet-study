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
from rme.datasets import cifar10
from rme.callbacks import Step, MetaCheckpoint
import os
import yaml
import numpy as np

### TODO: This script is still being tested.

def bottleneck_layer(x, nb_channels, kernel_size=3, stride=1, l2_reg=1e-4,
                     first=False):
    if first: # Skip BN-Relu
        out = x
    else:
        out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                                 beta_regularizer=l2(l2_reg))(x)
        out = Activation('relu')(out)
    # Apply the bottleneck convolution
    out = Convolution2D(nb_channels, 1, 1,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    # 3x3 conv with bottlenecked channels
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(out)
    out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(out)
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
        out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                                 beta_regularizer=l2(l2_reg))(x)
        out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(out)
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
    else:
        out = two_conv_layer(x, nb_channels, kernel_size=kernel_size,
                             l2_reg=l2_reg, first=first)
    out = merge([x, out], mode='sum')
    return out

def downsample_block(x, nb_channels, kernel_size=3, bottleneck=True,
                     l2_reg=1e-4):
    if bottleneck:
        out = bottleneck_layer(x, nb_channels, kernel_size=kernel_size,
                               strides=2, l2_reg=l2_reg)
        # The output channels is 4x bigger on this case
        nb_channels = nb_channels * 4
    else:
        out = two_conv_layer(x, nb_channels, kernel_size=kernel_size,
                             stride=2, l2_reg=l2_reg)
    # Projection on the shortcut
    proj = Convolution2D(nb_channels, 1, 1, subsample=(2, 2),
                         border_mode='same', init='he_normal',
                         W_regularizer=l2(l2_reg), bias=False)(x)
    out = merge([out, proj], mode='sum')
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
    strides = [1, 2, 2]
    inputs = Input((32, 32, 3))
    x = Convolution2D(16, 3, 3, border_mode='same', init='he_normal',
                      W_regularizer=l2(l2_reg), bias=False)(inputs)
    x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    for n, f in zip(nb_channels, [True, False, False]):
        x = block_stack(x, n, nb_blocks, bottleneck=bottleneck, l2_reg=l2_reg,
                        first=f)
    # Last BN-Relu
    x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    model = Model(input=inputs, output=x)
    return model

if __name__ == '__main__':
    model = resnet_model(2, bottleneck=False)
