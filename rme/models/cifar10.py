from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers.convolutional import ZeroPadding2D, Convolution2D
from keras.layers.pooling import (MaxPooling2D, AveragePooling2D,
                                  GlobalAveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from blocks import (add_nin_block, add_nin_bn_block, #NiN
                    add_conv, add_conv_bn, #VGG
                    preact_layer, dense_block, transition_block, #Densenet
                    block_stack) #Resnet

import tensorflow as tf

### CHECK IF SENCOND POOLING SHOULD BE MAX OR AVG!!!!!
### WHEN USING BN, AVG IS BETTER
def nin_model(l2_reg=1e-4):
    '''
    Network in Network model as introduced in [1].

    [1] Lin et al. `Network in Network`. https://arxiv.org/abs/1312.4400
    '''
    model = Sequential()
    add_nin_block(model, [192, 160, 96], l2_reg=l2_reg, input_shape=(32, 32, 3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.5))
    add_nin_block(model, [192, 192, 192], l2_reg=l2_reg)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.5))
    add_nin_block(model, [192, 192, 10], kernel_size=3, l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


def nin_bn_model(l2_reg=1e-4):
    '''
    Modified Network in Network architecture to include batch normalization
    before every non-linearity.
    '''
    model = Sequential()
    add_nin_bn_block(model, [192, 160, 96], l2_reg=l2_reg,
                   input_shape=(32, 32, 3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.5))
    add_nin_bn_block(model, [192, 192, 192], l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                             border_mode='same'))
    model.add(Dropout(0.5))
    add_nin_bn_block(model, [192, 192, 10], l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


def vgg_model(l2_reg=5e-4):
    '''
    VGG-like model that utilizes batch normalization and dropout.
    Implementation based on [1].

    [1] Zagoruyko. `CIFAR.torch`: https://github.com/szagoruyko/cifar.torch
    '''
    # Input size is 32x32
    model = Sequential()
    add_conv_bn(model, 64, l2_reg=l2_reg, input_shape=(32, 32, 3))
    model.add(Dropout(0.3))
    add_conv_bn(model, 64, l2_reg=l2_reg)
    model.add(MaxPooling2D())

    # Input size is 16x16
    add_conv_bn(model, 128, l2_reg=l2_reg)
    model.add(Dropout(0.4))
    add_conv_bn(model, 128, l2_reg=l2_reg)
    model.add(MaxPooling2D())

    # Input size is 8x8
    add_conv_bn(model, 256, l2_reg=l2_reg)
    model.add(Dropout(0.4))
    add_conv_bn(model, 256, l2_reg=l2_reg)
    model.add(Dropout(0.4))
    add_conv_bn(model, 256, l2_reg=l2_reg)
    model.add(MaxPooling2D())

    # Input size is 4x4
    add_conv_bn(model, 512, l2_reg=l2_reg)
    model.add(Dropout(0.4))
    add_conv_bn(model, 512, l2_reg=l2_reg)
    model.add(Dropout(0.4))
    add_conv_bn(model, 512, l2_reg=l2_reg)
    model.add(MaxPooling2D())

    # Input size is 2x2
    # Manually pad the image to 4x4 and use VALID padding to get it to 2x2
    model.add(ZeroPadding2D(padding=(1,1)))
    add_conv_bn(model, 512, l2_reg=l2_reg, border_mode='valid')
    model.add(Dropout(0.4))
    model.add(ZeroPadding2D(padding=(1,1)))
    add_conv_bn(model, 512, l2_reg=l2_reg, border_mode='valid')
    model.add(Dropout(0.4))
    model.add(ZeroPadding2D(padding=(1,1)))
    add_conv_bn(model, 512, l2_reg=l2_reg, border_mode='valid')
    model.add(MaxPooling2D())

    # Input size is 1x1
    model.add(Flatten())

    # Classifier
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def densenet_model(nb_blocks, nb_layers, growth_rate, dropout=0., l2_reg=1e-4,
                   init_channels=16):
    '''
    Implementation of Densenet[1] model which concatenate all previous layers'
    outputs as the current layer's input.

    [1] Huang, Liu and Weinberger. `Densely Connected Convolutional
        Networks`: https://arxiv.org/abs/1608.06993

    '''
    with tf.variable_scope('densenet'):
        n_channels = init_channels
        inputs = Input(shape=(32, 32, 3))
        x = Convolution2D(init_channels, 3, 3, border_mode='same',
                          init='he_normal', W_regularizer=l2(l2_reg),
                          bias=False)(inputs)
        for i in range(nb_blocks - 1):
            # Create a dense block
            x = dense_block(x, nb_layers, growth_rate,
                            dropout=dropout, l2_reg=l2_reg)
            # Update the number of channels
            n_channels += nb_layers*growth_rate
            # Transition layer
            x = transition_block(x, n_channels, dropout=dropout, l2_reg=l2_reg)

        # Add last dense_block
        x = dense_block(x, nb_layers, growth_rate, dropout=dropout, l2_reg=l2_reg)
        # Add final BN-Relu
        x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                                 beta_regularizer=l2(l2_reg))(x)
        x = Activation('relu')(x)
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        x = Dense(10, W_regularizer=l2(l2_reg))(x)
        x = Activation('softmax')(x)

        model = Model(input=inputs, output=x)
    return model

def resnet_model(nb_blocks, bottleneck=True, l2_reg=1e-4):
    '''
    Resnet[1] model that uses preactivation[2]. Supports both regular and
    bottleneck residual units. Uses B-type shortcuts: shortcuts are identity
    unless output and input feature maps have different dimensions. In this
    case, a 1x1 convolution (possibly with stride 2) is used as projection.

    [1] He et al. `Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    [2] He et al. `Identity Mappings in Deep Residual Networks`:
        https://arxiv.org/abs/1603.05027
    '''
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

def allconv_model(l2_reg=1e-3):
    '''
    All Convolutional Net as described in [1].

    [1] Springenberg et al. `Striving for Simplicity: The All Convolutional
        Net`: https://arxiv.org/abs/1412.6806
    '''
    model = Sequential()
    # Apply dropout to inputs
    model.add(Dropout(0.2, input_shape=(32, 32, 3)))

    add_conv(model, 96, l2_reg=l2_reg)
    add_conv(model, 96, l2_reg=l2_reg)
    add_conv(model, 96, stride=2, l2_reg=l2_reg)
    model.add(Dropout(0.5))

    add_conv(model, 192, l2_reg=l2_reg)
    add_conv(model, 192, l2_reg=l2_reg)
    add_conv(model, 192, stride=2, l2_reg=l2_reg)
    model.add(Dropout(0.5))

    add_conv(model, 192, l2_reg=l2_reg)
    add_conv(model, 192, kernel_size=1, l2_reg=l2_reg)
    add_conv(model, 10, kernel_size=1, l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model

def allconv_bn_model(l2_reg=1e-5):
    model = Sequential()
    # Apply dropout to inputs
    model.add(Dropout(0.2, input_shape=(32, 32, 3)))

    add_conv_bn(model, 96, l2_reg=l2_reg)
    add_conv_bn(model, 96, l2_reg=l2_reg)
    add_conv_bn(model, 96, stride=2, l2_reg=l2_reg)
    model.add(Dropout(0.5))

    add_conv_bn(model, 192, l2_reg=l2_reg)
    add_conv_bn(model, 192, l2_reg=l2_reg)
    add_conv_bn(model, 192, stride=2, l2_reg=l2_reg)
    model.add(Dropout(0.5))

    add_conv_bn(model, 192, l2_reg=l2_reg)
    add_conv_bn(model, 192, kernel_size=1, l2_reg=l2_reg)
    add_conv_bn(model, 10, kernel_size=1, l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model
