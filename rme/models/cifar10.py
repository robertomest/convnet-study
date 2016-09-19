from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import (MaxPooling2D, AveragePooling2D,
                                        ZeroPadding2D)
from keras.layers.normalization import BatchNormalization

from blocks import (add_nin_block, add_nin_bn_block,
                    add_conv, add_conv_bn)

### CHECK IF SENCOND POOLING SHOULD BE MAX OR AVG!!!!!
### WHEN USING BN, AVG IS BETTER
def nin_model(l2_reg=1e-4):
    """
    Network in Network model as introduced in [1].

    [1]: Lin et al, `Network in Network`. https://arxiv.org/abs/1312.4400
    """
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
    """
    Modified Network in Network architecture to include batch normalization
    before every non-linearity.
    """
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
    """
    VGG-like model that utilizes batch normalization and dropout.
    Implementation based on [1].

    [1] Zagoruyko, `CIFAR.torch`: https://github.com/szagoruyko/cifar.torch
    """
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


def allconv_model(l2_reg=1e-3):
    '''
    All Convolutional Net as described in [1].

    [1]: Springenberg et al, `Striving for Simplicity:
       The All Convolutional Net`: https://arxiv.org/abs/1412.6806
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
