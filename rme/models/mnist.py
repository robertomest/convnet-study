from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from blocks import add_nin_bn_block

def simple_model():
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='same',
                          input_shape=(28, 28, 1), name='Conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 5, 5, border_mode='same', name='Conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1024, name='Dense1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, name='Dense2'))
    model.add(Activation('softmax'))
    return model

def nin_bn_model(l2_reg=1e-4):
    """
    Modified Network in Network architecture to include batch normalization
    before every non-linearity.
    """
    model = Sequential()
    add_nin_bn_block(model, [128, 96, 48], l2_reg=l2_reg,
                   input_shape=(28, 28, 1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.5))
    add_nin_bn_block(model, [128, 96, 48], l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                             border_mode='same'))
    model.add(Dropout(0.5))
    add_nin_bn_block(model, [192, 96, 10], l2_reg=l2_reg)
    model.add(AveragePooling2D(pool_size=(7, 7)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model
