from keras.layers import Activation
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

def add_conv(model, n_filters, kernel_size=3, init='normal',
             stride=1, l2_reg=1e-3, border_mode='same',
             input_shape=(None,None)):
  '''
  Adds a block of convolution and relu to the model.
  '''
  model.add(Convolution2D(n_filters, kernel_size, kernel_size,
                          border_mode=border_mode, init=init,
                          subsample=(stride, stride), W_regularizer=l2(l2_reg),
                          input_shape=input_shape))
  model.add(Activation('relu'))

def add_conv_bn(model, n_filters, kernel_size=3, init='he_uniform',
             stride=1, l2_reg=1e-3, border_mode='same',
             input_shape=(None,None)):
  '''
  Adds a block of convolution, batch normalization and relu to the model.
  '''
  model.add(Convolution2D(n_filters, kernel_size, kernel_size,
                          border_mode=border_mode, init=init,
                          subsample=(stride, stride), W_regularizer=l2(l2_reg),
                          input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

'''
Adds a network in network block to the model.
This is a block with a 5x5 convolution, followed by two 1x1 convolutions.
All of them have relu activations.
inputs:
model: Keras model to which the block will be added.
n_filters: a list with the amount of filter (kernels) of each layer.
regs: the amount of l2 regularization on each layer.
'''
def add_nin_block(model, n_filters, kernel_size=5, init='normal',
                  l2_reg=1e-4, input_shape=(None, None)):
  add_conv(model, n_filters[0], kernel_size=kernel_size, init=init,
           l2_reg=l2_reg, input_shape=input_shape)
  add_conv(model, n_filters[1], kernel_size=1, init=init, l2_reg=l2_reg)
  add_conv(model, n_filters[2], kernel_size=1, init=init, l2_reg=l2_reg)


def add_nin_bn_block(model, n_filters, kernel_size=5, init='he_uniform',
                     l2_reg=1e-4, input_shape=(None, None)):
  add_conv_bn(model, n_filters[0], kernel_size=kernel_size, init=init,
              l2_reg=l2_reg, input_shape=input_shape)
  add_conv_bn(model, n_filters[1], kernel_size=1, init=init, l2_reg=l2_reg)
  add_conv_bn(model, n_filters[2], kernel_size=1, init=init, l2_reg=l2_reg)


def add_vgg_block(model, n_filters, init='he_uniform', l2_reg=5e-4,
                  input_shape=(None, None), border='same'):
  model.add(Convolution2D(filters, 3, 3, border_mode=border,
                          input_shape=input_shape, init=init,
                          W_regularizer=l2(l2_reg)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
