import keras.backend as K
from keras.layers import Activation, merge, Dropout
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D

def add_conv(model, n_filters, kernel_size=3, init='normal',
             stride=1, l2_reg=1e-3, border_mode='same',
             input_shape=(None,None)):
    '''
    Adds a block of convolution and relu to the model.
    '''
    model.add(Convolution2D(n_filters, kernel_size, kernel_size,
                            border_mode=border_mode, init=init,
                            subsample=(stride, stride),
                            W_regularizer=l2(l2_reg), input_shape=input_shape))
    model.add(Activation('relu'))

def add_conv_bn(model, n_filters, kernel_size=3, init='he_uniform',
                stride=1, l2_reg=1e-3, border_mode='same',
                input_shape=(None,None)):
    '''
    Adds a block of convolution, batch normalization and relu to the model.
    '''
    model.add(Convolution2D(n_filters, kernel_size, kernel_size,
                            border_mode=border_mode, init=init,
                            subsample=(stride, stride),
                            W_regularizer=l2(l2_reg), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


def add_nin_block(model, n_filters, kernel_size=5, init='normal',
                  l2_reg=1e-4, input_shape=(None, None)):
    '''
    Adds a network in network block to the model.
    This is a block with a 5x5 convolution, followed by two 1x1 convolutions.
    All of them have relu activations.
    inputs:
    model: Keras model to which the block will be added.
    n_filters: a list with the amount of filter (kernels) of each layer.
    regs: the amount of l2 regularization on each layer.
    '''
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

### Blocks that use the functional API

### Blocks used for Densenet
def preact_layer(x, nb_channels, kernel_size=3, dropout=0., l2_reg=1e-4):
    '''
    Adds a preactivation layer for the densenet. This also includes l2
    reagularization on BatchNorm learnable parameters as in the original
    implementation.
    '''
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    out = Activation('relu')(out)
    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out

def dense_block(x, nb_layers, growth_rate, dropout=0., l2_reg=1e-4):
    '''
    Adds a dense block for the densenet.
    '''
    for i in range(nb_layers):
        # Get layer output
        out = preact_layer(x, growth_rate, dropout=dropout, l2_reg=l2_reg)
        if K.image_dim_ordering() == 'tf':
            merge_axis = -1
        elif K.image_dim_ordering() == 'th':
            merge_axis = 1
        else:
            raise Exception('Invalid dim_ordering: ' + K.image_dim_ordering())
        # Concatenate input with layer ouput
        x = merge([x, out], mode='concat', concat_axis=merge_axis)
    return x

def transition_block(x, nb_channels, dropout=0., l2_reg=1e-4):
    '''
    Adds a transition block for the densenet.
    '''
    x = preact_layer(x, nb_channels, kernel_size=1, dropout=dropout,
                     l2_reg=l2_reg)
    # x = Convolution2D(n_channels, 1, 1, border_mode='same',
    #                   init='he_normal', W_regularizer=l2(l2_reg))(x)
    x = AveragePooling2D()(x)
    return x

### Blocks used for Resnet
def bottleneck_layer(x, nb_channels, kernel_size=3, stride=1, l2_reg=1e-4,
                     first=False):
    '''
    Resnet preactivation bottleneck layer with 1x1xn, 3x3xn, 1x1x4n convolution
    layers.
    '''
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
    '''
    Regular resnet preactivation two convolution 3x3 layer.
    '''
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
    '''
    Resnet residual block. Output is the sum of the layer's output and the
    input (shortcut connection).
    '''
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
    '''
    Resnet residual block that downsamples the feature maps.
    '''
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
    '''
    Resnet block stack with residual units that share the same feature map size.
    '''
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
