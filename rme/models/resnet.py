import keras
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Convolution2D, Activation, BatchNormalization,
                          merge, GlobalAveragePooling2D, Dense, Dropout)
from keras.regularizers import l2
from rme.datasets import cifar10, cifar100, svhn, mnist, preprocessing
from rme.callbacks import Step


def bottleneck_layer(x, num_channels, kernel_size, l2_reg, stride=1,
                     first=False, name=''):
    '''
    Resnet preactivation bottleneck layer with 1x1xn, 3x3xn, 1x1x4n convolution
    layers.
    '''
    if first: # Skip BN-Relu
        out = x
    else:
        out = BatchNormalization(name=name + '_bn1')(x)
        out = Activation('relu', name=name + '_relu1')(out)
    # Apply the bottleneck convolution
    out = Convolution2D(num_channels, 1, 1,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False,
                        name=name + '_conv1')(out)
    # 3x3 conv with bottlenecked channels
    # We stride it on 3x3 conv as done on Facebook's implementation
    out = BatchNormalization(name=name + '_bn2')(out)
    out = Activation('relu', name=name + '_relu2')(out)
    out = Convolution2D(num_channels, kernel_size, kernel_size,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False,
                        name=name + '_conv2')(out)
    out = BatchNormalization(name=name + '_bn3')(out)
    out = Activation('relu', name=name + '_relu3')(out)
    # 1x1 conv that expands the number of channels
    out = Convolution2D(num_channels*4, 1, 1,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False,
                        name=name + '_conv3')(out)
    return out


def two_conv_layer(x, num_channels, kernel_size, l2_reg, stride=1,
                   first=False, name=''):
    '''
    Regular resnet preactivation two convolution 3x3 layer.
    '''
    if first: # Skip BN-Relu
        out = x
    else:
        out = BatchNormalization(name=name + '_bn1')(x)
        out = Activation('relu', name=name + '_relu1')(out)
    out = Convolution2D(num_channels, kernel_size, kernel_size,
                        subsample=(stride, stride),
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False,
                        name=name + '_conv1')(out)
    out = BatchNormalization(name=name + '_bn2')(out)
    out = Activation('relu', name=name + '_relu2')(out)
    out = Convolution2D(num_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False,
                        name=name + '_conv2')(out)
    return out


def residual_block(x, num_channels, kernel_size, l2_reg, bottleneck, stride=1,
                   first=False, name=''):
    '''
    Resnet residual block. Output is the sum of the layer's output and the
    input (shortcut connection).
    '''
    if bottleneck:
        out = bottleneck_layer(x, num_channels, kernel_size, l2_reg,
                               stride=stride, first=first, name=name)
        # if first:
        #     # Shortcut needs mapping for the first bottleneck layer
        #     x = Convolution2D(num_channels * 4, 1, 1,
        #                       border_mode='valid', init='he_normal',
        #                       W_regularizer=l2(l2_reg), bias=False,
        #                       name=name + '_shortcut_proj')(x)
    else:
        out = two_conv_layer(x, num_channels, kernel_size, l2_reg,
                             stride=stride, first=first, name=name)

    out_shape = K.int_shape(out)
    if out_shape == K.int_shape(x): # Identity mapping
        shortcut = x
    else: # If dimensions change, we project the input to the new size
        if first:
            # Do not apply BN-ReLU
            shortcut = x
        else:
            shortcut = BatchNormalization(name=name + '_shortcut_bn')(x)
            shortcut = Activation('relu', name=name + '_shortcut_relu')(shortcut)
        shortcut = Convolution2D(out_shape[-1], 1, 1, subsample=(stride, stride),
                                 border_mode='valid',
                                 init='he_normal', W_regularizer=l2(l2_reg),
                                 bias=False, name=name + '_shortcut_conv')(shortcut)

    out = merge([shortcut, out], mode='sum', name=name + '_sum')
    return out


def downsample_block(x, num_channels, kernel_size, l2_reg, bottleneck,
                     name=''):
    '''
    Resnet residual block that downsamples the feature maps.
    '''
    # Perform pre-activation for both the residual and the projection
    x = BatchNormalization(name=name+'_shared_bn')(x)
    x = Activation('relu', name=name+'_shared_relu')(x)

    if bottleneck:
        out = bottleneck_layer(x, num_channels, kernel_size, l2_reg,
                               stride=2, first=True, name=name)
        # The output channels is 4x bigger on this case
        num_channels = num_channels * 4
    else:
        out = two_conv_layer(x, num_channels, kernel_size, l2_reg,
                             stride=2, first=True, name=name)
    # Projection on the shortcut
    # Pre-activated conv
    proj = Convolution2D(num_channels, 1, 1, subsample=(2, 2),
                         border_mode='valid', init='he_normal',
                         W_regularizer=l2(l2_reg), bias=False,
                         name=name + '_shortcut_proj')(x)
    # proj = AveragePooling2D((1, 1), (2, 2))(x)
    out = merge([proj, out], mode='sum', name=name + '_sum')
    return out


def block_stack(x, num_channels, num_blocks, kernel_size, l2_reg, bottleneck,
                first=False, name=''):
    '''
    Resnet block stack with residual units that share the same feature map size.
    '''
    if first:
        x = residual_block(x, num_channels, kernel_size, l2_reg, bottleneck,
                           first=True, name=name + '_resblock1')
    else:
        x = residual_block(x, num_channels, kernel_size, l2_reg, bottleneck,
                           stride=2, name=name + '_downsample')
    for i in range(num_blocks-1):
        x = residual_block(x, num_channels, kernel_size, l2_reg, bottleneck,
                           name=name + '_resblock%d' %(i + 2))
    return x


def model(dataset, num_blocks=18, width=1, bottleneck=True, l2_reg=1e-4):
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
    num_channels = [16*width, 32*width, 64*width]

    if dataset == 'cifar10':
        x = Input((32, 32, 3))
    else:
        raise ValueError('Model is not defined for dataset: %s' %dataset)

    o = Convolution2D(16, 3, 3, border_mode='same', init='he_normal',
                      W_regularizer=l2(l2_reg), bias=False)(x)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    for i, (n, f) in enumerate(zip(num_channels, [True, False, False])):
        o = block_stack(o, n, num_blocks, 3, l2_reg, bottleneck,
                        first=f, name='stack%d' %(i+1))
    # Last BN-Relu
    o = BatchNormalization(name='last_bn')(o)
    o = Activation('relu', name='last_relu')(o)
    o = GlobalAveragePooling2D()(o)
    o = Dense(10)(o)
    o = Activation('softmax')(o)

    return Model(input=x, output=o)


def preprocess_data(train_set, valid_set, test_set, dataset):
    if dataset == 'cifar10':
        train_set = cifar10.preprocess(train_set)
        valid_set = cifar10.preprocess(valid_set)
        test_set = cifar10.preprocess(test_set)
    else:
        raise ValueError('Preprocessing not defined for dataset: %s' %dataset)

    return train_set, valid_set, test_set


def default_args(dataset):
    training_args = {}
    if dataset == 'cifar10':
        training_args['lr'] = 0.1
        training_args['epochs'] = 164
        training_args['batch_size'] = 64
    else:
        print('Default args not defined for dataset: %s' %dataset)

    return training_args

def schedule(dataset, lr):
    if dataset == 'cifar10':
        steps = [82, 123]
        lrs = [lr, lr/10, lr/100]
    else:
        raise ValueError('Schedule not defined for dataset: %s' %dataset)
    return Step(steps, lrs)
