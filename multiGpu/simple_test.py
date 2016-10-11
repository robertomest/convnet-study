import keras
import keras.backend as K
from keras.layers import Input, InputLayer
from keras.models import Model, Sequential
from keras.optimizers import SGD
import tensorflow as tf

from rme.models.cifar10 import densenet_model, nin_model
from rme.datasets import cifar10
import numpy as np

####
from keras.layers import merge
from keras.layers.core import Lambda

def make_parallel(model_fn, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1]/parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1]/parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    scope = 'shared' + str(K.get_uid(prefix='shared'))
    def shared_model(model_fn, reuse=False, device='/cpu:0', input_layer=None):
        with tf.device(device):
            with tf.variable_scope(scope, reuse=reuse):
                model = model_fn(input_layer)
                if isinstance(model, Sequential):
                    model = Model(input=model.input, output=model.output)
                # model.device=device
        return model

    def get_layers(layers, t_layers):
        new_layers = [l for l in layers if (l not in t_layers and type(l) is not InputLayer)]
        t_layers += new_layers
        for l in new_layers:
            for n in l.inbound_nodes:
                t_layers = get_layers(n.inbound_layers, t_layers)
        return t_layers

    outputs_all = []
    cpu_model = shared_model(model_fn) # Allocate one model on the cpu
    for i in xrange(len(cpu_model.outputs)):
        outputs_all.append([])
    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in xrange(gpu_count):
        dev = '/gpu:%d' %i
        with tf.device(dev):
            inputs = []
            #Slice each input into a piece for processing on this GPU
            for x in cpu_model.inputs:
                input_shape = tuple(x.get_shape().as_list())[1:]
                slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                inputs.append(slice_n)
            outputs = shared_model(model_fn, reuse=True, device=dev,
                                   input_layer=inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Change layer names so we can merge the model later
            layers = []
            # Get all model layers
            for out in outputs:
                l, _, _ = out._keras_history
                layers.append(l)
            t_layers = get_layers(layers, [])
            # Change their names
            for l in t_layers:
                l.name += dev

            #Save all the outputs for merging back together later
            for l in xrange(len(outputs)):
                outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

    return Model(input=cpu_model.inputs, output=merged)

####
_NGPUS = 2
from keras.layers import Dense, Activation
def simple_model(input_layer=None):
    if input_layer:
        x = input_layer
    else:
        inp = Input(shape=(3072,))
        x = inp
    for i in range(8):
        x = Dense(5000, name='hidden%i' %i)(x)
        x = Activation('relu', name='relu%i' %i)(x)
    x = Dense(10, activation='softmax', name='output')(x)
    if input_layer is None:
        return Model(inp, x)
    return x
# with tf.device('/cpu:0'):
#     cpu_model = densenet_model(3, 12, 12)
# inputs = []
# outputs = []
# for i in range(_NGPUS):
#     with tf.device('/gpu:%d' %i):
#         inputs.append(Input(shape=(32, 32, 3)))
#         outputs.append(cpu_model(inputs[-1]))
# model = densenet_model(3, 12, 12)
# model = nin_model()
# model_fn = lambda input_layer: densenet_model(3, 32, 12, input_layer=input_layer)
model_fn = simple_model
model = make_parallel(model_fn, 2)

# model = Model(input=inputs, output=outputs)
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_data(data_set):
    mean = np.array([125.3, 123.0, 113.9])
    std = np.array([63.0, 62.1, 66.7])

    data_set -= mean
    data_set /= std
    return data_set

train_set, _, test_set = cifar10.load('data/cifar10', zca=False, gcn=False)
# X = [train_set['data'][:25000], train_set['data'][25000:]]
# y = [train_set['labels'][:25000], train_set['labels'][25000:]]
# X_v = [test_set['data'][:5000], test_set['data'][5000:]]
# y_v = [test_set['labels'][:5000], test_set['labels'][5000:]]
X = np.reshape(preprocess_data(train_set['data']), (-1, 3072))
y = train_set['labels']
X_v = np.reshape(preprocess_data(test_set['data']), (-1, 3072))
y_v = test_set['labels']
model.fit(X, y, nb_epoch=2, batch_size=1000,
          validation_data=(X_v, y_v), shuffle=True)
