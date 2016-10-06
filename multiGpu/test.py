import tensorflow as tf
import sys
import numpy as np

sys.path = ['/home/robertomest/repos/keras'] + sys.path
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from rme.datasets import mnist, cifar10
from multiGpu.model import MultiGPUModel
# from rme.models.cifar10 import densenet_model
_NGPUS = 2

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# K.set_session(sess)

train_set, _, test_set, _, _, = mnist.load('data/mnist')
X, Y = train_set['data'].reshape((60000, -1)), train_set['labels']

# train_set, _, test_set = cifar10.load('data/cifar10', gcn=False, zca=False)

# X = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.1]])
# y = np.array([[0, 1], [1, 0]])

def simple_model(shape=(784,)):
    model = Sequential()
    for i in range(8):
        model.add(Dense(5000, input_shape=shape, activation='relu', name='hidden%i' %i))
    model.add(Dense(10, input_shape=shape, activation='softmax', name='output'))
    return model

def share_model(model_fun, reuse=False, device='/cpu:0'):
    with tf.device(device):
        with tf.variable_scope('shared', reuse=reuse):
            model = model_fun()
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

# model = share_model(simple_model)
# gpu_replicas = []
# for i in range(_NGPUS):
#     dev = '/gpu:' + str(i)
#     gpu_replicas.append(share_model(simple_model, reuse=True, device=dev))
# loss = []
# for m in gpu_replicas:
#     loss.append(K.mean(K.categorical_crossentropy(m.output, m.model.targets[0])))
multi_gpu = MultiGPUModel(n_gpus=2, model_fn=simple_model)
sgd = SGD(lr=0.1, momentum=0.9)
multi_gpu.compile(optimizer=sgd, loss='categorical_crossentropy')
updates = sgd.get_updates(multi_gpu.trainable_weights, {}, multi_gpu.losses)
x = X[:20000]
y = Y[:20000]
f_upd = K.function(multi_gpu.inputs + multi_gpu.targets + multi_gpu.sample_weights,
                   multi_gpu.losses, updates)
print f_upd(multi_gpu.get_inputs(x, y))
# inputs = [m.input for m in gpu_replicas]
# targets = [m.model.targets[0] for m in gpu_replicas]
# f_upd = K.function(inputs + targets, loss, updates)
#
# loss = K.mean(K.categorical_crossentropy(model.output, model.model.targets[0]))
# W_grad = K.gradients(loss, model.layers[0].W)
# f_loss = K.function([model.input, model.model.targets[0]], [loss])
# f_W = K.function([model.input, model.model.targets[0]], W_grad)
# f_treta = K.function([model.input, model.model.targets[0],
#                       model.model.sample_weights[0]], [model.model.total_loss])
# f_out = K.function([model.input], [model.output])
