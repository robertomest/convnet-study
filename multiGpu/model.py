import tensorflow as tf

import keras.backend as K
from keras.models import Sequential, Model
from keras import optimizers, objectives
from keras.engine.training import collect_trainable_weights, slice_X

from tensorflow.python.client import device_lib

def number_of_gpus():
    l = device_lib.list_local_devices()
    return len([x.name for x in l if x.device_type=="GPU"])


class MultiGPUModel(object):
    def __init__(self, n_gpus, model_fn):
        if K._BACKEND != 'tensorflow':
            raise Exception('MultiGPUModel is only available for Tensorflow.')

        if n_gpus > number_of_gpus():
            raise ValueError('Requested %i GPUs with only %i GPUs detected.'
                             %(n_gpus, number_of_gpus()))

        self.scope = 'shared' + str(K.get_uid(prefix='shared'))
        self.model_fn = model_fn
        self.n_gpus = n_gpus
        # Create cpu model that will update the weights
        self.cpu_model = self.shared_model()
        # Create gpu models that will do computation
        self.gpu_models = []
        for i in range(self.n_gpus):
            dev = '/gpu:' + str(i)
            self.gpu_models.append(self.shared_model(reuse=True, device=dev))

    def shared_model(self, reuse=False, device='/cpu:0'):
        with tf.device(device):
            with tf.variable_scope(self.scope, reuse=reuse):
                model = self.model_fn()
                if isinstance(model, Sequential):
                    model = Model(input=model.input, output=model.output)
                    model.device=device
        return model

    def compile(self, optimizer='sgd', loss='categorical_crossentropy'):
        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)
        self.losses = []
        self.targets = []
        self.inputs = []
        self.outputs = []
        self.sample_weights = []
        self.trainable_weights = collect_trainable_weights(self.cpu_model)
        # Compile all the models
        for m in self.gpu_models:
            with tf.device(m.device):
                m.compile(optimizer=None, loss=loss)
                self.losses.append(m.total_loss)
                self.targets += m.targets
                if type(m.input) is list:
                    self.inputs += m.input
                else:
                    self.inputs.append(m.input)
                if type(m.output) is list:
                    self.outputs += m.output
                else:
                    self.outputs.append(m.output)
                self.sample_weights += m.sample_weights

    def get_inputs(self, x, y):
        x, y, w = self.gpu_models[0]._standardize_user_data(x, y)
        std_list = x + y + w
        nb_train_sample = std_list[0].shape[0]
        nb_per_gpu = nb_train_sample/self.n_gpus
        slices = []
        for i in range(self.n_gpus):
            slices.append(slice_X(std_list, start=nb_per_gpu*i,
                                  stop=nb_per_gpu*(i + 1)))
        inputs = [sl[i] for i in range(len(slices[0])) for sl in slices]
        return inputs
