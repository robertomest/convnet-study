import tensorflow as tf

import keras.backend as K
from keras.models import Sequential, Model
from keras import optimizers, objectives
from keras.engine.training import collect_trainable_weights, slice_X
from keras.layers import merge

from tensorflow.python.client import device_lib

def number_of_gpus():
    l = device_lib.list_local_devices()
    return len([x.name for x in l if x.device_type=="GPU"])

def get_layers(layers, t_layers):
    new_layers = [l for l in layers if l not in t_layers]
    t_layers += new_layers
    for l in new_layers:
        for n in l.inbound_nodes:
            t_layers = get_layers(n.inbound_layers, t_layers)
    return t_layers


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
        self.inputs = []
        self.outputs = []
        # Create cpu model that will update the weights
        self.cpu_model = self.shared_model()
        # Create gpu models that will do computation
        for i in range(self.n_gpus):
            K.reset_uids()
            dev = '/gpu:' + str(i)
            model = self.shared_model(reuse=True, device=dev)
            self.inputs += model.inputs
            self.outputs.append(model.outputs)

        outputs = [[out[i] for out in self.outputs] for i in range(len(self.outputs[0]))]
        merged = []
        with tf.device('/cpu:0'):
            for outs in outputs:
                merged.append(merge(outs, mode='concat', concat_axis=0))
        self.outputs = merged
        self.model = Model(input=self.inputs, output=self.outputs)

    def shared_model(self, reuse=False, device='/cpu:0'):
        with tf.device(device):
            with tf.variable_scope(self.scope, reuse=reuse):
                model = self.model_fn()
                if isinstance(model, Sequential):
                    model = Model(input=model.input, output=model.output)
                for l in model.layers:
                    l.name += device
                model.device=device
        return model

    def split_inputs(self, x):
        nb_train_sample = x.shape[0]
        nb_per_gpu = nb_train_sample/self.n_gpus
        inputs = []
        for i in range(self.n_gpus):
            inputs.append(x[nb_per_gpu*i:nb_per_gpu*(i + 1)])
        return inputs
