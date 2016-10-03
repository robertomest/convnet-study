import keras
import keras.backend as K

from keras.preprocessing.image import Iterator

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from rme.models.cifar10 import nin_bn_model
from rme.datasets import svhn
from rme.datasets.utils import one_hotify
from rme.callbacks import Step, MetaCheckpoint

import os
import numpy as np

# This iterator will get a batch, cast it to float32 and cast the labels into
# one-hot vectors. Since the database is very big, there is not enough memory
# to have it all in float32, so we have to cast it on-the-fly.
class MyIterator(Iterator):
    def __init__(self, X, y, batch_size=64, shuffle=False, seed=None):
        self.X = X
        self.y = y
        super(MyIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # Cast it as float32 and get it to range (0, 1)
        batch_x = self.X[index_array].copy().astype('float32')/255.
        batch_y = one_hotify(self.y[index_array], nb_classes=10)

        return (batch_x, batch_y)

if __name__ == '__main__':
    file_name = os.path.join('weights', 'svhn', 'ninbn')
    data_path = os.path.join('data', 'svhn')
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model = nin_bn_model() # We'll use the same model used for CIFAR-10
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define some callbacks
    nb_epochs = 40
    steps = [nb_epochs/10*i for i in range(1, 10)]
    rates = [0.1/2**i for i in range(10)]
    schedule = Step(steps, rates, verbose=1)
    model_cbk = ModelCheckpoint(file_name + '.h5')
    meta_cbk = MetaCheckpoint(file_name + '.meta', schedule=schedule)

    print('Model compiled.')
    print('Loading data...')
    train_set, valid_set, test_set = svhn.load(data_path)
    print('Data loaded.')

    it = MyIterator(train_set['data'], train_set['labels'], shuffle=True)
    # Since the validation set is small, we'll just cast it all to float32
    valid_set['data'] = valid_set['data'].astype('float32')/255.
    # Conver labels to one-hot
    valid_set['labels'] = one_hotify(valid_set['labels'])

    model.fit_generator(it,
                        samples_per_epoch=train_set['data'].shape[0],
                        nb_epoch=nb_epochs, verbose=2,
                        callbacks=[schedule, model_cbk, meta_cbk],
                        validation_data=(valid_set['data'], valid_set['labels']))
