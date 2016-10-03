import keras
from keras.models import Model
from keras.preprocessing.image import Iterator
from keras.layers import Flatten

from rme.datasets import svhn
from rme.datasets.utils import one_hotify

import os

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
    file_name = os.path.join('weights', 'svhn', 'ninbn.h5')
    data_path = os.path.join('data', 'svhn')
    model = keras.models.load_model(file_name)

    print('Loading data...')
    train_set, valid_set, test_set = svhn.load(data_path)
    print('Data loaded.')

    it = MyIterator(train_set['data'], train_set['labels'], shuffle=True)
    # Since the validation set is small, we'll just cast it all to float32
    valid_set['data'] = valid_set['data'].astype('float32')/255.
    # Conver labels to one-hot
    valid_set['labels'] = one_hotify(valid_set['labels'])
    test_set['data'] = test_set['data'].astype('float32')/255.
    test_set['labels'] = one_hotify(test_set['labels'])
    last_conv = model.layers[-4].output
    features = Flatten()(last_conv)
    model = Model(input=model.input, output=features)
