import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.callbacks import History

from rme.datasets import mnist
from rme.models.mnist import simple_model
from rme.callbacks import Step

# Load our data.
train_set, valid_set, test_set, _, _ = mnist.load('data/mnist',
                                       valid_ratio=1./6, normalize=True)

# Load our model. This is a simple model taken from tensorflow's tutorial
model = simple_model()

# Use the adam optimizer, since we are using a schedule callback. The
# learning rate we define here doesn't really matter
adam = optimizers.Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback that implements learning rate schedule
schedule = Step([20], [1e-4, 1e-5])


history = model.fit(train_set['data'], train_set['labels'],
    batch_size=128, nb_epoch=40, validation_data=(valid_set['data'],
    valid_set['labels']), callbacks=[schedule], verbose=2, shuffle=True)

(loss, acc) = model.evaluate(test_set['data'], test_set['labels'],
                             batch_size=128)
print('Test set loss = %g, accuracy = %g' %(loss, acc))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training set accuracy', 'test set accuracy'])
plt.show()
