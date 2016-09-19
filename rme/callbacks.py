import keras.backend as K
from keras.callbacks import Callback
import yaml

class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))

class TriangularCLR(Callback):

    def __init__(self, learning_rates, half_cycle):
        self.lr = learning_rates
        self.hc = half_cycle

    def on_train_begin(self, logs={}):
        # Setup an iteration counter
        self.itr = -1

    def on_batch_begin(self, batch, logs={}):
        self.itr += 1
        cycle = 1 + self.itr/int(2*self.hc)
        x = self.itr - (2.*cycle - 1)*self.hc
        x /= self.hc
        new_lr = self.lr[0] + (self.lr[1] - self.lr[0])*(1 - abs(x))/cycle

        K.set_value(self.model.optimizer.lr, new_lr)


class MetaCheckpoint(Callback):
    '''
    Checkpoints some training information on a meta file. Together with the
    Keras model saving, this should enable resuming training and having training
    information on every checkpoint.
    '''

    def __init__(self, filepath, schedule=None, training_args=None):
        self.filepath = filepath
        self.meta = {'epoch': []}
        if schedule:
            self.meta['schedule'] = schedule.get_config()
        if training_args:
            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        self.epoch_offset = len(self.meta['epoch'])

    def on_epoch_end(self, epoch, logs={}):
        # Get statistics
        self.meta['epoch'].append(epoch + self.epoch_offset)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        with open(filepath, 'wb') as f:
            yaml.dump(self.meta, f)
