import copy
import numpy as np
from keras import models
from keras import callbacks as cbks
from keras.engine.training import make_batches, slice_X
import keras.backend as K

class Model(models.Model):
    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, start_epoch=0):
        '''
        Adaptation of the Keras fit call that makes it easier to resume
        training without changing learning rate schedules.
        '''
        # validate user data
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           class_weight=class_weight,
                                                           check_batch_dim=False,
                                                           batch_size=batch_size)
        # prepare validation data
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise
            val_x, val_y, val_sample_weights = self._standardize_user_data(val_x, val_y,
                                                                           sample_weight=val_sample_weight,
                                                                           check_batch_dim=False,
                                                                           batch_size=batch_size)
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase and type(K.learning_phase()) is not int:
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
            y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
            sample_weights, val_sample_weights = (
                slice_X(sample_weights, 0, split_at), slice_X(sample_weights, split_at))
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase and type(K.learning_phase()) is not int:
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights
        else:
            do_validation = False
            val_f = None
            val_ins = None

        # prepare input arrays and training function
        if self.uses_learning_phase and type(K.learning_phase()) is not int:
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        self._make_train_function()
        f = self.train_function

        # prepare display labels
        out_labels = self.metrics_names

        # rename duplicated metrics name
        # (can happen with an output layer shared among multiple dataflows)
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label)
                new_label += '_' + str(dup_idx + 1)
            deduped_out_labels.append(new_label)
        out_labels = deduped_out_labels

        if do_validation:
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        # delegate logic to _fit_loop
        return self._fit_loop(f, ins, out_labels=out_labels,
                              batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              start_epoch=start_epoch)

    def _fit_loop(self, f, ins, out_labels=[], batch_size=32,
                      nb_epoch=100, verbose=1, callbacks=[],
                      val_f=None, val_ins=None, shuffle=True,
                      callback_metrics=[], start_epoch=0):
            '''Abstract fit function for f(ins).
            Assume that f returns a list, labeled by out_labels.
            # Arguments
                f: Keras function returning a list of tensors
                ins: list of tensors to be fed to `f`
                out_labels: list of strings, display names of
                    the outputs of `f`
                batch_size: integer batch size
                nb_epoch: number of times to iterate over the data
                verbose: verbosity mode, 0, 1 or 2
                callbacks: list of callbacks to be called during training
                val_f: Keras function to call for validation
                val_ins: list of tensors to be fed to `val_f`
                shuffle: whether to shuffle the data at the beginning of each epoch
                callback_metrics: list of strings, the display names of the metrics
                    passed to the callbacks. They should be the
                    concatenation of list the display names of the outputs of
                     `f` and the list of display names of the outputs of `f_val`.
            # Returns
                `History` object.
            '''
            do_validation = False
            if val_f and val_ins:
                do_validation = True
                if verbose:
                    print('Train on %d samples, validate on %d samples' %
                          (ins[0].shape[0], val_ins[0].shape[0]))

            nb_train_sample = ins[0].shape[0]
            index_array = np.arange(nb_train_sample)

            self.history = cbks.History()
            callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
            if verbose:
                callbacks += [cbks.ProgbarLogger()]
            callbacks = cbks.CallbackList(callbacks)

            # it's possible to callback a different model than self
            # (used by Sequential models)
            if hasattr(self, 'callback_model') and self.callback_model:
                callback_model = self.callback_model
            else:
                callback_model = self

            callbacks._set_model(callback_model)
            callbacks._set_params({
                'batch_size': batch_size,
                'nb_epoch': nb_epoch,
                'nb_sample': nb_train_sample,
                'verbose': verbose,
                'do_validation': do_validation,
                'metrics': callback_metrics,
            })
            callbacks.on_train_begin()
            callback_model.stop_training = False
            self.validation_data = val_ins

            for epoch in range(start_epoch, nb_epoch):
                callbacks.on_epoch_begin(epoch)
                if shuffle == 'batch':
                    index_array = batch_shuffle(index_array, batch_size)
                elif shuffle:
                    np.random.shuffle(index_array)

                batches = make_batches(nb_train_sample, batch_size)
                epoch_logs = {}
                for batch_index, (batch_start, batch_end) in enumerate(batches):
                    batch_ids = index_array[batch_start:batch_end]
                    try:
                        if type(ins[-1]) is float:
                            # do not slice the training phase flag
                            ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                        else:
                            ins_batch = slice_X(ins, batch_ids)
                    except TypeError:
                        raise Exception('TypeError while preparing batch. '
                                        'If using HDF5 input data, '
                                        'pass shuffle="batch".')
                    batch_logs = {}
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = len(batch_ids)
                    callbacks.on_batch_begin(batch_index, batch_logs)
                    outs = f(ins_batch)
                    if type(outs) != list:
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)

                    if batch_index == len(batches) - 1:  # last batch
                        # validation
                        if do_validation:
                            # replace with self._evaluate
                            val_outs = self._test_loop(val_f, val_ins,
                                                       batch_size=batch_size,
                                                       verbose=0)
                            if type(val_outs) != list:
                                val_outs = [val_outs]
                            # same labels assumed
                            for l, o in zip(out_labels, val_outs):
                                epoch_logs['val_' + l] = o
                callbacks.on_epoch_end(epoch, epoch_logs)
                if callback_model.stop_training:
                    break
            callbacks.on_train_end()
            return self.history


class Sequential(models.Sequential):

    def build(self, input_shape=None):
        if not self.inputs or not self.outputs:
            raise Exception('Sequential model cannot be built: model is empty.'
                            ' Add some layers first.')
        # actually create the model
        self.model = Model(self.inputs, self.outputs[0], name=self.name + '_model')

        # mirror model attributes
        self.supports_masking = self.model.supports_masking
        self._output_mask_cache = self.model._output_mask_cache
        self._output_tensor_cache = self.model._output_tensor_cache
        self._output_shape_cache = self.model._output_shape_cache
        self.input_layers = self.model.input_layers
        self.input_layers_node_indices = self.model.input_layers_node_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_node_indices = self.model.output_layers_node_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self.nodes_by_depth = self.model.nodes_by_depth
        self.container_nodes = self.model.container_nodes
        self.output_names = self.model.output_names
        self.input_names = self.model.input_names

        # make sure child model callbacks will call the parent Sequential model:
        self.model.callback_model = self

        self.built = True

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, start_epoch=0, **kwargs):
        '''
        Adaptation of the Keras fit call that makes it easier to resume
        training without changing learning rate schedules.
        '''
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              nb_epoch=nb_epoch,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              sample_weight=sample_weight,
                              start_epoch=start_epoch)
