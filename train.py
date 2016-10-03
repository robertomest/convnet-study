### This requires KERAS 1.0.7
import argparse
import fnmatch
import pickle
import yaml
import os

import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from rme import models
from rme import loaders
from rme import datasets
import rme.callbacks
from rme.callbacks import MetaCheckpoint

parser = argparse.ArgumentParser(description='Train a model on CIFAR-10.')
parser.add_argument('-m', '--model', help='Model loader name.', type=str,
                    nargs='+', default=None)
parser.add_argument('-b', '--batch_size', help='Mini-batch size.', type=int,
                    default=None)
parser.add_argument('--l2', help='l2 regularization weight.', type=float,
                    default=None)
parser.add_argument('-l', '--load', help='Load checkpoint from this file.',
                    type=str, default=None)
parser.add_argument('-s', '--save', help='Save checkpoint to this file.',
                    type=str, default=None)
parser.add_argument('--schedule',
                    help='Learning rate schedule name as defined in\
                    schedule.yaml.',
                    type=str, default=None)
parser.add_argument('--lr',
                    help='Learning rate. Ignored if schedule is passed.',
                    type=float, default=None)
parser.add_argument('-e', '--epochs', help='How many epochs to train for.\
                    This will resume from the checkpoint if one was provided.',
                    type=int, default=None)
parser.add_argument('-d', '--dataset', help='Choose the dataset by name.',
                    type=str, default=None)
parser.add_argument('-v', '--valid', help='Training set ratio to be used as\
                                           validation.', type=float, default=0.)
args = parser.parse_args()
args = vars(args)
callback_list = []
schedule = None
meta_cbk = None
epoch_offset = 0
default_save = 'checkpoint'

def clean_dir_list(dir_list):
    blacklist = ['absolute_import', 'utils', 'base', 'ModelLoader']
    new_list = fnmatch.filter(dir_list, '[!__]*')
    new_list = fnmatch.filter(dir_list, '*[!_model]')
    new_list = [it for it in new_list if it not in blacklist]
    return new_list

## TODO: For now if you load a model, it will continue to train it as it was
######   when it was saved. We should be able to redefine most parameters if
######   we wanted
if args['load']:
    # Load checkpointed data and continue training
    file_name = args['load'] + '.h5'
    meta_file_name = args['load'] + '.meta'
    # Load model from file
    model = keras.models.load_model(file_name)
    # Recover metadata
    with open(meta_file_name, 'rb') as f:
        meta = yaml.load(f)
    # meta contains information about arguments, schedule and training statistics
    # Offset the number of epochs we already trained for.
    epoch_offset = len(meta['epoch'])
    # Restore args
    args = meta['training_args']

    # Load schedule from config
    schedule_config = meta.get('schedule')
    if schedule_config:
        schedule_config['epoch_offset'] = epoch_offset
        schedule_class = getattr(rme.callbacks, schedule_config['class'])
        schedule = schedule_class.from_config(schedule_config)
        callback_list.append(schedule)

    # Instantiate ModelCheckpoint callback
    model_cbk = ModelCheckpoint(file_name)
    callback_list.append(model_cbk)

    # Instantiate MetaCheckpoint callback and restore meta
    meta_cbk = MetaCheckpoint(meta_file_name)
    meta_cbk.meta = meta
    callback_list.append(meta_cbk)

else:
    if args['model'] is None:
        raise Exception('You must provide a model loader or a checkpoint.')
    if args['dataset'] is None:
        raise Exception('You must provide a dataset or a checkpoint.')

    try:
        loaders_module = getattr(loaders, args['dataset'])
    except AttributeError:
        datasets_list = clean_dir_list(dir(loaders))
        raise Exception(('Dataset %s is not available. '
                         'Available datasets are:\n%s.')
                        %(args['dataset'], ', '.join(datasets_list)))
    ## Instantiate a new model
    # Verify that the passed model exists
    model_name = args['model'][0]
    try:
        loader = getattr(loaders_module, model_name)
    except AttributeError:
        loader_list = clean_dir_list(dir(loaders_module))
        raise Exception('Loader %s not found. Available loaders are: %s.'
                        %(model_name, ', '.join(loader_list)))

    # Get default args from the ModelLoader
    model_loader = loader(args)
    # Instantiate the model
    model = model_loader()
    # Defining optimizer
    sgd = SGD(lr=args['lr'], momentum=0.9, decay=0.0, nesterov=True)
    # Compiling model
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ## Instantiate the callbacks
    # Schedule callback
    schedule_cbk = None
    if args['schedule']:
        # Get schedule callback if there is a schedule
        with open('schedules.yaml', 'r') as f:
            schedules = yaml.load(f)
        try:
            schedule_config = schedules[args['schedule']]
        except KeyError:
            raise Exception('Schedule %s not found. Available schedules are:\n%s'
                           %(args['schedule'], ', '.join(schedules.keys())))
        schedule_cbk = schedule_config['class'](*schedule_config['args'], verbose=1)
        callback_list.append(schedule_cbk)

    # ModelCheckpoint callback
    if args['save'] is None:
        args['save'] = default_save
    file_name = args['save'] + '.h5'
    model_cbk = ModelCheckpoint(file_name)
    callback_list.append(model_cbk)

    # MetaCheckpoint callback
    meta_file_name = args['save'] + '.meta'
    meta_cbk = MetaCheckpoint(meta_file_name, schedule=schedule_cbk,
                              training_args=args)
    callback_list.append(meta_cbk)

try:
    dataset = getattr(datasets, args['dataset'])
except AttributeError:
    datasets_list = clean_dir_list(dir(datasets))
    raise Exception(('Dataset %s is not available. '
                     'Available datasets are:\n%s.')
                    %(args['dataset'], ', '.join(datasets_list)))

train_set, valid_set, test_set, _, _ = dataset.load(os.path.join('data',
                                                    args['dataset']))

if valid_set is None or valid_set['data'].size == 0:
    valid_set = test_set

nb_epoch = args['epochs'] - epoch_offset

model.fit(train_set['data'], train_set['labels'],
          batch_size=args['batch_size'], nb_epoch=nb_epoch, verbose=2,
          validation_data=(valid_set['data'], valid_set['labels']),
          callbacks=callback_list, shuffle=True)
