### This requires KERAS 1.0.7
import argparse
import fnmatch
import pickle
import yaml

import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from rme import models
from rme import loaders
import rme.callbacks
from rme.callbacks import MetaCheckpoint
from rme.datasets.utils import load_CIFAR10

parser = argparse.ArgumentParser(description='Train a model on CIFAR-10.')
parser.add_argument('-m', '--model', help='Model loader name.', type=str,
                    required=True)
parser.add_argument('-b', '--batch_size', help='Mini-batch size.', type=int,
                    default=None)
parser.add_argument('--l2', help='l2 regularization weight.', type=float,
                    default=None)
parser.add_argument('-l', '--load', help='Load checkpoint from this file.',
                    type=str, default=None)
parser.add_argument('-s', '--save', help='Save checkpoint to this file.',
                    type=str, default=None)
parser.add_argument('--schedule',
                    help='Schedule name as defined in schedule.yaml.',
                    type=str, default=None)
parser.add_argument('--lr',
                    help='Learning rate. Ignored if schedule is passed.',
                    type=float, default=None)
parser.add_argument('-e', '--epochs', help='How many epochs to train for.\
                    This will resume from the checkpoint if one was provided.',
                    type=int, default=None)
parser.add_argument('-d', '--dataset', help='Load the dataset from a pickled\
                    file instead of using the standard function.', type=str,
                    default=None)
parser.add_argument('-v', '--valid', help='Training set ratio to be used as\
                                           validation.', type=float, default=0.)
args = parser.parse_args()

callback_list = []
schedule = None
meta_cbk = None
epoch_offset = 0

if args.load:
    print('Loading model from %s' %args.load)
    # Load model from file
    model = keras.models.load_model(args.load)
    # If no save file specified, we save on the loaded file
    if args.save is None:
        args.save = args.load

    # Recover metadata
    with open(args.load + '.meta', 'rb') as f:
        meta = yaml.load(f)
    meta_cbk = MetaCheckpoint(args.save + '.meta')
    meta_cbk.meta = meta
    callback_list.append(meta_cbk)
    epoch_offset = meta['epoch'][-1]
    schedule_config = meta.get('schedule')
    if schedule_config:
        # Recreate schedule correcting the epochs already trained
        schedule_config['steps'] = [s - epoch_offset
                                    for s in schedule_config['steps']]
        schedule_class = getattr(rme.callbacks, schedule_config['class'])
        schedule = schedule_class.from_config(schedule_config)
        callback_list.append(schedule)
else:
    # Instantiate a new model
    # Verify that the passed model exists
    try:
        loader = getattr(loaders.cifar10, args.model)
    except AttributeError:
        print '%s not found.' %args.model
        print 'Available loaders:'
        print '\n'.join(fnmatch.filter(dir(loaders.cifar10), '[!__][!ModelLoader]*'))
        raise
    model_loader = loader(args)
    model = model_loader()

training_args = vars(args)
# model.summary()

if args.schedule and schedule is None:
    # We Instantiate a new schedule if we don't already have one loaded from
    # the meta file
    with open('schedules.yaml', 'r') as f:
        schedules = yaml.load(f)
    try:
        schedule_config = schedules[args.schedule]
    except KeyError:
        print 'Schedule %s not found.' %args.schedule
        print 'Available schedules:'
        print '\n'.join(schedules.keys())
        raise
    schedule = schedule_config['class'](*schedule_config['args'])
    callback_list.append(schedule)

if args.dataset:
    try:
        with open(args.dataset, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
    except FileNotFoundError:
        raise Exception('Dataset file %s not found.' %args.dataset)
else:
    train_set, valid_set, test_set, whitening, mean = load_CIFAR10(
        'data/cifar10', valid_ratio=args.valid, gcn=True, zca=True)

# If there is no validation set, we simply print the test set performance.
if valid_set is None or valid_set['data'].size == 0:
    valid_set = test_set

# If the model has not been compiled yet, we create the optimizer and compile it
if not hasattr(model, 'optimizer'):
    # Defining optimizer
    sgd = SGD(lr=args.lr, momentum=0.9, decay=0.0, nesterov=True)

    # Compiling model
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

if args.save is None:
    args.save = 'checkpoint.h5'

callback_list.append(ModelCheckpoint(args.save))

if meta_cbk is None:
    # If we did not load it we create a MetaCheckpoint here
    meta_file_name = args.save + '.meta'
    meta_cbk = MetaCheckpoint(meta_file_name, schedule=schedule,
                              training_args=training_args)
    callback_list.append(meta_cbk)

nb_epoch = args.epochs - epoch_offset
history = model.fit(train_set['data'], train_set['labels'],
          batch_size=args.batch_size, nb_epoch=nb_epoch, verbose=2,
          validation_data=(valid_set['data'], valid_set['labels']),
          callbacks=callback_list, shuffle=True)
