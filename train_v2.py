### This requires KERAS 1.0.7
import argparse
import os

import tensorflow as tf
import keras
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import rme.models
from rme.utils import config_gpu, load_meta, parse_training_args, parse_kwparams
from rme import datasets
from rme.callbacks import MetaCheckpoint
from rme import schedules
from rme import preprocessing

if __name__ == '__main__':

    available_archs = {'nin': rme.models.nin, 'baseline': rme.models.baseline}

    parser = argparse.ArgumentParser(description='Train a model on the desired dataset.')
    parser.add_argument('--architecture', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_checkpoint', type=str, default='checkpoint.h5')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    # Hyperparameters
    parser.add_argument('--kwparams', type=str, nargs='+', default=None)
    # Training args
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--schedule', type=str, default=None)
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--augmented', default=False, action='store_true')
    # GPU args
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()

    config_gpu(args.gpu, args.allow_growth)

    training_args = vars(args)

    if args.load_checkpoint:
        # Continue training
        model = load_model(args.load_checkpoint)
        meta = load_meta(args.load_checkpoint)
        args.dataset = meta['training_args']['dataset']
        arch = available_archs[meta['training_args']['architecture']]
        training_args = meta['training_args']
        chkpt_cbk = MetaCheckpoint(args.save_checkpoint, meta=meta)
        initial_epoch = meta['epochs'][-1] + 1
    else:
        try:
            arch = getattr(rme.models, args.architecture)
            # arch = available_archs[args.architecture]
        except KeyError as e:
            raise ValueError('Architecture %s is not available.' %args.architecture)

        parse_training_args(training_args, arch.default_args(args.dataset))
        training_args['kwparams'] = parse_kwparams(args.kwparams)

        chkpt_cbk = MetaCheckpoint(args.save_checkpoint, training_args=training_args)

        model = arch.model(args.dataset, **training_args['kwparams'])
        opt = SGD(lr=training_args['lr'], momentum=0.9, nesterov=True)

        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        initial_epoch = 0

    # Load dataset
    print('Loading dataset: %s' %training_args['dataset'])
    if args.dataset == 'mnist':
        train_set, valid_set, test_set = datasets.mnist.load('data/mnist')
    elif args.dataset == 'cifar10':
        train_set, valid_set, test_set = datasets.cifar10.load('data/cifar10')
    elif args.dataset == 'cifar100':
        train_set, valid_set, test_set = datasets.cifar100.load('data/cifar100')
    elif args.dataset == 'svhn':
        train_set, valid_set, test_set = datasets.svhn.load('data/svhn')
    else:
        raise NotImplementedError('Dataset %s is not available.' %training_args['dataset'])

    # Preprocess it
    print('Preprocessing dataset: %s' %training_args['dataset'])
    if training_args['preprocessing']:
        try:
            preprocess_fun = getattr(rme.preprocessing, training_args['preprocessing'])
            print('Using custom preprocessing: %s' %training_args['preprocessing'])
        except AttributeError:
            raise NotImplementedError('Preprocessing %s is not availabe' %training_args['preprocessing'])
    else:
        print('Using standard preprocessing for architecture %s' %training_args['architecture'])
        preprocess_fun = arch.preprocess_data

    (train_set['data'], valid_set['data'],
     test_set['data']) = preprocess_fun(train_set['data'],
                                        valid_set['data'],
                                        test_set['data'], args.dataset)

    callbacks = [chkpt_cbk]

    if valid_set is None or valid_set['data'].size == 0:
        print('No validation set, using test set as validation data.')
        validation_data = (test_set['data'], test_set['labels'])
    else:
        chkpt_path, chkpt_name = os.path.split(training_args['save_checkpoint'])
        best_model_name = os.path.join(chkpt_path, 'best_' + chkpt_name)
        print('Saving model with best validation accuracy with name %s.'
              %best_model_name)
        best_cbk = MetaCheckpoint(best_model_name, save_best_only=True,
                                   training_args=training_args)
        validation_data = (valid_set['data'], valid_set['labels'])
        # Append it to callbacks list
        callbacks.append(best_cbk)

    if training_args['schedule'] != 'none':
        # Set learning rate schedule
        if training_args['schedule'] is None:
            # Use default
            schedule_fun = arch.schedule
        else:
            try:
                schedule_fun = getattr(rme.schedules, training_args['schedule'])
            except AttributeError:
                raise NotImplementedError('Schedule %s is not availabe' %training_args['schedule'])
            # raise NotImplementedError('You should implement custom schedules.')
        schedule = schedule_fun(training_args['dataset'], training_args['lr'])
        callbacks.append(schedule)
    else:
        # Use fixed learning rate
        print('No learning rate scheduling. Learning rate will be constant')


    print('Training with:')
    print('%s' %str(training_args))

    if training_args['augmented']:
        print('Training with data augmentation: crops and flips.')
        data_gen = ImageDataGenerator(horizontal_flip=True,
                                      width_shift_range=0.125,
                                      height_shift_range=0.125,
                                      fill_mode='constant')
        data_iter = data_gen.flow(train_set['data'], train_set['labels'],
                                  batch_size=training_args['batch_size'],
                                  shuffle=True)

        model.fit_generator(data_iter,
                            samples_per_epoch=train_set['data'].shape[0],
                            nb_epoch=training_args['epochs'],
                            validation_data=(test_set['data'],
                                             test_set['labels']),
                            callbacks=callbacks, initial_epoch=initial_epoch)
    else:
        model.fit(train_set['data'], train_set['labels'],
                  batch_size=training_args['batch_size'],
                  nb_epoch=training_args['epochs'],
                  validation_data=validation_data,
                  callbacks=callbacks, initial_epoch=initial_epoch,
                  shuffle=True)

    test_loss, test_acc = model.evaluate(test_set['data'], test_set['labels'],
                                         verbose=2)
    print('Test set loss = %g. Test set accuracy = %g' %(test_loss, test_acc))
