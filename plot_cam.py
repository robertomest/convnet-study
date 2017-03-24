import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import keras
import keras.backend as K
from keras.models import load_model
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense

from rme.utils import config_gpu, load_meta
import rme.datasets
import rme.models
import rme.preprocessing


def apply_map(cam, img, coef):
    jet = plt.get_cmap('jet')
    cam = jet(cam) # CAM as heatmap

    if img.shape[-1] == 1: # Grayscale (MNIST)
        gray = plt.get_cmap('gray_r')
        img = gray(np.squeeze(img))
    else:
        img = img/255.

    return coef * cam[:, : , :3] + (1 - coef) * img[:, :, :3]

def maps_pred_fun(checkpoint):
    # Load model
    model = load_model(checkpoint)
    x = model.input
    # Get feature maps before GAP
    o = [l for l in model.layers if type(l) == GlobalAveragePooling2D][-1].input

    # Setup CAM
    dense_list = [l for l in model.layers if type(l) == Dense]
    num_dense = len(dense_list)
    if num_dense > 1:
        raise ValueError('Expected only one dense layer, found %d' %num_dense)
    # If there is no dense layer after (NiN), the maps are already class maps
    if num_dense: # Apply CAM if there is a dense layer
        dense_layer = dense_list[0]
        # Get dense layer weights
        W = K.get_value(dense_layer.W)[None, None] # (1, 1, ?, ?)
        b = K.get_value(dense_layer.b)

        # Transform it into a 1x1 conv
        # This convolution will map the feature maps into class 'heatmaps'
        o = Convolution2D(W.shape[-1], 1, 1, border_mode='valid', weights=[W, b])(o)

    # Resize with bilinear method
    maps = tf.image.resize_images(o, K.shape(x)[1:3], method=tf.image.ResizeMethod.BILINEAR)
    return K.function([x, K.learning_phase()], [maps, model.output])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Class Activation Maps (CAM).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint file')
    parser.add_argument('--alpha', type=float, default=0.3)
    # GPU args
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()

    config_gpu(args.gpu, args.allow_growth)
    palette = sns.color_palette()

    meta = load_meta(args.checkpoint)

    sns.set_style('whitegrid')
    # Fetch info from meta file
    dataset_name = meta['training_args']['dataset']
    arch = getattr(rme.models, meta['training_args']['architecture'])
    preprocessing = meta['training_args']['preprocessing']
    if preprocessing is None: # It used the default preprocessing
        prep_fun = arch.preprocess_data
    else:
        prep_fun = getattr(rme.preprocessing)

    # Prepare function that generates maps + predictions
    print('Preparing model...')
    get_maps = maps_pred_fun(args.checkpoint)

    # Load dataset
    print('Loading dataset %s...' %dataset_name)
    dataset = getattr(rme.datasets, dataset_name)
    train_set, valid_set, test_set = dataset.load('data/%s' %dataset_name, one_hot=False)
    data = test_set['data'].copy()
    _, _, X = prep_fun(train_set['data'], valid_set['data'], test_set['data'], dataset_name)
    # data contains the unprocessed images, X contrains preprocessed images

    idx = np.random.randint(10000, size=(10,))

    # idx = [7843, 5822, 7939, 8776, 2558, 5372, 7680, 8723, 3777, 5180, 9514, 8728, 5678, 8302, 3669, 5337, 9820]

    print('Indices: %s' %(', '.join([str(i) for i in idx])))
    imgs = data[idx]
    y = test_set['labels'][idx]
    M, P = get_maps([X[idx], 0.])

    num_classes = 10
    if dataset_name == 'cifar10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        classes = [str(i) for i in range(10)]

    for i in range(len(idx)):
        l = y[i]
        pred = P[i].argmax()
        m = M[i, :, :,pred]
        m -= m.min()
        m /= m.max()
        plt.figure()

        plt.subplot(2, 3, 1)
        plt.imshow(np.squeeze(imgs[i].astype('uint8')))
        plt.axis('off')
        plt.title('Image idx: %d' %idx[i])

        plt.subplot(2, 3, 2)
        plt.imshow(m[:, :], cmap='gray')
        plt.axis('off')
        plt.title('Predicted: %s' %classes[pred])

        plt.subplot(2, 3, 3)
        # Merge two images
        plt.imshow(apply_map(m, imgs[i], args.alpha))
        plt.axis('off')

        plt.subplot(2, 3, 4)
        height = 0.5
        pos = np.arange(num_classes) - height/2
        plt.barh(pos, P[i], height=height, color=palette[0])
        plt.barh(pos[pred], P[i][pred], height=height, color=palette[2])
        plt.barh(pos[l], P[i][l], height=height, color=palette[1])
        plt.xlim([0, 1])
        plt.ylim([-0.5, 9.5])
        plt.yticks(np.arange(10), classes)
        plt.title('Class probabilities')

        m = M[i, :, :, l]
        m -= m.min()
        m /= m.max()
        plt.subplot(2, 3, 5)
        plt.imshow(m[:, :], cmap='gray')
        plt.axis('off')
        plt.title('GT: %s' %classes[l])

        plt.subplot(2, 3, 6)
        # Merge two images
        plt.imshow(apply_map(m, imgs[i], args.alpha))
        plt.axis('off')

        print('Image index: %d' %idx[i])
        print('Class probabilities:')
        for c, p in zip(classes, P[i]):
            print(' %s: %g' %(c, p))

        plt.show()
