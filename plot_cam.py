import argparse

import numpy as np
import matplotlib.pyplot as plt
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
    # GPU args
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--allow_growth', default=False, action='store_true')

    args = parser.parse_args()

    config_gpu(args.gpu, args.allow_growth)

    meta = load_meta(args.checkpoint)
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

    # idx = np.random.randint(10000, size=(10,))
    # idx = [2625, 4619, 7511, 5822, 7939, 5949, 2396, 8776, 8832, 2558]
    idx = [7843, 5822, 7939, 8776, 2558, 5372, 7680, 8723, 3777, 5180]
    # GOOD INDICES: [7843, 5822, 7939, 8776, 2558, 5372, 7680, 8723, 3777, 5180]
    # 2625, 4619, 7511, 5822, 7939, 5949, 2396, 8776, 8832, 2558
    # PLANE WITH DISTRACTION: 3155, 4605, 5778, 9023, 5805, 1108, 5089, 1083, 7843, 7833
    print('Indices: %s' %(', '.join([str(i) for i in idx])))
    imgs = data[idx]
    y = test_set['labels'][idx]
    M, P = get_maps([X[idx], 0.])

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    for i in range(len(idx)):
        l = y[i]
        m = M[i, :, :, P[i].argmax()]
        m -= m.min()
        m /= m.max()
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('GT: %s' %classes[l])
        plt.imshow(np.squeeze(imgs[i].astype('uint8')))

        plt.subplot(1, 3, 2)
        plt.imshow(m[:, :], cmap='gray')
        plt.title('Predicted class: %s' %classes[P[i].argmax()])
        plt.subplot(1, 3, 3)
        # Merge two images
        plt.title('Image idx: %d' %idx[i])
        plt.imshow(apply_map(m, imgs[i], 0.4))
        plt.show()
