import rme.datasets
import numpy as np
from scipy.misc import imsave

if __name__ == '__main__':
    margins = [0, 1, 1]
    datasets = ['mnist', 'cifar10', 'svhn']

    for dataset, margin in zip(datasets, margins):
        print('processing %s' %dataset)
        module = getattr(rme.datasets, dataset)
        print('loading...')
        train_set, _, _ = module.load('data/%s' %dataset, one_hot=False, dtype='uint8')
        print('loaded.')
        imgs = np.vstack([train_set['data'][train_set['labels'] == i][:1] for i in range(10)])
        N = imgs.shape[1]
        #2 x 5 panel
        H = 2 * N + margin
        W = 5 * N + 4 * margin
        panel = (255 * np.ones((H, W, imgs.shape[-1]))).astype('uint8')
        for idx, img in enumerate(imgs):
            w = (idx % 5) * (N + margin)
            h = int(idx > 4) * (N + margin)
            if dataset == 'mnist':
                img = 255 - img
            panel[h:h+N, w:w+N, :] = img
        imsave('%s.png' %dataset, np.squeeze(panel))
