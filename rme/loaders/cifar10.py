from __future__ import absolute_import

from .base import ModelLoader
from ..models.cifar10 import (nin_bn_model, vgg_model, densenet_model,
                              resnet_model)

class NiNBN(ModelLoader):

    def __call__(self):
        return nin_bn_model(l2_reg=self.l2)

    def default_args(self):
        args = super(NiNBN, self).default_args()
        args['batch_size'] = 64
        return args

class VGGBN(ModelLoader):

    def __call__(self):
        return vgg_model(l2_reg=self.l2)

    def default_args(self):
        args = super(VGGBN, self).default_args()
        args['batch_size'] = 128
        args['l2'] = 5e-4
        return args

class Densenet(ModelLoader):
    def __init__(self, args):
        super(Densenet, self).__init__(args)
        self.params = []
        for a in args['model'][1:]:
            try:
                b = int(a)
            except ValueError:
                b = float(a)

            self.params.append(b)

    def __call__(self):
        return densenet_model(*self.params, l2_reg=self.l2)

    def default_args(self):
        args = super(Densenet, self).default_args()
        args['batch_size'] = 64
        args['epochs'] = 300
        args['schedule'] = 'step_densenet'
        return args

class Resnet(ModelLoader):
    def __init__(self, args):
        super(Resnet, self).__init__(args)
        self.params = []
        for a in args['model'][1:]:
            try:
                b = int(a)
            except ValueError:
                b = a == 'True'

            self.params.append(b)

    def __call__(self):
        return resnet_model(*self.params)

    def default_args(self):
        args = super(Resnet, self).default_args()
        args['batch_size'] = 64
        args['epochs'] = 164
        args['schedule'] = 'step_resnet'
        return args
