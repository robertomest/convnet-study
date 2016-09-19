from __future__ import absolute_import

from .base import ModelLoader
from ..models.cifar10 import nin_bn_model, vgg_model

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
