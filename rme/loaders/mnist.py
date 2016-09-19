from __future__ import absolute_import

from .base import ModelLoader
from ..models.mnist import simple_model, nin_bn_model

class Simple(ModelLoader):

    def __call__(self):
        return simple_model()

    def default_args(self):
        args = super(Simple, self).default_args()
        args['schedule'] = None
        args['lr'] = 1e-4
        args['epochs'] = 30
        return args

class NiNBN(ModelLoader):

  def __call__(self):
    return nin_bn_model(l2_reg=self.l2)

  def default_args(self):
    args = super(NiNBN, self).default_args()
    args['epochs'] = 60
    args['schedule'] = 'step4'
    return args
