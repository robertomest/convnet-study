import argparse

class ModelLoader(object):

  def __init__(self, args):
    d_args = self.default_args()
    for k, v in d_args.items():
      args[k] = args[k] or v

    # Save l2 to build model
    self.l2 = args['l2']

  def __call__(self):
    raise NotImplementedError

  def default_args(self):
    args = {
    'lr': 1e-1,
    'batch_size': 128,
    'l2': 1e-4,
    'schedule': 'step1',
    'epochs': 250
    }
    return args
