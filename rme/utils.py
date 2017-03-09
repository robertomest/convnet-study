import tensorflow as tf
import keras.backend as K
import h5py
import yaml

def config_gpu(gpu, allow_growth):
    # Choosing gpu
    if gpu == '-1':
        config = tf.ConfigProto(device_count ={'GPU': 0})
    else:
        if gpu == 'all' or gpu == '':
            gpu = ''
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = gpu
    if allow_growth == True:
        config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

def load_meta(checkpoint):
    meta = {}
    with h5py.File(checkpoint, 'r') as f:
        meta_group = f['meta']
        meta['training_args'] = yaml.load(meta_group.attrs['training_args'])
        for k in meta_group.keys():
            meta[k] = list(meta_group[k])
    return meta

def parse_training_args(cli_args, default_args):
    for k, v in default_args.items():
        cli_args[k] = cli_args[k] or v

def cast_arg(v):
    'Cast argument to appropriate type'
    try:
        # Try int
        return int(v)
    except ValueError:
        pass
    try:
        # Maybe float?
        return float(v)
    except ValueError:
        pass
        # bool
    if v in ['True', 'False']:
        return v == 'True'
    # it's probably a string...
    return v


def parse_kwparams(kwparams):
    if kwparams:
        kwparams = dict([(k, cast_arg(v)) for k, v in zip(kwparams[::2], kwparams[1::2])])
        return kwparams
    else:
        return {}
