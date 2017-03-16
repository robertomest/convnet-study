from rme.utils import load_meta
import numpy as np
import argparse
import matplotlib

if __name__ == '__main__':
    label = {'loss': 'Loss', 'acc': 'Accuracy', 'err': 'Error'}
    parser = argparse.ArgumentParser(description='Plot training curves.')
    parser.add_argument('--checkpoints', type=str, nargs='+')
    parser.add_argument('--metric', type=str, default='err',
                         choices=['loss', 'acc', 'err'])
    parser.add_argument('--arch_names', type=str, nargs='+')
    parser.add_argument('--metric_names', type=str, nargs='+', default=['training error', 'validation error'])
    parser.add_argument('--save', type=str, default=None)

    args = parser.parse_args()

    if args.arch_names is None:
        args.arch_names = ['' for _ in range(len(args.checkpoints))]
    # checkpoints = ['models/baseline_nodrop.h5', 'models/baseline_dropout.h5']
    # metric = 'acc'
    # prefixes = ['no dropout', 'dropout', 'nin']
    # names = ['training error', 'testing error']
    handles = []

    # Configure plotting options
    if args.save:
        matplotlib.use('PDF')

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5)
    num_curves = len(args.checkpoints)
    if  num_curves <= 6:
        palette = sns.color_palette()
    else:
        palette = sns.hls_palette(len(args.checkpoints), l=.4)

    for idx, (chkpt, pref) in enumerate(zip(args.checkpoints, args.arch_names)):
        meta = load_meta(chkpt)
        epochs = np.array(meta['epochs']) + 1
        if args.metric == 'err':
            m = 1 - np.array(meta['acc'])
            val_m = 1 - np.array(meta['val_acc'])
        else:
            m = meta[args.metric]
            val_m = meta['val_%s' %args.metric]
        h, = plt.plot(epochs, m, '--', label='%s %s' %(pref, args.metric_names[0]), color=palette[idx])
        if num_curves <= 6:
            handles.append(h)
        h, = plt.plot(epochs, val_m, label='%s %s' %(pref, args.metric_names[1]), color=palette[idx])
        handles.append(h)

    plt.xlabel('Epochs')
    plt.ylabel(label[args.metric])

    plt.legend(handles=handles, frameon=True, loc='best')
    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
    else:
        plt.show()
