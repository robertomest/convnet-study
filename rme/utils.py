import matplotlib.pyplot as plt
import keras.backend as K

def plot_metrics(meta, title):
    f, axarr = plt.subplots(2, sharex=True)
    loc = [1, 4]
    for i, name in enumerate(['loss', 'acc']):
        legend = ['Training ' + name, 'Validation ' + name]
        axarr[i].plot(meta[name])
        axarr[i].plot(meta['val_' + name])
        axarr[i].legend(legend, loc=loc[i])
    axarr[0].set_title(title)
    plt.show()

def name_with_uid(name):
    return name + '%d' %K.get_uid(name)
