import torch, itertools
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fftshift, ifftshift, fftn

def fftn_(image, dim=None):
    if dim is None:
        dim = tuple(range(1, len(image.shape)))
    kspace = fftshift(fftn(ifftshift(image, dim=dim), dim=dim, norm='ortho'), dim=dim)
    return kspace

def complex_to_chan(complex_tensor, chan_dim=1, num_chan=1):
    assert complex_tensor.dtype in [torch.complex32, torch.complex64, torch.complex128], 'The dtype of the input must be torch.complex32/64/128!'
    assert num_chan in [1, 2], 'Number of channels must be either 1 or 2!'

    if num_chan == 1:
        real_tensor = complex_tensor.abs().unsqueeze(-1)
    elif num_chan == 2:
        real_tensor = torch.view_as_real(complex_tensor)
    dim = np.arange(len(real_tensor.shape))
    dim[chan_dim+1:] = dim[chan_dim:-1]
    dim[chan_dim] = -1
    return real_tensor.permute(*dim)

def plot_confusion_matrix(cm,
                          target_names,
                          fname, 
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    fname:        filename of the figure for saving

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(fname)
    plt.close()