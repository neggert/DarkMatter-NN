import theano.tensor as T
import theano
import numpy as np
import gzip
import cPickle
import os
import copy

import scipy.ndimage.interpolation

def shared_dataset(x, y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y


def load_data(dataset, train_size=200, valid_size=50, rotate=True, flip=True):

    datafile = np.load(dataset)
    data = datafile['output']
    targets = datafile['targets']

    # shuffle data
    rng = np.random.RandomState(43)
    indices = np.arange(data.shape[0])
    rng.shuffle(indices)
    data = data[indices]
    targets = targets[indices]

    # note: there are 300 training examples
    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    train_targets = targets[:train_size]
    valid_targets = targets[train_size:train_size+valid_size]
    test_targets = targets[train_size+valid_size:]

    # add rotated versions to the training data
    td_orig = copy.copy(train_data)
    tt_orig = copy.copy(train_targets)

    if flip:
        train_data = np.append(train_data, flip_data_x(td_orig), axis=0)
        train_targets = np.append(train_targets, flip_targets_x(tt_orig), axis=0)

    td_orig = copy.copy(train_data)
    tt_orig = copy.copy(train_targets)

    if rotate:
        for ang in (90.,180.,270.):
            train_data = np.append(train_data, rotate_data(td_orig, ang), axis=0)
            train_targets = np.append(train_targets, rotate_targets(tt_orig, ang), axis=0)

    test_set_x, test_set_y = shared_dataset(test_data, test_targets)
    valid_set_x, valid_set_y = shared_dataset(valid_data, valid_targets)
    train_set_x, train_set_y = shared_dataset(train_data, train_targets)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_test(dataset):

    datafile = np.load(dataset)
    data = datafile['output']
    skies = datafile['sky_name']

    return [data, skies]

def rotate_data(x, angle):
    if int(angle) not in (90,180,-90, 270):
        raise ValueError("angle must be a multiple of 90 degrees")
    out = scipy.ndimage.interpolation.rotate(x, angle, axes=(2,3), reshape=False, order=0)
    if int(angle) in (90,-90,270):
        out[:,1:2,:,:] *= -1
    return out

def rotate_targets(x, angle):
    a_rad = angle*np.pi/180.

    # move so that image center is at 0
    centered_x = x - 2100

    # rotate
    out = np.zeros_like(centered_x)
    # x values
    out[:,0::2] = centered_x[:,::2]*np.cos(a_rad)+centered_x[:,1::2]*np.sin(a_rad)
    # y values
    out[:,1::2] = -centered_x[:,::2]*np.sin(a_rad)+centered_x[:,1::2]*np.cos(a_rad)

    # move back so corner is at 0.
    out += 2100

    # set non-halos to 0
    out[out==4200] = 0.

    return out

def flip_data_x(data):
    out = data[:,:,::-1,:]
    out[:,2,:,:] *= -1 # only e2 changes sign
    return out

def flip_targets_x(targs):
    out = targs.copy()
    out -= 2100 # center
    out[:,0::2] *= -1 # flip x
    out += 2100 # move back so corner is at 0
    out[out==4200] = 0.
    return out


