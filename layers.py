import theano.tensor as T
import theano
import numpy as np
import gzip
import cPickle
import os
import copy
from pandas import *

class SoftmaxLayer(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W_sm', borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b_sm', borrow=True)

        self.output = T.nnet.softmax(T.dot(input, self.W)+self.b)

        self.pred = T.argmax(self.output, axis=1)

        self.params = [self.W, self.b]

        # self.L1 = T.mean(abs(self.W))

    def nll(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def copy_params(self, other_layer):
        self.W.set_value(other_layer.W.get_value())
        self.b.set_value(other_layer.b.get_value())


class LogisticLayer(object):
    def __init__(self, input, rng, n_in, n_out):
        W_values = np.asarray(rng.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)), dtype=theano.config.floatX)
        W_values *= 4
        self.W = theano.shared(W_values,
                               name='W_log', borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b_log', borrow=True)

        self.output = T.nnet.sigmoid(T.dot(input, self.W)+self.b)

        self.params = [self.W, self.b]

        self.L1 = T.sum(abs(self.W))

    def copy_params(self, other_layer):
        self.W.set_value(other_layer.W.get_value())
        self.b.set_value(other_layer.b.get_value())

class RectifiedLinearLayer(object):
    def __init__(self, input, rng, n_in, n_out):
        W_values = np.asarray(rng.uniform(
        low=-1./n_in,
        high=1./n_in,
        size=(n_in, n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(W_values,
                               name='W_linear', borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b_linear', borrow=True)
        z = T.dot(input, self.W)+self.b
        self.output = z

        self.params = [self.W, self.b]

    def sq_error(self, t):
        return T.mean((t-self.output)**2)

    def copy_params(self, other_layer):
        self.W.set_value(other_layer.W.get_value())
        self.b.set_value(other_layer.b.get_value())

class ConvLayer(object):

    def __init__(self, input, rng, image_shape, filter_shape):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
            name='W_conv',
                               borrow=True)

        # # the bias is a 1D tensor -- one bias per output feature map
        # b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        # self.b = theano.shared(value=b_values, borrow=True, name="b_conv")

        # convolve input feature maps with filters
        self.output = T.flatten(T.nnet.conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape), outdim=2)

        # store parameters of this layer
        self.params = [self.W]

    def copy_params(self, other_layer):
        self.W.set_value(other_layer.W.get_value())

class ImageLayer(object):
    # this basically just reshapes its input and provides a cost function
    def __init__(self, input, output_shape):
        self.output = input.reshape(output_shape, ndim=3)

    def nll(self, target):
        return -T.mean(T.log(T.mul(self.output,target))) # hopefully this is element-wise multiplication


class SmoothingLayer(object):
    def __init__(self, input, sigma, nbins):
        self.input = input
        self.sigma = sigma
        self.nbins = nbins
        self.bin_width = 4200./nbins

        # x and y coordinate of each pixel
        dy_np, dx_np = np.mgrid[:nbins, :nbins]
        dx_np *= self.bin_width
        dy_np *= self.bin_width
        # translate to bin centers
        dx_np += self.bin_width/2
        dy_np += self.bin_width/2

        self.dx = theano.shared(dx_np.astype(theano.config.floatX), name='x_coord', borrow=True)
        self.dy = theano.shared(dy_np.astype(theano.config.floatX), name='y_coord', borrow=True)


    def predict(self, x,y):
        x0 = self.dx.dimshuffle('x', 0, 1)
        y0 = self.dy.dimshuffle('x', 0, 1)
        return T.sum(self.input/np.sqrt(2*np.pi*self.sigma**2)*T.exp(-(T.sub(x0,x.dimshuffle(0,'x','x'))**2+T.sub(y0,y.dimshuffle(0,'x','x'))**2)/2/self.sigma), axis=(1,2))

    def nll(self, t):
        eps = 1e-6
        return -T.mean(T.log(self.predict(t[:,0],t[:,1]))
                       # + T.log(self.predict(t[:,2],t[:,3]))#*(t[:,2]>eps) # contribution is 0 for rows where x = 0
                       # + T.log(self.predict(t[:,4],t[:,5]))#*(t[:,4]>eps)
                      )

