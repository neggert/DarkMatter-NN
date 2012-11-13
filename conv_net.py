import theano.tensor as T
import theano
import numpy as np
import gzip
import cPickle
import os
import copy

import layers
reload(layers)
import utilities

class ConvNet(object):
    def __init__(self, input, rng, input_shape, filter_shape, n_hidden, output_shape, l1_penalty):
        self.l1_penalty = np.float32(l1_penalty)

        self.conv_layer = layers.ConvLayer(input, rng, image_shape=input_shape, filter_shape=filter_shape)
        self.hidden_layer = layers.LogisticLayer(self.conv_layer.output, rng, filter_shape[1]*(input_shape[3]-filter_shape[3]+1)**2, n_hidden)
        self.softmax_layer = layers.SoftmaxLayer(self.hidden_layer.output, n_hidden, output_shape**2)
        self.image_layer = layers.ImageLayer(self.softmax_layer.output, (input_shape[0], output_shape, output_shape))
        self.output_layer = layers.SmoothingLayer(self.image_layer.output, 100, output_shape)

        self.output = self.output_layer.predict

        self.params = self.conv_layer.params+self.hidden_layer.params+self.softmax_layer.params

    def costL1(self, t):
        return T.cast(self.output_layer.nll(t)+self.l1_penalty*(self.hidden_layer.L1+self.softmax_layer.L1), theano.config.floatX)

    def cost(self, t):
        return self.output_layer.nll(t)

    def save_params(self, filename):
        np.save(filename, (self.conv_layer.W.get_value(),
                           self.hidden_layer.W.get_value(),
                             self.hidden_layer.b.get_value(),
                             self.softmax_layer.W.get_value(),
                             self.softmax_layer.b.get_value()))

    def load_params(self, filename):
        cW, cb, hW, hb, oW, ob = np.load(filename)
        self.conv_layer.W.set_value(cW)
        self.hidden_layer.W.set_value(hW)
        self.hidden_layer.b.set_value(hb)
        self.softmax_layer.W.set_value(oW)
        self.softmax_layer.b.set_value(ob)

def train_conv_net():
    theano.config.compute_test_value = 'warn'


    # initialize some stuff
    # probably eventually want to un-hard-code this
    train_size=200
    valid_size=60
    filter_size = 10
    nbins_out = 22
    n_hidden = 22**2
    batch_size=4*train_size
    learning_rate = 0.5
    rng = np.random.RandomState(4321)

    # load the data
    datasets = utilities.load_data("/home/nic/DM/data/train_skies_grid.npz", train_size, valid_size)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # get the shape of the input image from data
    nbins = train_set_x.get_value().shape[3]

    # prepare theano objects
    data = T.ftensor4('x')
    data.tag.test_value = train_set_x.get_value()
    target = T.fmatrix('y')
    target.tag.test_value = train_set_y.get_value()

    # create the net
    conv_net_params = [rng, [batch_size, 3, nbins,nbins], (3, 3, filter_size,filter_size), n_hidden, nbins_out, .001]
    cls = ConvNet(data, *conv_net_params )

    # this is just we don't get the same output for each test case when testing
    # cls.softmax_layer.W.set_value(rng.uniform(size=(n_hidden, nbins_out**2)).astype(theano.config.floatX))

    #Sanity check to make sure the net works
    cost = theano.function(inputs=[],
                           outputs=cls.cost(target),
                           givens={data:train_set_x,
                                   target: train_set_y})

    print "Testing to make sure forward propagation works"
    print cost()

    # Setup learning rule
    # Currently using gradient decent with momentum
    grads = T.grad(cls.costL1(target), cls.params)

    updates = {}
    momentum = {}
    for p, g in zip(cls.params, grads):
        momentum[p] = theano.shared(np.zeros_like(p.get_value()))
        updates[p] = p+learning_rate*momentum[p]-(1-learning_rate)*g
        updates[momentum[p]] = learning_rate*momentum[p]-(1-learning_rate)*g

    # compile the training function in theano
    train_model = theano.function(inputs=[],
                                  outputs=cls.costL1(target),
                                  givens = {
                                    data: train_set_x,
                                    target: train_set_y
                                  },
                                  updates = updates
                                  # ,mode="DebugMode"
                                 )

    # do the actual training
    print "Training"
    for i in xrange(5000):
        if i%10 == 0:
            # check the score on the validation set every 100 epochs
            # note that this returns the cost *without* the L1 penalty
            print "Validation Cost:", validate_conv_net(cls.params, valid_set_x, valid_set_y, conv_net_params)
            print train_model()
        print train_model()

    # save the model parameters
    cls.save_params("test_weights.npy")

def validate_conv_net(params, set_x, set_y, conv_net_params):

    nbins = set_x.get_value().shape[3]
    batch_size=set_x.get_value().shape[0]

    data = T.tensor4('x')
    data.tag.test_value = set_x.get_value()
    target = T.fmatrix('y')
    target.tag.test_value = set_y.get_value()

    conv_net_params[1][0] = batch_size

    cls = ConvNet(data, *conv_net_params)
    cW, hW, hb, oW, ob = params
    cls.conv_layer.W.set_value(cW.get_value())
    cls.hidden_layer.W.set_value(hW.get_value())
    cls.hidden_layer.b.set_value(hb.get_value())
    cls.softmax_layer.W.set_value(oW.get_value())
    cls.softmax_layer.b.set_value(ob.get_value())

    cost = theano.function(inputs=[],
                       outputs=cls.cost(target),
                       givens={data:set_x,
                               target: set_y})
    return cost()
