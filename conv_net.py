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
reload(utilities)

class PrintEverythingMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [print_eval])
        super(PrintEverythingMode, self).__init__(wrap_linker, optimizer='fast_run')

class ConvNet(object):
    def __init__(self, input, rng, input_shape, filter_shape, n_hidden, output_shape):

        self.conv_layer = layers.ConvLayer(input, rng, image_shape=input_shape, filter_shape=filter_shape)
        self.hidden_layer = layers.LogisticLayer(self.conv_layer.output, rng, filter_shape[0]*(input_shape[3]-filter_shape[3]+1)**2, n_hidden)
        self.softmax_layer = layers.SoftmaxLayer(self.hidden_layer.output, n_hidden, output_shape**2)
        self.image_layer = layers.ImageLayer(self.softmax_layer.output, (input_shape[0], output_shape, output_shape))
        self.output_layer = layers.SmoothingLayer(self.image_layer.output, 100, output_shape)

        self.output = self.output_layer.predict

        self.params = self.conv_layer.params+self.hidden_layer.params+self.softmax_layer.params

    def cost(self, t):
        return self.output_layer.nll(t)

    def save_params(self, filename):
        np.save(filename, (self.conv_layer.W.get_value(),
                           self.hidden_layer.W.get_value(),
                             self.hidden_layer.b.get_value(),
                             self.softmax_layer.W.get_value(),
                             self.softmax_layer.b.get_value()))

    def load_params(self, filename):
        cW, hW, hb, oW, ob = np.load(filename)
        self.conv_layer.W.set_value(cW)
        self.hidden_layer.W.set_value(hW)
        self.hidden_layer.b.set_value(hb)
        self.softmax_layer.W.set_value(oW)
        self.softmax_layer.b.set_value(ob)

    def copy_params(self, other_net):
        self.conv_layer.copy_params(other_net.conv_layer)
        self.hidden_layer.copy_params(other_net.hidden_layer)
        self.softmax_layer.copy_params(other_net.softmax_layer)

def train_convnet(train_size=200, valid_size=60, iterations=10000, momentum_decay=0.9, learning_rate=0.7, filter_size=10, n_hidden=500, n_filters=6, output_size=21, plot=False):
    theano.config.compute_test_value = 'off'


    # initialize some stuff
    # probably eventually want to un-hard-code this
    nbins_out = output_size
    batch_size=train_size
    rng = np.random.RandomState(4321)

    # load the data
    datasets = utilities.load_data("data/train_skies_grid.npz", train_size, valid_size, flip=False, rotate=False)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # get the shape of the input image from data
    nbins = train_set_x.get_value().shape[3]

    # prepare theano objects
    data = T.tensor4('x')
    data.tag.test_value = train_set_x.get_value()
    target = T.matrix('y')
    target.tag.test_value = train_set_y.get_value()

    # create the net
    conv_net_params = [rng, [batch_size, 3, nbins,nbins], (n_filters, 3, filter_size,filter_size), n_hidden, nbins_out]
    cls = ConvNet(data, *conv_net_params )

    val_params = copy.copy(conv_net_params)
    val_params[1][0] = valid_size
    val = ConvNet(data, *val_params)

    #Sanity check to make sure the net works
    cost = theano.function(inputs=[],
                           outputs=[cls.cost(target),cls.softmax_layer.output, cls.hidden_layer.output,
                            cls.conv_layer.output, cls.output_layer.predict(target[:,0], target[:,1])],
                           givens={data:train_set_x,
                                   target: train_set_y}
                         # ,mode=PrintEverythingMode()
    )

    print "Testing to make sure forward propagation works"
    print cost()

    # Setup learning rule
    # Currently using gradient decent with momentum
    grads = T.grad(cls.cost(target), cls.params)

    updates = {}
    momentum = {}
    for p, g in zip(cls.params, grads):
        momentum[p] = theano.shared(np.zeros_like(p.get_value()))
        updates[p] = p+learning_rate*(momentum_decay*momentum[p]-(1-momentum_decay)*g)
        updates[momentum[p]] = momentum_decay*momentum[p]-(1-momentum_decay)*g


    train_model = theano.function(inputs=[],
                                  outputs=[cls.cost(target), grads[0]],
                                  givens = {
                                    data: train_set_x,
                                    target: train_set_y
                                  },
                                  updates = updates
                                 )

    validation_cost = theano.function(inputs=[],
                                     outputs = val.cost(target),
                                     givens = {
                                        data: valid_set_x,
                                        target: valid_set_y
                                     })

    # do the actual training
    print "Training"
    val_score = []
    train_score = []
    for i in xrange(iterations):
        if i%100 == 0:
            # check the score on the validation set every 100 epochs
            # note that this returns the cost *without* the L1 penalty
            val.copy_params(cls)
            vc = validation_cost()
            print "Validation Cost:", vc
            val_score.append(vc)
            # print "Validation Prediction\n", validation_pred()[-1]
            tc = train_model()
            print tc[0], np.cast[np.ndarray](tc[1])
            train_score.append(tc)
            if i > 1500:
                # check stopping condition
                # linear least squares to last 10 points in train_score
                # see np.linalg.lstsq for explanation of how this works
                A = np.vstack([np.arange(10)*100, np.ones(10)]).T
                y = np.asarray(train_score[-10:])
                slope, intercept = np.linalg.lstsq(A, y)[0]
                if -slope < .1:
                    print "{} iterations".format(i)
                    print "Final slope: ", slope
                    print "Final intercept: ", intercept
                    break
        train_model()

        # import pdb
        # pdb.set_trace()

    print "Final Training Cost: {}".format(train_model())
    print "Final Validation Cost: {}".format(validation_cost())

    print "Validation preditions"
    print validation_pred()

    # save the model parameters
    cls.save_params("test_weights_regress.npy")

    if plot:
        plt.figure()
        plt.plot(val_score)
        plt.plot(train_score)
        plt.legend(["Validation Cost", "Training Cost"])
        plt.show()

