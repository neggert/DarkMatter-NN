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

import matplotlib.pyplot as plt

class RegressConvNet(object):
    def __init__(self, input, rng, input_shape, filter_shape, n_hidden, n_out, l1_penalty):
        self.l1_penalty = l1_penalty

        self.conv_layer = layers.ConvLayer(input, rng, image_shape=input_shape, filter_shape=filter_shape)
        self.hidden_layer = layers.LogisticLayer(self.conv_layer.output, rng, filter_shape[1]*(input_shape[3]-filter_shape[3]+1)**2, n_hidden)
        self.output_layer = layers.RectifiedLinearLayer(self.hidden_layer.output, rng, n_hidden, n_out)

        self.output = self.output_layer.output

        self.params = self.conv_layer.params+self.hidden_layer.params+self.output_layer.params

    def costL1(self, t):
        return self.output_layer.sq_error(t)+self.l1_penalty*(self.hidden_layer.L1)

    def cost(self, t):
        return self.output_layer.sq_error(t)

    def save_params(self, filename):
        np.save(filename, (self.conv_layer.W.get_value(),
                           self.hidden_layer.W.get_value(),
                             self.hidden_layer.b.get_value(),
                             self.output_layer.W.get_value(),
                             self.output_layer.b.get_value()))

    def load_params(self, filename):
        cW, cb, hW, hb, oW, ob = np.load(filename)
        self.conv_layer.W.set_value(cW)
        self.hidden_layer.W.set_value(hW)
        self.hidden_layer.b.set_value(hb)
        self.output_layer.W.set_value(oW)
        self.output_layer.b.set_value(ob)

    def copy_params(self, other_net):
        self.conv_layer.copy_params(other_net.conv_layer)
        self.hidden_layer.copy_params(other_net.hidden_layer)
        self.output_layer.copy_params(other_net.output_layer)

class RegressNetWithDropoutTrain(RegressConvNet):
    def __init__(self, input, rng, input_shape, filter_shape, n_hidden, n_out, l1_penalty):
        self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
        self.conv_mask = self.rng.binomial(size=(input_shape[0], filter_shape[0]*(input_shape[3]-filter_shape[3]+1)**2), n=1, p=0.5)
        self.hidden_mask = self.rng.binomial(size=(input_shape[0], n_hidden), n=1, p=0.5)
        self.l1_penalty = l1_penalty

        self.conv_layer = layers.ConvLayer(input, rng, image_shape=input_shape, filter_shape=filter_shape)
        conv_output = self.conv_layer.output*self.conv_mask
        self.hidden_layer = layers.LogisticLayer(conv_output, rng, filter_shape[0]*(input_shape[3]-filter_shape[3]+1)**2, n_hidden)
        hidden_output = self.hidden_layer.output*self.hidden_mask
        self.output_layer = layers.RectifiedLinearLayer(hidden_output, rng, n_hidden, n_out)

        self.output = self.output_layer.output

        self.params = self.conv_layer.params+self.hidden_layer.params+self.output_layer.params

class RegressNetWithDropoutPredict(RegressConvNet):
    def load_params(self, filename):
        cW, cb, hW, hb, oW, ob = np.load(filename)
        self.conv_layer.W.set_value(cW/2)
        self.hidden_layer.W.set_value(hW/2)
        self.hidden_layer.b.set_value(hb/2)
        self.output_layer.W.set_value(oW)
        self.output_layer.b.set_value(ob)

    def copy_params(self, other_net):
        self.conv_layer.copy_params(other_net.conv_layer)
        self.hidden_layer.copy_params(other_net.hidden_layer)
        self.output_layer.copy_params(other_net.output_layer)
        self.conv_layer.W.set_value(self.conv_layer.W.get_value()/2)
        self.hidden_layer.W.set_value(self.hidden_layer.W.get_value()/2)
        self.hidden_layer.b.set_value(self.hidden_layer.b.get_value()/2)



def train_regress_net():
    # theano.config.compute_test_value = 'warn'
    theano.config.DebugMode.check_strides = 0


    # initialize some stuff
    # probably eventually want to un-hard-code this
    train_size=200
    valid_size=60
    filter_size = 10
    nbins_out = 6
    n_hidden = 500
    batch_size=4*train_size
    learning_rate = 0.9
    learning_rate_scale = .7
    rng = np.random.RandomState(4321)

    # load the data
    datasets = utilities.load_data("/home/nic/DM/data/train_skies_grid.npz", train_size, valid_size)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # get the shape of the input image from data
    nbins = train_set_x.get_value().shape[3]

    # prepare theano objects
    data = T.tensor4('x')
    # data.tag.test_value = train_set_x.get_value()
    target = T.matrix('y')
    # target.tag.test_value = train_set_y.get_value()

    # create the net
    net_params = [rng, [batch_size, 3, nbins,nbins], (6, 3, filter_size,filter_size), n_hidden, nbins_out, .001]
    cls = RegressNetWithDropoutTrain(data, *net_params )

    # create a validation net
    val_params = copy.copy(net_params)
    val_params[1][0] = valid_size
    val = RegressNetWithDropoutPredict(data, *val_params)

    #Sanity check to make sure the net works
    cost = theano.function(inputs=[],
                           outputs=cls.costL1(target),
                           givens={data:train_set_x,
                                   target: train_set_y})

    print "Testing to make sure forward propagation works"
    print cost()

    # Setup learning rule
    # Currently using gradient decent with momentum
    grads = T.grad(cls.cost(target), cls.params)

    updates = {}
    momentum = {}
    for p, g in zip(cls.params, grads):
        momentum[p] = theano.shared(np.zeros_like(p.get_value()))
        updates[p] = p+learning_rate_scale*(learning_rate*momentum[p]-(1-learning_rate)*g)
        updates[momentum[p]] = learning_rate_scale*(learning_rate*momentum[p]-(1-learning_rate)*g)

    # compile the training function in theano
    train_model_debug = theano.function(inputs=[],
                                  outputs=[cls.cost(target), cls.output, cls.conv_layer.output, cls.hidden_layer.output],
                                  givens = {
                                    data: train_set_x,
                                    target: train_set_y
                                  },
                                  updates = updates
                                  # ,mode="DebugMode"
                                 )
    train_model = theano.function(inputs=[],
                                  outputs=cls.cost(target),
                                  givens = {
                                    data: train_set_x,
                                    target: train_set_y
                                  },
                                  updates = updates
                                  # ,mode="DebugMode"
                                 )

    validation_cost = theano.function(inputs=[],
                                     outputs = val.cost(target),
                                     givens = {
                                        data: valid_set_x,
                                        target: valid_set_y
                                     })

    validation_pred = theano.function(inputs=[],
                                      outputs = val.output,
                                      givens= {data: valid_set_x})

    # do the actual training
    print "Training"
    val_score = []
    train_score = []
    for i in xrange(10000):
        if i%100 == 0:
            # check the score on the validation set every 100 epochs
            # note that this returns the cost *without* the L1 penalty
            val.copy_params(cls)
            vc = validation_cost()
            print "Validation Cost:", vc
            val_score.append(vc)
            # print "Validation Prediction\n", validation_pred()
            tc = train_model()
            print tc
            train_score.append(tc)
        train_model()

        # import pdb
        # pdb.set_trace()

    # save the model parameters
    cls.save_params("test_weights_regress.npy")

    plt.figure()
    plt.plot(val_score)
    plt.plot(train_score)
    plt.legend(["Validation Cost", "Training Cost"])
    plt.show()
