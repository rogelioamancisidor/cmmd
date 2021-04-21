import numpy as np
import theano
import theano.tensor as T
from utils import floatX
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.extra_ops import to_one_hot
from optimizers import optimizer
from rpy2.robjects.packages import importr
from rpy2 import robjects
hmeasure = importr("hmeasure")

# XXX
rng = np.random.RandomState()
srng = RandomStreams()
e = 1e-8

class HiddenLayer(object):

    # adapted from http://deeplearning.net/tutorial/mlp.html
    # finds last weights to be assigend in putput layer: last_weights = Weights[-4:]
    # finds the weights positions in the list list(set(Weights)-set(last_weights))

    def __init__(self, input, n_in, n_out, W=None, b=None,activation=T.tanh, prefix=''):
        self.n_in = n_in
        self.n_out = n_out
        method = 'glorot'      
        if W is None:
            # NOTE tried glorot init and randn and glorot init worked better
            # after 1 epoch with adagrad
            if method == 'glorot':
                W_values = np.asarray(
                            rng.uniform(
                                low=-np.sqrt(2. / (n_in + n_out)),
                                high=np.sqrt(2. / (n_in + n_out)),
                                size=(n_in, n_out)
                            ),dtype=theano.config.floatX)
            elif method == 'random':            
                W_values = np.asarray(
                         rng.normal(size=(n_in,n_out),loc=0,scale=0.001),
                        dtype=floatX
                        )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'_W', borrow=True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'_b', borrow=True)
        
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        
        if activation is None:
            output = lin_output
        elif activation == T.nnet.relu:
            output = activation(lin_output,0.0)
        else:
            output = activation(lin_output)
        self.output = output
            
        # parameters of the model
        self.params = [self.W, self.b]

class _MLP(object):
    # building block for MLP instantiations defined below
    def __init__(self, x, n_in, n_hid, nlayers=1, prefix='', activation = T.nnet.nnet.softplus,  Weights=None):
        self.nlayers = nlayers
        self.hidden_layers = list()
        inp = x

        factor = 2                      #facor is the number of parameters per layer. 
    
        if Weights == None:        
            Weights = np.tile([None],self.nlayers*factor) # create an array with Nones so I can pass None weihts in the loop below
        
        for k in xrange(self.nlayers): 
            # keep track of the weight index as function of k            
            idx_weight = k*factor
            idx_bias = idx_weight + 1
            idx_gamma = idx_bias + 1
            idx_beta  = idx_gamma + 1

            hlayer = HiddenLayer(
                W=Weights[idx_weight],
                b=Weights[idx_bias],
                input=inp,
                n_in=n_in,
                n_out=n_hid[k],
                activation=activation,
                prefix=prefix + ('_%d' % (k + 1))
                            )

            n_in = n_hid[k]
            inp = hlayer.output
            self.hidden_layers.append(hlayer)

        self.params = [param for l in self.hidden_layers for param in l.params]
        self.input = input
        # NOTE output layer computed by instantations

class CLS(_MLP):
    # Weight is a list with W's and b's to be pass into HiddenLayer, mu_layer and log_var_layer
    def __init__(self, n_in, n_hid, n_out=2, nlayers=1, Weights=None, activation=T.tanh): 
        self.x = T.matrix('x',dtype=floatX)
        self.y = T.matrix('y',dtype=floatX)
        if Weights is not None:
            # the pass_output flag controls whether to pass output layers weights
            # pass_output = True:  assigns weights to both hidden and output layers
            # pass_output = False: assigns weights only to hidden layers
            pass_output  = Weights[1]    
            Weights      = Weights[0]

            if pass_output == True:
                output_weights =  Weights[-2:] # this extracts the last two set of weights in the list. 
                W = output_weights[0]    
                b = output_weights[1]       
            else:
                output_weights =  Weights[-2:]
                W = None   
                b = None
            
            hidden_weights = Weights[0:len(Weights)-len(output_weights)]
            Weights = hidden_weights 
        else:
            W = None   
            b = None

            Weights = None
 
        super(CLS, self).__init__(self.x, n_in, n_hid, nlayers=nlayers, activation=activation, prefix='GaussianMLP_hidden',Weights=Weights)
        # for use as classifier
        self.cls_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=T.nnet.softmax,
            prefix='GaussianMLP_cls',
            W=W,
            b=b
        )
        self.params = self.params + self.cls_layer.params #+ self.BNlayer.params
        self.pi = self.cls_layer.output                             # this will be a list of size equal to the minibatch, and each 
                                                                    # entry has k elements, where k is the number of classes
        self.y_pred = to_one_hot(T.argmax(self.pi, axis=1),n_out)       # this is a one hot encoding list with size equal to the minibatch with the predicted class

        self.cls_cost = T.sum(T.nnet.categorical_crossentropy(self.pi+e,self.y))

        self.draw_pi = theano.function(
                inputs  = [self.x],
                outputs = self.pi
        )

        self.label_hat = theano.function(
                inputs  = [self.x],
                outputs = self.y_pred
        )
        
        # use this to get all model parameters 
        self.get_params = theano.function(inputs=[],outputs=self.params)

    def train_fn(self,data_x,data_y,batch_size=150,lr=0.0001):
        index = T.lscalar()
        print '<<<<<<<<<<< training CLS! ... >>>>>>>>>>'
        self.gparams = [T.grad(self.cls_cost, p) for p in self.params]

        self.optimizer = optimizer(self.params,self.gparams,lr)
        self.updates   = self.optimizer.adam()

        train = theano.function(
            inputs=[index],
            outputs=[self.cls_cost],
            updates=self.updates,
            givens={self.x: data_x[index*batch_size : (index+1)*batch_size],
                    self.y: data_y[index*batch_size : (index+1)*batch_size]     
                    }
        ) 
        return train

    def training(self,x,y,epochs=101,lr=0.0001,batch_size=150):
        train = self.train_fn(x,y,batch_size,lr)
        n_batches = x.get_value().shape[0]/batch_size        
        for e in range(epochs):
            for i in range(n_batches):
                vae_cost = train(i)

            if e % 20 == 0 and e > 0:           
                print 'cls cost at epoch ', e, ' is ' ,vae_cost[0]

    def performance(self,x,y,x_cal=None,y_cal=None,scores=None, calibrate = False):
        if scores is None:
            scores = self.draw_pi(x)
            scores = scores[:,1]
            if calibrate == True:
                from betacal import BetaCalibration

                scores_cal = self.draw_pi(x_cal)
                bc =  BetaCalibration(parameters="abm")
                bc.fit(scores_cal[:,1],y_cal)
                scores = bc.predict(scores)

        d = {'Segment 0': robjects.FloatVector(scores)}
        dataf = robjects.DataFrame(d)
        rho = 1.0*sum(y)/float(y.shape[0])
        results = hmeasure.HMeasure(robjects.IntVector(y),dataf,threshold=rho)

        H    = results[0][0][0]
        Gini = results[0][1][0]
        AUC  = results[0][2][0]
        TP   = results[0][18][0]
        FP   = results[0][19][0]
        FN   = results[0][21][0]
        Recall  = 1.0*TP/(TP+FN+e)
        Precision  = 1.0*TP/(TP+FP+e)
        error = 1.0*(FP+FN)/x.shape[0]
        F1 = 2.0*(Recall*Precision)/(Recall+Precision+e)

        return {'AUC':AUC, 'Gini':Gini, 'H':H, 'Recall':Recall, 'Precision': Precision, 'scores':scores,'error':error, 'f1':F1}
