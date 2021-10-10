import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import time 
import matplotlib.pyplot as plt
from mlp_dropout import GaussianMLP, BernoulliMLP, CLS
from utils import kld_unit_mvn, shared_dataset,log_diag_mvn, floatX, loss_q_logp, nrmse, normalized_mse, compute_mmd, compute_kernel, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from optimizers import optimizer
from rpy2.robjects.packages import importr
from rpy2 import robjects
hmeasure = importr("hmeasure")

e = 1e-8
class CBMD(object):
    def __init__(self, x2dim,x1dim,args,ydim=2,enc_weights=None,dec_weights=None,dec2_weights=None,cls_weights=None):
        self.lr = args.lr_l
        self.batch_size = args.batch_size
        self.x2dim = x2dim
	self.x1dim = x1dim
        self.ydim  = ydim
        self.hdim_enc = args.hdim_enc
        self.hdim_dec = args.hdim_dec
        self.hdim_dec2 = args.hdim_dec2
        self.hdim_cls = args.hdim_cls
        self.zdim = args.zdim
        self.x2 = T.matrix('x2', dtype=floatX)	
        self.x1 = T.matrix('x1', dtype=floatX)
        self.y  = T.matrix('y', dtype=floatX) 
	self.eps = T.tensor3('eps', dtype=floatX)
        self.nlayers_enc = args.nlayers_enc
        self.nlayers_dec = args.nlayers_dec
        self.nlayers_dec2 = args.nlayers_dec2
        self.nlayers_cls = args.nlayers_cls
	self.samples_z   = args.samples_z
	self.beta  = T.fscalar('beta')
        self.alpha = T.fscalar('alpha')
        self.lamda = T.fscalar('lamda')
        self.training_mode = T.scalar('t_mode')

        if args.hl_activation == 'softplus':
            activation = T.nnet.nnet.softplus
        elif args.hl_activation == 'relu':
            activation = T.nnet.relu

        # q(z|x2,x1,y)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_enc)
        input_to_enc = T.concatenate([self.x2,self.x1, self.y],axis=1)                           
        self.enc_mlp = GaussianMLP(input_to_enc, self.x2dim+self.x1dim+self.ydim, self.hdim_enc, self.zdim, self.training_mode, nlayers=self.nlayers_enc, eps=self.eps, Weights=enc_weights, dropout_rate=dropout_rate, activation=activation)
	
        # q(z|x2,x1,y)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_enc)
        self.enc_mlp_copy = GaussianMLP(input_to_enc, self.x2dim+self.x1dim+self.ydim, self.hdim_enc, self.zdim, self.training_mode, nlayers=self.nlayers_enc, eps=self.eps, Weights=[self.enc_mlp.params,True],dropout_rate=dropout_rate,activation=activation)
        
        # p(x2|z,x1)
	# loop trough each samples of z
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec)
	for i in range(self.samples_z): 
	    input_to_dec = T.concatenate([self.enc_mlp.draw_sample[i],self.x1],axis=1)                              
     
	    if i == 0:
                if args.dec_type == 'gaussian':
		    self.dec_mlp = GaussianMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode,nlayers=self.nlayers_dec, y=self.x2, Weights=dec_weights,dropout_rate=dropout_rate,activation=activation)
                elif args.dec_type == 'bernoulli':
		    self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode,nlayers=self.nlayers_dec, y=self.x2, Weights=dec_weights,dropout_rate=dropout_rate,activation=activation)
	    else:
                if args.dec_type == 'gaussian':
		    self.dec_mlp = GaussianMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],  y=self.x2, dropout_rate=dropout_rate,activation=activation)
                elif args.dec_type == 'bernoulli':
                    self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],  y=self.x2, dropout_rate=dropout_rate,activation=activation)

            dec_cost = self.dec_mlp.cost.reshape((self.batch_size,1))
	    if i == 0:
	        self.all_dec_cost = dec_cost
	    else:
		self.all_dec_cost = T.concatenate([self.all_dec_cost,dec_cost],axis=1)

	# p(z|x1)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec2)
        input_to_dec2 = self.x1
	self.dec2_mlp = GaussianMLP(input_to_dec2, self.x1dim, self.hdim_dec2, self.zdim, self.training_mode,nlayers=self.nlayers_dec2, eps=self.eps, Weights=dec2_weights, dropout_rate=dropout_rate,activation=activation)
        
	# copy pf p(z|x1) to use at test
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec2)
	self.dec2_mlp_copy = GaussianMLP(input_to_dec2, self.x1dim, self.hdim_dec2, self.zdim, self.training_mode, nlayers=self.nlayers_dec2, eps=self.eps, Weights=[self.dec2_mlp.params,True], dropout_rate=dropout_rate, activation=activation)
        
        # make a copy of the decoder to use at test
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec)
        input_to_dec_copy = T.concatenate([self.dec2_mlp.mu,self.x1],axis=1)
        if args.dec_type == 'gaussian':
            self.dec_mlp_copy = GaussianMLP(input_to_dec_copy, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],y=self.x2, dropout_rate=dropout_rate,activation=activation)
        elif args.dec_type == 'bernoulli':
            self.dec_mlp_copy = BernoulliMLP(input_to_dec_copy, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],y=self.x2, dropout_rate=dropout_rate,activation=activation)

        # q(y|z) now i'm gonna test using only z
        #input_to_cls = T.concatenate([self.x1,self.dec2_mlp.mu],axis=1)   
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_cls)
        input_to_cls = self.dec2_mlp.mu   
        self.cls_mlp = CLS(input_to_cls, self.zdim, self.hdim_cls, self.ydim, self.training_mode, nlayers=self.nlayers_cls,Weights=cls_weights, dropout_rate=dropout_rate,activation=activation)
        
        input_to_cls_copy = self.enc_mlp.mu   
        self.cls_mlp_copy = CLS(input_to_cls_copy, self.zdim, self.hdim_cls, self.ydim, self.training_mode, nlayers=self.nlayers_cls,Weights=[self.cls_mlp.params,True],dropout_rate=dropout_rate,activation=activation)

	# take average cost across number of samples
	self.dec_mlp_cost = T.mean(self.all_dec_cost,axis=1)
	
        # Model parameters
        self.params = self.enc_mlp.params + self.dec2_mlp.params + self.dec_mlp.params + self.cls_mlp.params
        
        # use this to get encoder parameters 
        #self.enc_params = theano.function(inputs=[],outputs=self.enc_mlp.params)

        # use this to get decode parameters 
        #self.dec_params = theano.function(inputs=[],outputs=self.dec_mlp.params)

        # use this to get all model parameters 
        self.all_params = theano.function(inputs=[],outputs=self.params)
                
        
        self.y_pred = theano.function(
                inputs  = [self.x1, self.training_mode],
                outputs = self.cls_mlp.y_pred
        )
	
        
        # q(z|x1,x2,y)
        #self.draw_z_q = theano.function(
        #    inputs = [self.x2, self.x1, self.y, self.eps, self.training_mode],
        #    outputs = self.enc_mlp_copy.draw_sample
        #)
        self.draw_z_q = theano.function(
            inputs = [self.x2, self.x1, self.y, self.training_mode],
            outputs = self.enc_mlp.mu
        )

        # p(z|x1)
        self.draw_z_p = theano.function(
            inputs = [self.x1,self.eps, self.training_mode],
            outputs = self.dec2_mlp_copy.draw_sample
        )
	
        # p(x2|z,x1)
        self.draw_x2 = theano.function(
            inputs = [self.x1, self.training_mode],
            outputs = self.dec_mlp_copy.sample
        )

        # p(x2|z, x1)
        self.dec_params = theano.function(
            inputs = [self.x1, self.training_mode],
            outputs = [self.dec_mlp_copy.mu, self.dec_mlp_copy.var]
        )
	
        # q(z|x1,x2,y)
        self.enc_params = theano.function(
            inputs = [self.x2,self.x1, self.y, self.training_mode],
            outputs = [self.enc_mlp_copy.mu, self.enc_mlp_copy.var]
        )

	# p(z|x1)
        self.prior_params = theano.function(
            inputs = [self.x1, self.training_mode],
            outputs = [self.dec2_mlp_copy.mu, self.dec2_mlp_copy.var]
        )
        
        # q(y|z_prior)
        self.draw_pi = theano.function(
                inputs  = [self.x1, self.training_mode],
                #inputs  = [self.x1,self.x2,self.eps],
                outputs = self.cls_mlp.pi
        )
        
        # q(y|z_posterior)
        self.draw_pi_posterior = theano.function(
                inputs  = [self.x2,self.x1,self.y, self.training_mode],
                outputs = self.cls_mlp_copy.pi
        )


        ''' defining cost function '''
	kl = 0.5*T.sum(T.log(self.dec2_mlp.var),axis=1) - 0.5*T.sum(1+T.log(self.enc_mlp.var),axis=1) + T.sum(self.enc_mlp.var/self.dec2_mlp.var,axis=1) + 0.5*T.sum(T.square(self.enc_mlp.mu-self.dec2_mlp.mu)/self.dec2_mlp.var,axis=1)
        self.cls_cost = T.sum(self.alpha*T.nnet.categorical_crossentropy(self.cls_mlp.pi+e,self.y))

	self.kl = kl 
	self.nrmse = mean_squared_error(self.x2,self.dec_mlp.sample) # this is just to see nrmse during training

        all_z_kernel = 0.0
	for i in range(self.samples_z): 
            z_kernel = compute_mmd(self.dec2_mlp.draw_sample[i],self.enc_mlp.draw_sample[i])
            all_z_kernel+=z_kernel
        self.z_kernel = self.lamda * all_z_kernel/self.samples_z

        cost1 = T.sum(self.kl + self.dec_mlp_cost)/args.batch_size
        cost2 = T.sum(self.dec_mlp_cost)/args.batch_size + self.z_kernel
        self.vae_cost = self.beta*cost1 + (1.0-self.beta)*cost2 + self.cls_cost

    def get_performance(self,x1,x2,y,eps,x1_cal=None,x2_cal=None,y_cal=None,scores=None, calibrate = False):
        if scores is None:
            #scores = self.draw_pi(x1,x2,eps)
            scores = self.draw_pi(x1)
            scores = scores[:,1]
            if calibrate == True:
                from betacal import BetaCalibration
                eps_cal  = np.random.randn(1,x1_cal.shape[0],eps.shape[2]).astype(np.float32)
                #scores_cal = self.draw_pi(x1_cal,eps_cal)
                scores_cal = self.draw_pi(x1_cal)
                scores_cal = scores_cal[:,1] 
                bc =  BetaCalibration(parameters="abm")
                bc.fit(scores_cal,y_cal)
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

        return {'AUC':AUC, 'Gini':Gini, 'H':H, 'Recall':Recall, 'Precision': Precision, 'scores':scores}


    def get_performance_posterior(self,x1,x2,y,scores=None):
        from keras.utils import to_categorical
        
        if scores is None:
            scores1 = self.draw_pi_posterior(x2,x1,to_categorical(y))
            scores2 = self.draw_pi_posterior(x2,x1,to_categorical(1-y))
            scores = 0.5*scores1[:,1]+0.5*scores2[:,1]

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

        return {'AUC':AUC, 'Gini':Gini, 'H':H, 'Recall':Recall, 'Precision': Precision, 'scores':scores}

    def training(self,data_x2,data_x1,data_y):
        index = T.lscalar()
        
        print '<<<<<<<<<<< training ... >>>>>>>>>>'
        self.gparams = [T.grad(self.vae_cost, p) for p in self.params]

        self.optimizer = optimizer(self.params,self.gparams,self.lr)
        self.updates   = self.optimizer.adam()
        n_batches = data_x2.get_value().shape[0]/self.batch_size        

        train = theano.function(
            inputs=[index, self.eps,self.beta,self.alpha,self.lamda, self.training_mode],
            outputs=[self.vae_cost, self.dec_mlp_cost, self.kl, self.cls_cost, self.nrmse, self.z_kernel, self.dec2_mlp.var, self.dec_mlp.var, self.enc_mlp.var],
            updates=self.updates,
            givens={self.x2: data_x2[index*self.batch_size : (index+1)*self.batch_size],
                    self.x1: data_x1[index*self.batch_size : (index+1)*self.batch_size],     
                    self.y:  data_y[index*self.batch_size : (index+1)*self.batch_size]
                    }
        ) 

        return train
       
    def train_model(self,data_x2,data_x1,data_y, epochs, beta=1,alpha=1):
	n_batches = data_x2.get_value().shape[0]/self.batch_size
	train = self.training(data_x2,data_x1,data_y)

        for e in range(epochs):
            for i in range(n_batches):
                eps  = np.random.randn(self.samples_z,self.batch_size, self.zdim).astype(floatX)
                vae_cost = train(i,eps,beta,alpha)

            if e % 50 == 0 and e > 0:           
                print 'cost at epoch ', e, ' is ' ,vae_cost[0], ', decoder cost: ', np.mean(vae_cost[1]), ', kl is ', np.mean(0.5*vae_cost[2]), ' and cls is ', np.mean(vae_cost[3])
