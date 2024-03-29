import numpy as np
import theano
import theano.tensor as T
from mlp_dropout import GaussianMLP, BernoulliMLP, CLS
from utils import kld_unit_mvn, shared_dataset,log_diag_mvn, floatX, loss_q_logp,compute_mmd, compute_kernel, t_rmse
from optimizers import optimizer

#
e = 1e-8
class CMMD(object):
    def __init__(self, x2dim,x1dim,args,ydim=2,enc_weights=None,dec_weights=None,prior_weights=None,cls_weights=None):
        self.lr = args.lr_l
        self.batch_size = args.batch_size
        self.x2dim = x2dim
	self.x1dim = x1dim
        self.ydim  = ydim
        self.hdim_enc = args.hdim_enc
        self.hdim_dec = args.hdim_dec
        self.hdim_prior = args.hdim_prior
        self.hdim_cls = args.hdim_cls
        self.zdim = args.zdim
        self.x2 = T.matrix('x2', dtype=floatX)	
        self.x1 = T.matrix('x1', dtype=floatX)
        self.y  = T.matrix('y', dtype=floatX) 
	self.eps = T.tensor3('eps', dtype=floatX)
        self.nlayers_enc = args.nlayers_enc
        self.nlayers_dec = args.nlayers_dec
        self.nlayers_prior = args.nlayers_prior
        self.nlayers_cls = args.nlayers_cls
	self.samples_z   = args.samples_z
	self.omega  = T.fscalar('omega')
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
	
        # p(x2|z,x1)
	# loop trough each samples of z
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec)
	for i in range(self.samples_z): 
	    input_to_dec = T.concatenate([self.enc_mlp.draw_sample[i],self.x1],axis=1)                              
     
	    if i == 0:
                if args.dec_type == 'gaussian':
		    self.dec_mlp = GaussianMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode, nlayers=self.nlayers_enc, y=self.x2, Weights=dec_weights, dropout_rate=dropout_rate,activation=activation)
                elif args.dec_type == 'bernoulli':
		    self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode, nlayers=self.nlayers_dec, y=self.x2, Weights=dec_weights,dropout_rate=dropout_rate,activation=activation)
	    else:
                if args.dec_type == 'gaussian':
		    self.dec_mlp = GaussianMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode, nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],  y=self.x2, dropout_rate=dropout_rate,activation=activation)
                elif args.dec_type == 'bernoulli':
                    self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode, nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],  y=self.x2, dropout_rate=dropout_rate,activation=activation)

            dec_cost = self.dec_mlp.cost.reshape((self.batch_size,1))
	    if i == 0:
	        self.all_dec_cost = dec_cost
	    else:
		self.all_dec_cost = T.concatenate([self.all_dec_cost,dec_cost],axis=1)

	# p(z|x1)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_prior)
        input_to_prior = self.x1
	self.prior_mlp = GaussianMLP(input_to_prior, self.x1dim, self.hdim_prior, self.zdim, self.training_mode, nlayers=self.nlayers_prior, eps=self.eps, Weights=prior_weights, dropout_rate=dropout_rate,activation=activation)
        
        # make a copy of the decoder to use at test
        # it takes z ~ p(z|x1) 
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec)
        input_to_dec_copy = T.concatenate([self.prior_mlp.mu,self.x1],axis=1)
        if args.dec_type == 'gaussian':
            self.dec_mlp_copy = GaussianMLP(input_to_dec_copy, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode, nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],y=self.x2, dropout_rate=dropout_rate,activation=activation)
        elif args.dec_type == 'bernoulli':
            self.dec_mlp_copy = BernoulliMLP(input_to_dec_copy, self.zdim+self.x1dim, self.hdim_dec, self.x2dim, self.training_mode, nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],y=self.x2, dropout_rate=dropout_rate,activation=activation)

        # q(y|z) 
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_cls)
        input_to_cls = self.prior_mlp.mu   
        self.cls_mlp = CLS(input_to_cls, self.zdim, self.hdim_cls, self.ydim, self.training_mode, nlayers=self.nlayers_cls, Weights=cls_weights, dropout_rate=dropout_rate,activation=activation)
        
	# take average cost across number of samples
	self.dec_mlp_loss = T.mean(self.all_dec_cost,axis=1)
	
        # Model parameters
        self.params = self.enc_mlp.params + self.prior_mlp.params + self.dec_mlp.params + self.cls_mlp.params
        
        # use this to get all networks parameters 
        self.all_params = theano.function(inputs=[],outputs=self.params)
                
        # make a prediction 
        self.y_pred = theano.function(
                inputs  = [self.x1, self.training_mode],
                outputs = self.cls_mlp.y_pred
        )
	
        
        # q(z|x1,x2,y)
        self.draw_z_q = theano.function(
            inputs = [self.x2,self.x1,self.y, self.training_mode],
            outputs = self.enc_mlp.mu
        )

        # p(z|x1)
        self.draw_z_p = theano.function(
            inputs = [self.x1,self.eps, self.training_mode],
            outputs = self.prior_mlp.draw_sample
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
            outputs = [self.enc_mlp.mu, self.enc_mlp.var]
        )

	# p(z|x1)
        self.prior_params = theano.function(
            inputs = [self.x1, self.training_mode],
            outputs = [self.prior_mlp.mu, self.prior_mlp.var]
        )
        
        # q(y|z_prior)
        self.draw_pi = theano.function(
                inputs  = [self.x1, self.training_mode],
                outputs = self.cls_mlp.pi
        )
        

        ''' defining cost function '''
	kl = 0.5*T.sum(T.log(self.prior_mlp.var),axis=1) - 0.5*T.sum(1+T.log(self.enc_mlp.var),axis=1) \
           + T.sum(self.enc_mlp.var/self.prior_mlp.var,axis=1) + 0.5*T.sum(T.square(self.enc_mlp.mu-self.prior_mlp.mu)/self.prior_mlp.var,axis=1)

        self.cls_loss = T.sum(self.alpha*T.nnet.categorical_crossentropy(self.cls_mlp.pi+e,self.y))

	self.kl = kl 
	self.rmse = t_rmse(self.x2,self.dec_mlp.sample) # this is just to see rmse during training

        all_mmd = 0.0
	for i in range(self.samples_z): 
            mmd = compute_mmd(self.prior_mlp.draw_sample[i],self.enc_mlp.draw_sample[i])
            all_mmd+=mmd
        self.mmd = self.lamda * all_mmd/self.samples_z

        loss1 = T.sum(self.kl + self.dec_mlp_loss)/args.batch_size
        loss2 = T.sum(self.dec_mlp_loss)/args.batch_size + self.mmd
        self.loss = self.omega*loss1 + (1.0-self.omega)*loss2 + self.cls_loss


    def training(self,data_x2,data_x1,data_y):
        index = T.lscalar()
        
        print '<<<<<<<<<<< training ... >>>>>>>>>>'
        self.gparams = [T.grad(self.loss, p) for p in self.params]

        self.optimizer = optimizer(self.params,self.gparams,self.lr)
        self.updates   = self.optimizer.adam()
        n_batches = data_x2.get_value().shape[0]/self.batch_size        

        train = theano.function(
            inputs=[index, self.eps,self.omega,self.alpha,self.lamda,self.training_mode],
            outputs=[self.loss, self.dec_mlp_loss, self.kl, self.cls_loss, self.rmse, self.mmd],
            updates=self.updates,
            givens={self.x2: data_x2[index*self.batch_size : (index+1)*self.batch_size],
                    self.x1: data_x1[index*self.batch_size : (index+1)*self.batch_size],     
                    self.y:  data_y[index*self.batch_size : (index+1)*self.batch_size]
                    }
        ) 

        return train
