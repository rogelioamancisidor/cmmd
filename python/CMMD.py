import numpy as np
import theano
import theano.tensor as T
from mlp_dropout import GaussianMLP, BernoulliMLP, CLS
from utils import kld_unit_mvn, shared_dataset,log_diag_mvn, floatX, loss_q_logp, nrmse, normalized_mse, compute_mmd, compute_kernel, mean_squared_error
from optimizers import optimizer

#
e = 1e-8
class CMMD(object):
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
	self.omega  = T.fscalar('omega')
        self.alpha = T.fscalar('alpha')
        self.lamda = T.fscalar('lamda')

        if args.hl_activation == 'softplus':
            activation = T.nnet.nnet.softplus
        elif args.hl_activation == 'relu':
            activation = T.nnet.relu

        # q(z|x2,x1,y)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_enc)
        input_to_enc = T.concatenate([self.x2,self.x1, self.y],axis=1)                           
        self.enc_mlp = GaussianMLP(input_to_enc, self.x2dim+self.x1dim+self.ydim, self.hdim_enc, self.zdim, nlayers=self.nlayers_enc, eps=self.eps, Weights=enc_weights, dropout_rate=dropout_rate, activation=activation)
	
        # q(z|x2,x1,y)
        dropout_rate = np.tile(0,args.nlayers_enc)
        self.enc_mlp_copy = GaussianMLP(input_to_enc, self.x2dim+self.x1dim+self.ydim, self.hdim_enc, self.zdim, nlayers=self.nlayers_enc, eps=self.eps, Weights=[self.enc_mlp.params,True],dropout_rate=dropout_rate,activation=activation)
        
        # p(x2|z,x1)
	# loop trough each samples of z
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec)
	for i in range(self.samples_z): 
	    input_to_dec = T.concatenate([self.enc_mlp.draw_sample[i],self.x1],axis=1)                              
     
	    if i == 0:
                if args.dec_type == 'gaussian':
		    self.dec_mlp = GaussianMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim,nlayers=self.nlayers_dec, y=self.x2, Weights=dec_weights,dropout_rate=dropout_rate,activation=activation)
                elif args.dec_type == 'bernoulli':
		    self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim,nlayers=self.nlayers_dec, y=self.x2, Weights=dec_weights,dropout_rate=dropout_rate,activation=activation)
	    else:
                if args.dec_type == 'gaussian':
		    self.dec_mlp = GaussianMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],  y=self.x2, dropout_rate=dropout_rate,activation=activation)
                elif args.dec_type == 'bernoulli':
                    self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.x1dim, self.hdim_dec, self.x2dim,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],  y=self.x2, dropout_rate=dropout_rate,activation=activation)

            dec_cost = self.dec_mlp.cost.reshape((self.batch_size,1))
	    if i == 0:
	        self.all_dec_cost = dec_cost
	    else:
		self.all_dec_cost = T.concatenate([self.all_dec_cost,dec_cost],axis=1)

	# p(z|x1)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec2)
        input_to_dec2 = self.x1
	self.dec2_mlp = GaussianMLP(input_to_dec2, self.x1dim, self.hdim_dec2, self.zdim, nlayers=self.nlayers_dec2, eps=self.eps, Weights=dec2_weights, dropout_rate=dropout_rate,activation=activation)
        
	# copy pf p(z|x1) to use at test
        dropout_rate = np.tile(0,args.nlayers_dec2)
	self.dec2_mlp_copy = GaussianMLP(input_to_dec2, self.x1dim, self.hdim_dec2, self.zdim, nlayers=self.nlayers_dec2, eps=self.eps, Weights=[self.dec2_mlp.params,True], dropout_rate=dropout_rate, activation=activation)
        
        # make a copy of the decoder to use at test
        dropout_rate = np.tile(0,args.nlayers_dec)
        input_to_dec_copy = T.concatenate([self.dec2_mlp.mu,self.x1],axis=1)
        if args.dec_type == 'gaussian':
            self.dec_mlp_copy = GaussianMLP(input_to_dec_copy, self.zdim+self.x1dim, self.hdim_dec, self.x2dim,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],y=self.x2, dropout_rate=dropout_rate,activation=activation)
        elif args.dec_type == 'bernoulli':
            self.dec_mlp_copy = BernoulliMLP(input_to_dec_copy, self.zdim+self.x1dim, self.hdim_dec, self.x2dim,nlayers=self.nlayers_dec, Weights=[self.dec_mlp.params,True],y=self.x2, dropout_rate=dropout_rate,activation=activation)

        # q(y|z) 
        dropout_rate = np.tile(0,args.nlayers_cls)
        input_to_cls = self.dec2_mlp.mu   
        self.cls_mlp = CLS(input_to_cls, self.zdim, self.hdim_cls, self.ydim, nlayers=self.nlayers_cls,Weights=cls_weights, dropout_rate=dropout_rate,activation=activation)
        
        input_to_cls_copy = self.enc_mlp.mu   
        self.cls_mlp_copy = CLS(input_to_cls_copy, self.zdim, self.hdim_cls, self.ydim, nlayers=self.nlayers_cls,Weights=[self.cls_mlp.params,True],dropout_rate=dropout_rate,activation=activation)

	# take average cost across number of samples
	self.dec_mlp_cost = T.mean(self.all_dec_cost,axis=1)
	
        # Model parameters
        self.params = self.enc_mlp.params + self.dec2_mlp.params + self.dec_mlp.params + self.cls_mlp.params
        

        # use this to get all networks parameters 
        self.all_params = theano.function(inputs=[],outputs=self.params)
                
        
        self.y_pred = theano.function(
                inputs  = [self.x1],
                outputs = self.cls_mlp.y_pred
        )
	
        
        # q(z|x1,x2,y)
        self.draw_z_q = theano.function(
            inputs = [self.x2,self.x1,self.y,self.eps],
            outputs = self.enc_mlp_copy.draw_sample
        )

        # p(z|x1)
        self.draw_z_p = theano.function(
            inputs = [self.x1,self.eps],
            outputs = self.dec2_mlp_copy.draw_sample
        )
	
        # p(x2|z,x1)
        self.draw_x2 = theano.function(
            inputs = [self.x1],
            outputs = self.dec_mlp_copy.sample
        )

        # p(x2|z, x1)
        self.dec_params = theano.function(
            inputs = [self.x1],
            outputs = [self.dec_mlp_copy.mu, self.dec_mlp_copy.var]
        )
	
        # q(z|x1,x2,y)
        self.enc_params = theano.function(
            inputs = [self.x2,self.x1, self.y],
            outputs = [self.enc_mlp_copy.mu, self.enc_mlp_copy.var]
        )

	# p(z|x1)
        self.prior_params = theano.function(
            inputs = [self.x1],
            outputs = [self.dec2_mlp_copy.mu, self.dec2_mlp_copy.var]
        )
        
        # q(y|z_prior)
        self.draw_pi = theano.function(
                inputs  = [self.x1],
                outputs = self.cls_mlp.pi
        )
        
        # q(y|z_posterior)
        self.draw_pi_posterior = theano.function(
                inputs  = [self.x2,self.x1,self.y],
                outputs = self.cls_mlp_copy.pi
        )


        ''' defining cost function '''
	kl = 0.5*T.sum(T.log(self.dec2_mlp.var),axis=1) - 0.5*T.sum(1+T.log(self.enc_mlp.var),axis=1) + T.sum(self.enc_mlp.var/self.dec2_mlp.var,axis=1) + 0.5*T.sum(T.square(self.enc_mlp.mu-self.dec2_mlp.mu)/self.dec2_mlp.var,axis=1)
        self.cls_cost = T.sum(self.alpha*T.nnet.categorical_crossentropy(self.cls_mlp.pi+e,self.y))

	self.kl = kl 
	self.nrmse = mean_squared_error(self.x2,self.dec_mlp.sample) # this is just to see rmse during training

        all_mmd = 0.0
	for i in range(self.samples_z): 
            mmd = compute_mmd(self.dec2_mlp.draw_sample[i],self.enc_mlp.draw_sample[i])
            all_mmd+=mmd
        self.mmd = self.lamda * all_mmd/self.samples_z

        cost1 = T.sum(self.kl + self.dec_mlp_cost)/args.batch_size
        cost2 = T.sum(self.dec_mlp_cost)/args.batch_size + self.mmd
        self.vae_cost = self.omega*cost1 + (1.0-self.omega)*cost2 + self.cls_cost


    def training(self,data_x2,data_x1,data_y):
        index = T.lscalar()
        
        print '<<<<<<<<<<< training ... >>>>>>>>>>'
        self.gparams = [T.grad(self.vae_cost, p) for p in self.params]

        self.optimizer = optimizer(self.params,self.gparams,self.lr)
        self.updates   = self.optimizer.adam()
        n_batches = data_x2.get_value().shape[0]/self.batch_size        

        train = theano.function(
            inputs=[index, self.eps,self.omega,self.alpha,self.lamda],
            outputs=[self.vae_cost, self.dec_mlp_cost, self.kl, self.cls_cost, self.nrmse, self.mmd],
            updates=self.updates,
            givens={self.x2: data_x2[index*self.batch_size : (index+1)*self.batch_size],
                    self.x1: data_x1[index*self.batch_size : (index+1)*self.batch_size],     
                    self.y:  data_y[index*self.batch_size : (index+1)*self.batch_size]
                    }
        ) 

        return train
