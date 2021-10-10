import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from numba import prange
import numba


floatX = theano.config.floatX
e = 1e-8

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('not a valid boolean value in --balanced parameter')

def rmse(x_true, x_recon):
    return np.sqrt(np.mean(np.square(x_recon - x_true)))

def t_rmse(x_true, x_recon):
    return T.sqr(T.mean(T.square(x_recon - x_true)))

def weights_to_gpu(np_weights,n_layers,distribution,components = 2):
    gpu_weights = []
    if distribution == 'Gaussian':
        k = 4
    elif distribution == 'GMM':
        k = 4 * components
    elif distribution in ('Classifier','Bernoulli'):
        k = 2
        
    tot_length_w = 2*n_layers + k # get the length of all weights. +4 only for gaussian!!
    
    layer_i = 0
    mu_counter=0
    sigma_counter=0
    for i in range(len(np_weights)):  
        if i%2 == 0:
            layer = 'W'
            layer_i += 1
        else:
            layer = 'b'
        
        if distribution == 'Gaussian':
            if i < (n_layers*2):     
                name_layer = 'GaussianMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            elif i >= (n_layers*2) and i < (tot_length_w-2):
                name_layer = 'GaussianMLP_mu_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            else:
                name_layer = 'GaussianMLP_logvar_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
        elif distribution == 'Classifier':
            if i < (n_layers*2):     
                name_layer = 'GaussianMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            else:
                name_layer = 'GaussianMLP_cls_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
        elif distribution == 'Bernoulli':
            if i < (n_layers*2):     
                name_layer = 'BernoulliMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            else:
                name_layer = 'BernoulliMLP_x_hat_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
        elif distribution == 'GMM':
            if i < (n_layers*2):     
                name_layer = 'GaussianMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            elif i >= (n_layers*2):
                if mu_counter < 2:
                    name_layer = 'GaussianMLP_mu_'+layer
                    gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
                    mu_counter += 1 
		    if mu_counter == 2:
                        sigma_counter = 0
                else:
                    name_layer = 'GaussianMLP_logvar_'+layer
                    gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
                    sigma_counter += 1
                    if sigma_counter == 2:
                        mu_counter = 0

    return gpu_weights


# XXX dataset parameters
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    if len(data_xy)==2:
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        #return shared_x, T.cast(shared_y, 'int32')
        return shared_x, shared_y
    if len(data_xy)==1:
        data_x   = data_xy[0]

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x
    
# costs
def kld_unit_mvn(mu, var,flag='normal'):
    import numpy as np
    # KL divergence from N(0, I)
    if flag == 'normal':
        return (mu.shape[1] + T.sum(T.log(var), axis=1) - T.sum(T.square(mu), axis=1) - T.sum(var, axis=1)) / 2.0
    else:
        raise RuntimeError('the flag pass into the function is not recognized')

def loss_q_logp(mu1,var1,mu2,var2):
    return -0.5*(T.sum(1+T.log(var1+e))) + 0.5*(T.sum(T.log(var2+e),axis=1) + T.sum((var1+e)/(var2+e),axis=1) + T.sum((1.0 / (var2+e)) * (mu1 - mu2)*(mu1 - mu2),axis=1))

def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = mu.shape[1]

        logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * T.sum(T.log(var), axis=1) - T.sum(0.5 * (1.0 / var) * (x - mu) * (x - mu), axis=1)
        return logp
    return f

def compute_kernel(x,z, ndim=3):
    x_size = T.shape(x)[0]
    z_size = T.shape(z)[0]
    dim    = T.shape(x)[1]

    tiled_x =  T.tile( T.reshape(x,  [x_size, 1, dim]),  [1, z_size, 1], ndim=ndim)
    tiled_z =  T.tile( T.reshape(z,  [1, z_size, dim]), [x_size, 1, 1], ndim=ndim)
    
    return T.exp(-T.mean(T.sqr(tiled_x - tiled_z), axis=2) / T.cast(dim, 'float32'))

def compute_mmd(x, z, ndim=3):
    x_kernel  = compute_kernel(x, x, ndim)
    z_kernel  = compute_kernel(z, z, ndim)
    xz_kernel = compute_kernel(x, z, ndim)
    
    return T.mean(x_kernel) + T.mean(z_kernel) - 2 * T.mean(xz_kernel)

def kernel_np(x, z):
    x_size = x.shape[0]
    z_size = z.shape[0]
    dim    = x.shape[1]

    tiled_x =  np.tile( np.reshape(x,  [x_size, 1, dim]),  [1, z_size, 1])
    tiled_z =  np.tile( np.reshape(z,  [1, z_size, dim]), [x_size, 1, 1])
    
    return np.exp(-np.mean(np.square(tiled_x - tiled_z), axis=2) / dim)

def mmd_np(x,z):
    x_kernel  = kernel_np(x, x)
    z_kernel  = kernel_np(z, z)
    xz_kernel = kernel_np(x, z)
    
    return np.mean(x_kernel) + np.mean(z_kernel) - 2 * np.mean(xz_kernel)

