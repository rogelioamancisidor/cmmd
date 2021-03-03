import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from numba import prange
import numba


'''
helper functions
'''

floatX = theano.config.floatX
e = 1e-8

@numba.njit(fastmath=True)
def logdiagmvn(x, mu, var):
    # expects batches
    k = mu.shape[1]

    logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * np.sum(np.log(var), axis=1) - np.sum(0.5 * (1.0 / var) * (x - mu) * (x - mu), axis=1)
    return logp

@numba.njit(fastmath=True)
def kl_divergence(q_mean, q_logvar, p_mean = 0, p_logvar = 0):
    return 0.5 * (p_logvar - q_logvar + (np.exp(q_logvar) \
                      + (q_mean - p_mean)**2) / np.exp(p_logvar) - 1)

@numba.njit(fastmath=True)
def logsumexp(X,N):
    r = 0.0
    for x in X:
        r += np.exp(x) 
    r = r/N    
    return np.log(r)

@numba.njit(fastmath=True,parallel=True)
def marginal_kl(z,mu,var,zdim, S=1):
    kl=[]
    N = z.shape[0]
    print("calculating the marginal KL numerically ...")
    for s in range(S):
        # iterate over z
        for i in prange(N):
            # broadcasting the ith observation to the same size 
            # as parameters to evaluate it in all qs
            z_s_x = np.zeros(z.shape)
            for j in range(N):
                z_s_x[j,:] = z[i,:]

            # calculate q(z)
            pdf_qz_x = logdiagmvn(z_s_x,mu, var)
            logpdf_qz = logsumexp(pdf_qz_x,N)
            
            # now evaluate the prior
            pdf_pz_s = logdiagmvn(z_s_x,np.zeros(z_s_x.shape),np.ones(z_s_x.shape))
            logpdf_pz_s = logsumexp(pdf_pz_s,N)
            
            # KL, note that we drop the 1/N term using the logsumexp fucnt 
            # since they will cancell anyways
            kl_s = logpdf_qz - logpdf_pz_s

            kl.append(kl_s)

            if i%10000 == 0:
                print(i)

    return kl


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('not a valid boolean value in --balanced parameter')

def nrmse(x_true, x_recon):
    return T.mean(T.square(x_recon - x_true),axis=0)/(T.max(x_true,axis=0)-T.min(x_true,axis=0))


def np_nrmse(x_true, x_recon):
    return np.mean(np.square(x_recon - x_true),axis=0)/(np.max(x_true,axis=0)-np.min(x_true,axis=0))

def normalized_mse(x_true, x_recon):
    diff  = x_true - x_recon
    nom   = T.sqrt(T.sum(T.square(T.abs_(diff)),axis=0))
    denom = T.sqrt(T.sum(T.square(T.abs_(x_true)),axis=0))
    return nom/denom

def nmse(x_true, x_recon):
    diff  = x_true - x_recon
    nom   = np.sqrt(np.sum(np.square(np.abs(diff)),axis=0))
    denom = np.sqrt(np.sum(np.square(np.abs(x_true)),axis=0))
    return nom/denom

def mse(x_true, x_recon):
    return np.sqrt(np.mean(np.square(x_recon - x_true)))

def mean_squared_error(x_true, x_recon):
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

