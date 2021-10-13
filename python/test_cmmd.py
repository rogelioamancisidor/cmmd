import os
os.environ['KERAS_BACKEND'] = 'theano'
from utils import weights_to_gpu,  rmse, floatX
from CMMD import CMMD
import argparse
import numpy as np
import json
import cPickle as pickle
from get_data import load_mnist, load_xrmb, load_xrmb_oneperson, load_all_xrmb, load_flickr, get_id_xrmb, load_folds_xrmb
from keras.utils import to_categorical
from posterior_collapse import posterior_collapse, variance_collapse

def test_cmmd(folder_name,idx=0):
    # idx refers to which weights to be used in case there is more than one set 
    r = folder_name[-1] 
    # load parameters
    parser = argparse.ArgumentParser() 
    args = parser.parse_args()

    # define training mode
    training_mode = 0

    with open('../output/'+folder_name+'/commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    print('loading data set for {}'.format(args.dset))
    
    # give dimensions
    if args.dset == 'mnist':
        y_dim = 10
        x1_dim = 28*28
        x2_dim = x1_dim
        _, _, _, x1_te, x2_te, y_te, _, _, _ = load_mnist(path='../../cbmd_barney/data/MNIST.mat')
        y_te = to_categorical(y_te,num_classes=y_dim)
    elif args.dset == 'flickr':
        y_dim = 38
        x1_dim = 3857
        x2_dim = 2000
        _, _, _, x1_te, x2_te, y_te, _, _, _ = load_flickr()
    elif args.dset in ('xrmb','xrmb_all'):
        y_dim = 40
        x1_dim = 273
        x2_dim = 112
    elif args.dset=='xrmb_folds':
        y_dim = 40
        x1_dim = 273
        x2_dim = 112
        print('fold {} loaded'.format(args.fold))
        x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_folds_xrmb(idx=args.fold)
    
    # extract weights
    weights_path = '../output/'+folder_name+'/weights_'+str(idx)+'.npy'
    weights = np.load(weights_path)
    # order is enc(10), prior(10), dec(10), cls(6)

    if args.dec_type == 'gaussian':
        enc_weights  = weights[0:10]    
        prior_weights = weights[10:20] 
        dec_weights  = weights[20:30]
        cls_weights  = weights[30:]

        # specify the decoder distribution. Only Bernoulli for flickr data
        print 'loading weights to gpu ...'
        enc_weights  = [weights_to_gpu(enc_weights,args.nlayers_enc,'Gaussian'), True]
        prior_weights = [weights_to_gpu(prior_weights,args.nlayers_prior,'Gaussian'),True]
        dec_weights  = [weights_to_gpu(dec_weights,args.nlayers_dec,'Gaussian'),True]
        cls_weights  = [weights_to_gpu(cls_weights,args.nlayers_cls,'Classifier'),True]
    elif args.dec_type == 'bernoulli':
        enc_weights  = weights[0:10]    
        prior_weights = weights[10:20] 
        dec_weights  = weights[20:28]
        cls_weights  = weights[28:]

        # specify the decoder distribution. Only Bernoulli for flickr data
        print 'loading weights to gpu ...'
        enc_weights  = [weights_to_gpu(enc_weights,args.nlayers_enc,'Gaussian'), True]
        prior_weights = [weights_to_gpu(prior_weights,args.nlayers_prior,'Gaussian'),True]
        dec_weights  = [weights_to_gpu(dec_weights,args.nlayers_dec,'Bernoulli'),True]
        cls_weights  = [weights_to_gpu(cls_weights,args.nlayers_cls,'Classifier'),True]

    # load model with trained weights
    print 'creating CMMD model with pre-trained weights ...'
    model = CMMD(x2_dim
                ,x1_dim
                ,args
                ,y_dim
                ,enc_weights  = enc_weights
                ,prior_weights = prior_weights
                ,dec_weights  = dec_weights
                ,cls_weights  = cls_weights
                )
    
    if args.dset == 'mnist':
        label_hat = model.y_pred(x1_te, training_mode)
        print('error rate {:.4f}'.format(sum(np.sum(y_te != label_hat,axis=1)!=0)/float(y_te.shape[0])))
    elif args.dset in ('xrmb'):
        speaker_avg =[]
        speakers = [7,16,20,21,23,28,31,35]
        
        _, _, _, x1_te, x2_te, y_te, _, _, _ = load_all_xrmb(path='../../cbmd_barney/data/',idx_te=speakers)
        y_te = to_categorical(y_te,num_classes=y_dim)
        label_hat = model.y_pred(x1_te, training_mode)
        print('error rate for all speakers {:.5f}'.format(np.mean(np.sum(y_te != label_hat,axis=1)!=0)))

        _,id_te,_ = get_id_xrmb(path='../../cbmd_barney/data/')
        for speaker in speakers:
            print('error rate for speaker {} is {:.5f}'.format(speaker,np.mean(np.sum(y_te[id_te[:,0]==speaker,:] != label_hat[id_te[:,0]==speaker,:],axis=1)!=0)))
            x2_hat = model.draw_x2(x1_te[id_te[:,0]==speaker,:], training_mode)
            print('rmse for speaker {} is {:.5f}'.format(speaker,rmse(x2_te[id_te[:,0]==speaker,:],x2_hat)))
    elif args.dset == 'xrmb_folds':
        # posterior collapse
        y_te = to_categorical(y_te,num_classes=y_dim)
        z_mu, z_var = model.enc_params(x2_te,x1_te,y_te, training_mode)
        mu0, var0   = model.prior_params(x1_te, training_mode)

        eps1,collapse1 = posterior_collapse(mu0, np.log(var0),z_mu, np.log(z_var), path = '../output/'+folder_name+'/postcollapse_v1_te'+str(r)+'.pdf')
        eps2,collapse2 = posterior_collapse(mu0, np.log(var0), path = '../output/'+folder_name+'/postcollapse_v2_te'+str(r)+'.pdf')
        _, x2_var = model.dec_params(x1_te, training_mode)
        eps3,collapse3 = variance_collapse(x2_var, path = '../output/'+folder_name+'/variance_collapse_te'+str(r)+'.pdf')
        eps4,collapse4 = posterior_collapse(z_mu, np.log(z_var), path = '../output/'+folder_name+'/postcollapse_v3_te'+str(r)+'.pdf')

        dic1 = {'eps':eps1,'collapse':collapse1}
        dic2 = {'eps':eps2,'collapse':collapse2}
        dic3 = {'eps':eps3,'collapse':collapse3}
        dic4 = {'eps':eps4,'collapse':collapse4}
        with open('../output/'+folder_name+'/cmmd'+str(r)+'_collapse1.pk','wb') as f:
            pickle.dump(dic1,f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../output/'+folder_name+'/cmmd'+str(r)+'_collapse2.pk','wb') as f:
            pickle.dump(dic2,f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../output/'+folder_name+'/cmmd'+str(r)+'_collapse3.pk','wb') as f:
            pickle.dump(dic3,f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../output/'+folder_name+'/cmmd'+str(r)+'_collapse4.pk','wb') as f:
            pickle.dump(dic4,f, protocol=pickle.HIGHEST_PROTOCOL)
        

        eps  = np.random.randn(args.samples_z,x1_te.shape[0], args.zdim).astype(floatX)
        x2_hat = model.draw_x2(x1_te, training_mode)
        print('rmse x2: {:.4}'.format(rmse(x2_te,x2_hat)))


def main():
    import warnings
    warnings.filterwarnings("ignore")
    
    # XXX  mnist
    test_cmmd('mnist',idx=0)

    # XXX xrmb experiment
    test_cmmd('xrmb',idx=0)

if __name__ == '__main__':
    main()
