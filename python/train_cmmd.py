import os
os.environ['KERAS_BACKEND'] = 'theano'

def beta_annealing(epochs,epoch, every_epoch=1, factor = 0.0104, min_beta=1e-09 , max_beta=1,method='linear',hit_maxbeta=None):
    import numpy as np
    
    # two methods, linear and exponential. Snderby 2016 suggests linear.
    if epoch % every_epoch == 0:
        if method == 'linear':
            new_beta = factor*epoch
        elif method == 'linear_decreasing':
            new_beta = factor*(epochs-epoch-1)
        elif method=='exponential':
            new_beta = min_beta*np.exp(factor*epoch)

    # the first method set beta equals to max beta at any epoch greather or equal than hit max beta
    # the second method let beta grows according with the factor, but caps it at max_beta.
    if hit_maxbeta is not None and hit_maxbeta <= epoch:
        new_beta = max_beta
    else:
        new_beta = min(new_beta,max_beta)


    return np.float32(new_beta)

def main():
    # import libraries
    import numpy as np
    import cPickle as pickle
    import time 
    import gzip
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cls import CLS
    from CMMD import CMMD
    from utils import floatX, shared_dataset, np_nrmse, nmse, str_to_bool, mse
    from get_data import load_mnist, load_xrmb, load_xrmb_oneperson, load_all_xrmb, load_flickr, load_mnist_svhn, load_folds_xrmb
    import argparse
    import random
    import time
    from keras.utils import to_categorical
    import os
    import json
    from sklearn.metrics import average_precision_score
    from posterior_collapse import posterior_collapse
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('==========> time stamp =', dt_string, '<==========')

    start = time.time()
    parser = argparse.ArgumentParser() 
   
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--nlayers_enc', default=3, type=int, help='number of hidden layers in encoder MLP before output layers.') 
    parser.add_argument('--nlayers_dec', default=3, type=int, help='number of hidden layers in decoder MLP before output layers.') 
    parser.add_argument('--nlayers_dec2', default=3, type=int, help='number of hidden layers in decoder 2 MLP before output layers.')
    parser.add_argument('--nlayers_cls', default=2, type=int, help='number of hidden layers in classifier MLP.')
    parser.add_argument('--hdim_enc', default=[2500,2500,2500],nargs='+', type=int, help='dimension of hidden layer in enc. Must be a list')
    parser.add_argument('--hdim_dec', default=[1024,1024,1024],nargs='+', type=int, help='dimension of hidden layer in dec. Must be a list')
    parser.add_argument('--hdim_dec2', default=[1024,1024,1024],nargs='+', type=int, help='dimension of hidden layer in dec 2. Must be a list')
    parser.add_argument('--hdim_cls', default=[70,70],nargs='+', type=int, help='dimension of hidden layer in classifier. Must be a list')
    parser.add_argument('--zdim', default=50, type=int, help='dimension of continuous latent variable')
    parser.add_argument('--lr_l', default=0.0001, type=float, help='learning rate for supervised loss')
    parser.add_argument('--epochs', default=1001, type=int, help='number of epochs')
    parser.add_argument('--samples_z',default=1,type=int,help='no. of samples in the monte carlo expectation')
    parser.add_argument('--R', default=1, type=int, help='number of cross validations')
    parser.add_argument('--omega', default=0.5,type=float,help='omega parameter for KL or MMD divergence in the objective function.')
    parser.add_argument('--alpha', default=10,type=float,help='alpha parameter for classifier. This is used as max_alpha if annealing is used.')
    parser.add_argument('--lamda', default=1000,type=float,help='lambda parameter for mmd loss')
    parser.add_argument('--hit_maxbeta', default=1001, type=int, help='epoch at which maxbeta is used when beta_annealing is used.')
    parser.add_argument('--wu_epoch', default=1001,type=int,help='epoch when beta_annealing stops. at this epoch beta equals 1.')
    parser.add_argument('--outfile',default='mnist' ,help='name of file')
    parser.add_argument('--dset',default='mnist',help='choose one data set, eg. mnist, flickr, xrmb etc.')
    parser.add_argument('--dec_type',default='gaussian',help='distribution for the decoder, either gaussian or bernoulli')
    parser.add_argument('--hl_activation',default='softplus',help='activation function for hidden layers in all networks')
    parser.add_argument('--dropout_rate',default=0.2,type=float,help='dropout rate to be used in all hidden layers')
    parser.add_argument('--speaker', default=30, type=int, help='speaker id for speaker-dependet experiments')
    parser.add_argument('--fold', default=0, type=int, help='fold to use for testing, other folds are used for training.')

    args = parser.parse_args()
    

    output_path = '../output'
    full_path   = output_path

    print(args)
    best_rmse = []

    cls_epochs = 200
    recon_x2 = np.zeros((args.epochs,args.R))
    cost_cbmd = np.zeros((args.epochs,args.R))
    err_cbmd  = np.zeros((args.epochs,args.R))
    err_x1  = np.zeros((cls_epochs,args.R))
    prior_var  = np.zeros((args.epochs,args.R))
    posterior_var  = np.zeros((args.epochs,args.R))
    decoder_var  = np.zeros((args.epochs,args.R))

    path = "../output/"+str(args.outfile)

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    r = 0
    while r < args.R:
        if args.dset == 'mnist':
            # original output order
            x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_mnist()
        elif args.dset == 'xrmb_folds':
            x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_folds_xrmb(idx=args.fold)
        elif args.dset == 'xrmb':
            #x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_xrmb(nr_tr_ids = 10, nr_te_ids = 2)
            x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_xrmb_oneperson(idx=args.speaker)
            #x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_all_xrmb()
        elif args.dset == 'xrmb_all':
            x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_all_xrmb()
            xcls = x1[:,117:156]
            xcls_te = x1_te[:,117:156]
        elif args.dset == 'flickr':
            x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = load_flickr()
        elif args.dset == 'mnist_svhn':
            data_set, scaler =  load_mnist_svhn()
            #x1, x2, y_tr, x1_te, x2_te, y_te, x1_cal, x2_cal, y_cal = data_set
            x2, x1, y_tr, x2_te, x1_te, y_te, x2_cal, x1_cal, y_cal = data_set

        if args.dset in ('mnist','mnist_svhn'):
            y_dim = 10
        elif args.dset in ('xrmb','xrmb_all','xrmb_folds'):
            y_dim = 40
        elif args.dset == 'flickr':
            y_dim = 38

        if args.dset != 'flickr':
            y_tr = to_categorical(y_tr,num_classes=y_dim)
            y_te = to_categorical(y_te,num_classes=y_dim)

        print 'size x1_tr, x2_tr and y_tr ', x1.shape, x2.shape, y_tr.shape
        print 'size x1_te', x1_te.shape
        
        # create model instance
        if args.dset == 'xrmb_all':
            model = CBMDxrmb(x2.shape[1],x1.shape[1],args,y_dim)
        else:
            model = CMMD(x2.shape[1],x1.shape[1],args,y_dim)

        data_x1, data_y = shared_dataset((x1,y_tr))
        data_x2  = shared_dataset((x2,))
        n_batches = data_x1.get_value().shape[0]/args.batch_size

        if args.dset == 'xrmb_all':
            data_xcls  = shared_dataset((xcls,))
            train = model.training(data_x2,data_x1,data_y,data_xcls)
        else:
            train = model.training(data_x2,data_x1,data_y)

        prior_var  = np.zeros(args.epochs*n_batches)
        posterior_var  = np.zeros(args.epochs*n_batches)
        decoder_var  = np.zeros(args.epochs*n_batches)
        
        e = 0
        all_mdls = {}
        all_weights = {}

        while e < args.epochs:
            for i in range(n_batches):
                eps  = np.random.randn(args.samples_z,args.batch_size, args.zdim).astype(floatX)
                #factor_a = args.alpha/(args.wu_epoch - 1.0)
                #alpha = beta_annealing(args.epochs, e, factor = factor_a, max_beta=args.alpha, method='linear', hit_maxbeta = args.hit_maxbeta)
                omega  = args.omega
                alpha = args.alpha
                lamda = args.lamda
                vae_cost = train(i,eps,omega,alpha,lamda)
                
                if np.isnan(vae_cost[0]):
                    raise RuntimeError('cost is nan') 
                
            eps  = np.random.randn(args.samples_z,x1_te.shape[0], args.zdim).astype(floatX)
            x2_hat = model.draw_x2(x1_te)
            
            if args.dset == 'xrmb_all':
                label_hat = model.y_pred(x1_te,xcls_te)
            else:
                label_hat = model.y_pred(x1_te)

            recon_x2[e,r]   = mse(x2_te,x2_hat)
            cost_cbmd[e,r] = vae_cost[0]
            
            if args.dset == 'flickr':
                err_cbmd[e,r] = average_precision_score(y_te, model.draw_pi(x1_te),average='micro')
            else:
                err_cbmd[e,r]  = sum(np.sum(y_te != label_hat,axis=1)!=0)/float(y_te.shape[0])


            if e % 100 == 0 and e > 0:
                print 'cost at epoch ', e, ' is ' , round(vae_cost[0],5),  ', decoder ', round(np.mean(vae_cost[1]),5),  ', kl ', round(np.mean(vae_cost[2]),5),  ', cls ', round(np.mean(vae_cost[3]),5), ' test recon ', round(recon_x2[e,r],5), ' train recon ', round(vae_cost[4],5), ' error ', round(err_cbmd[e,r],5) 

            e += 1

        run_cls=True
        if run_cls:
            nr_x1 = data_x1.get_value().shape[1]
            n_out = data_y.get_value().shape[1]
            cls = CLS(nr_x1, args.hdim_cls, nlayers=args.nlayers_cls, n_out=n_out)

            train_cls = cls.train_fn(data_x1, data_y, batch_size=args.batch_size)
            n_batches_cls = data_x1.get_value().shape[0]/args.batch_size        

            for ee in range(cls_epochs):
                for ii in range(n_batches_cls):
                    cost_cls = train_cls(ii)
                label_hat = cls.label_hat(x1_te)
                err_x1[ee,r] = np.mean((np.sum(y_te != label_hat,axis=1)!=0))  

                if ee % 20 == 0 and ee > 0:           
                    print 'cls error at epoch ', ee, ' is ' , err_x1[ee,r]
         
        print 'saving weights ...'
        all_weights = model.all_params()
        np.save(full_path+'/'+str(args.outfile)+'/best_weights_'+str(r)+'.npy',all_weights)

        r+=1
    

    label_hat = model.y_pred(x1_te)
    print 'final error is ', sum(np.sum(y_te != label_hat,axis=1)!=0)/float(y_te.shape[0])


    plt.figure()
    plt.plot(err_x1,label='err_x1')
    plt.grid()
    plt.legend()
    plt.title('err = %0.4f' % np.mean(err_x1[-1,:]))
    plt.savefig(full_path+'/'+str(args.outfile)+'/err_x1_'+str(r)+'.pdf')
    
    plt.figure()
    plt.plot(recon_x2,label='recon_x2')
    plt.grid()
    plt.title('Reconstruction rmse')
    plt.xlabel('epoch')
    plt.savefig(full_path+'/'+str(args.outfile)+'/recon_x2.pdf')

    plt.figure()
    plt.plot(cost_cbmd,label='cost')
    plt.grid()
    plt.title('Cost')
    plt.xlabel('epoch')
    plt.savefig(full_path+'/'+str(args.outfile)+'/cost_cbmd.pdf')
    
    plt.figure()
    plt.plot(err_cbmd)
    plt.grid()
    plt.xlabel('epoch')
    plt.title('Classification error = %0.4f' % np.mean(err_cbmd[-1,:]))
    plt.savefig(full_path+'/'+str(args.outfile)+'/err_cbmd.pdf')
    
    plt.close('all')
    
    with open(full_path+'/'+str(args.outfile)+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    stop = time.time()
    print 'elapsed time ', stop-start


if __name__ == '__main__':
    main()
