def fixlabels(x):
    same_format = [1+i if i <9 else 0 for i in x]
    return same_format

def fsorted(listval):
    import numpy as np

    listvalf = np.copy(listval) 
    from operator import itemgetter
    indices, L_sorted = zip(*sorted(enumerate(listvalf), key=itemgetter(1)))
    return list(L_sorted), list(indices)

def match_3m(target_val, values):
    from collections import Counter
    import numpy as np

    # sort both lists
    t_v, t_i = fsorted(target_val)
    s_v, s_i = fsorted(values)

    # counts. This are the number of obs that we need per digit 
    counts = list(Counter(t_v).values())

    need_idx = []
    for i in np.unique(t_v):
        l_idx = [j for j, e in enumerate(values) if e == i]
        count = counts[i]
        l_idx = l_idx[:count]

        # get values that you need
        need_idx.extend(l_idx)
    # the 1st indices tells where to find the same digit in the 2 list (values)
    # the 2nd indices tells where to marge them to obtain a 3 modal data set
    return need_idx, t_i

def load_mnist(path='../data/MNIST.mat', shuffle_data=True):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.utils import shuffle
    
    data = loadmat(path)

    y_tr  = data['trainLabel']
    y_te  = data['testLabel']
    y_val  = data['tuneLabel']

    y_tr = y_tr.reshape(-1,)
    y_te = y_te.reshape(-1,)
    y_val = y_val.reshape(-1,)
    
    x1_tr = data['X1']
    x2_tr = data['X2']
    
    if shuffle_data:
        x1_tr,x2_tr,y_tr = shuffle(x1_tr,x2_tr,y_tr)
    
    x1_te  = data['XTe1']
    x2_te  = data['XTe2']
    
    x1_val  = data['XV1']
    x2_val  = data['XV2']

    # substract 1 to make labels start at 0
    all_data = (x1_tr.astype(np.float32),
                x2_tr.astype(np.float32),
                y_tr-1,
                x1_te.astype(np.float32),
                x2_te.astype(np.float32),
                y_te-1,
                x1_val.astype(np.float32),
                x2_val.astype(np.float32),
                y_val-1
                )

    return all_data

def load_xrmb_oneperson(path='../data/',idx=30):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn import preprocessing
    
    
    data_x1 = loadmat(path+'XRMB1.mat')
    data_x2 = loadmat(path+'XRMB2.mat')
    # keys x1 'XTe1', 'XV1', 'X1'
    # keys x2: 'trainID', 'trainLabel', 'XTe2', 'testLabel', 
    # 'XV2', 'testID', 'X2', 'tuneID', 'tuneLabel'
    x1 = data_x1['X1']
    x2 = data_x2['X2']
    y  = data_x2['trainLabel']
    x1_te = data_x1['XTe1']
    x2_te = data_x2['XTe2']
    y_te  = data_x2['testLabel']
    x1 = np.r_[x1,x1_te]
    x2 = np.r_[x2,x2_te]
    y = np.r_[y,y_te]
    
    person_id = np.r_[data_x2['trainID'],data_x2['testID']]
    print('person id is {} '.format(idx))
   
    bool_tr = np.array([any(idx==i) for i in person_id])

    x1 = x1[bool_tr,:]
    x2 = x2[bool_tr,:]
    y  = y[bool_tr,:]
    y  = y.reshape(-1,)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index, test_index in sss.split(x1, y):
        x1_tr, x1_te = x1[train_index], x1[test_index]
        x2_tr, x2_te = x2[train_index], x2[test_index]
        y_tr , y_te  =  y[train_index],  y[test_index]

    # XXX change this part  
    x1_val = x1_te[0:3000,:]
    x2_val = x2_te[0:3000,:]
    y_val  = y_te[0:3000]

    # XXX check if shuffling the data harms model training
    #x1_tr,x2_tr,y_tr = shuffle(x1_tr,x2_tr,y_tr)
    
    all_data = (x1_tr.astype(np.float32),
                x2_tr.astype(np.float32),
                y_tr,
                x1_te.astype(np.float32),
                x2_te.astype(np.float32),
                y_te,
                x1_val.astype(np.float32),
                x2_val.astype(np.float32),
                y_val
                )

    return all_data

def batch_first(data):
    new_data = np.zeros((data.shape[3],data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[3]):
        new_data[i,:,:,:] = data[:,:,:,i]
    return new_data

def load_mnist_3m(path='../data/',nr_tr=30000,nr_te=7000):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn import preprocessing
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mnist_tr_idx = np.load('../data/mnist_idx_tr.npy')
    mnist_te_idx = np.load('../data/mnist_idx_te.npy')
    svhn_tr_idx = np.load('../data/svhn_idx_tr.npy')
    svhn_te_idx = np.load('../data/svhn_idx_te.npy')
    

    shvm_path = path+'mnist_shvn/'
    shvm_te = loadmat(shvm_path+'test_32x32.mat')
    x2_te = shvm_te['X']
    x2_te = batch_first(x2_te)
    shvm_tr = loadmat(shvm_path+'train_32x32.mat')
    x2_tr = shvm_tr['X']
    x2_tr = batch_first(x2_tr)

    x1_tr = np.load(path+'mnist_tr.npy')
    x1_te = np.load(path+'mnist_te.npy')
    y_tr = np.load(path+'mnist_y_tr.npy')
    y_te = np.load(path+'mnist_y_te.npy')
    
    '''
    seed = 325162
    idx_seed = mnist_tr_idx[seed]
    print('label', y_tr[mnist_tr_idx[seed]])
    plt.imshow(x2_tr[svhn_tr_idx[seed],:,:,:])
    plt.savefig('svhn1.pdf')
    plt.imshow(x1_tr[mnist_tr_idx[seed],:,:])
    plt.savefig('mnist1.pdf')
    '''
    
    # reshape mnist
    x1_te = x1_te.reshape((x1_te.shape[0], x1_te.shape[1]*x1_te.shape[2])) 
    x1_tr = x1_tr.reshape((x1_tr.shape[0], x1_tr.shape[1]*x1_tr.shape[2])) 
    
    #scale 
    x3_tr = x2_tr/255.
    x3_te = x2_te/255.

    #plt.imshow(x3_tr[svhn_tr_idx[seed],:,:,:])
    #plt.savefig('svhn12.pdf')
    
    # randomly selected idx
    idx_tr_val = random.sample(range(svhn_tr_idx.shape[0]),nr_tr)
    idx_te_val = random.sample(range(svhn_te_idx.shape[0]),nr_te)

    # get labels
    y_tr_f = y_tr[mnist_tr_idx[idx_tr_val]]
    y_te_f = y_te[mnist_te_idx[idx_te_val]]

    # load mnist VCCA version to use as 3rd modality
    # this is the rotated version of the digit
    x2_tr,_,y2_tr,x2_te,_,y2_te,_,_,_ = load_mnist()
    y2_tr = fixlabels(y2_tr) 
    y2_te = fixlabels(y2_te) 
    
    need_idx, t_i = match_3m(y_tr_f, y2_tr)
    # get vales and labels in sorted fashion
    s_v = x2_tr[need_idx]
    # add indexing from y_tr_f
    s_v_o = np.c_[t_i,s_v]
    # sort based on t_i 
    x2_tr = s_v_o[s_v_o[:,0].argsort()][:,1:]

    #print(y2_te[0:20])
    #print('sending te to sortf')
    need_idx, t_i = match_3m(y_te_f, y2_te)
    # get vales and labels in sorted fashion
    s_v = x2_te[need_idx]
    # add indexing from y_tr_f
    s_v_o = np.c_[t_i,s_v]
    # sort based on t_i 
    x2_te = s_v_o[s_v_o[:,0].argsort()][:,1:]

    
    all_data = (x1_tr[mnist_tr_idx[idx_tr_val],:].astype(np.float32),
                x2_tr.astype(np.float32),
                x3_tr[svhn_tr_idx[idx_tr_val],:].astype(np.float32),
                y_tr[mnist_tr_idx[idx_tr_val]],
                x1_te[mnist_te_idx[idx_te_val],:].astype(np.float32),
                x2_te.astype(np.float32),
                x3_te[svhn_te_idx[idx_te_val],:].astype(np.float32),
                y_te[mnist_te_idx[idx_te_val]]
                )

    return all_data

def load_mnist_svhn(path='../data/',nr_tr=80000,nr_te=20000):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn import preprocessing
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MinMaxScaler
    
    mnist_tr_idx = np.load('../data/mnist_idx_tr.npy')
    mnist_te_idx = np.load('../data/mnist_idx_te.npy')
    svhn_tr_idx = np.load('../data/svhn_idx_tr.npy')
    svhn_te_idx = np.load('../data/svhn_idx_te.npy')
    

    shvm_path = path+'mnist_shvn/'
    shvm_te = loadmat(shvm_path+'test_32x32.mat')
    x2_te = shvm_te['X']
    #y2_te = shvm_te['y']
    shvm_tr = loadmat(shvm_path+'train_32x32.mat')
    x2_tr = shvm_tr['X']
    #y2_tr = shvm_tr['y']

    x1_tr = np.load(path+'mnist_tr.npy')
    x1_te = np.load(path+'mnist_te.npy')
    y_tr = np.load(path+'mnist_y_tr.npy')
    y_te = np.load(path+'mnist_y_te.npy')
    y_tr = y_tr 
    y_te = y_te
    
    # reshape mnist
    x1_te = x1_te.reshape((x1_te.shape[0], x1_te.shape[1]*x1_te.shape[2])) 
    x1_tr = x1_tr.reshape((x1_tr.shape[0], x1_tr.shape[1]*x1_tr.shape[2])) 
    
    # reshape shvm
    x2_te = x2_te.reshape(32*32*3, x2_te.shape[-1])
    x2_te = x2_te.transpose()
    x2_tr = x2_tr.reshape(32*32*3, x2_tr.shape[-1])
    x2_tr = x2_tr.transpose()
    
    #scaler 
    scaler = MinMaxScaler()
    x1_tr = scaler.fit_transform(x1_tr)
    x1_te = scaler.fit_transform(x1_te)
    x2_tr = scaler.fit_transform(x2_tr)
    x2_te = scaler.fit_transform(x2_te)

    # randomly selected idx
    idx_tr_val = random.sample(range(svhn_tr_idx.shape[0]),nr_tr)
    idx_te_val = random.sample(range(svhn_te_idx.shape[0]),nr_te)

    all_data = (x1_tr[mnist_tr_idx[idx_tr_val],:].astype(np.float32),
                x2_tr[svhn_tr_idx[idx_tr_val],:].astype(np.float32),
                y_tr[mnist_tr_idx[idx_tr_val]],
                x1_te[mnist_te_idx[idx_te_val],:].astype(np.float32),
                x2_te[svhn_te_idx[idx_te_val],:].astype(np.float32),
                y_te[mnist_te_idx[idx_te_val]],
                x1_te[mnist_te_idx[idx_te_val],:].astype(np.float32),
                x2_te[svhn_te_idx[idx_te_val],:].astype(np.float32),
                y_te[mnist_te_idx[idx_te_val]]
                )

    return all_data

def get_id_xrmb(path='../data/'):
    from scipy.io import loadmat
    
    data_x2 = loadmat(path+'XRMB2.mat')
    # keys x1 'XTe1', 'XV1', 'X1'
    # keys x2: 'trainID', 'trainLabel', 'XTe2', 'testLabel', 'XV2', 'testID', 'X2', 'tuneID', 'tuneLabel'

    id_tr  = data_x2['trainID']
    id_te  = data_x2['testID'] 
    id_val = data_x2['tuneID']

    return id_tr, id_te, id_val

def load_folds_xrmb(path='../data/',idx=0):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn import preprocessing
    from sklearn.utils import shuffle
    
    data_x1 = loadmat(path+'XRMB1.mat')
    data_x2 = loadmat(path+'XRMB2.mat')
    # keys x1 'XTe1', 'XV1', 'X1'
    # keys x2: 'trainID', 'trainLabel', 'XTe2', 'testLabel', 'XV2', 'testID', 'X2', 'tuneID', 'tuneLabel'
    x1_te = data_x1['XTe1']
    x2_te = data_x2['XTe2']
    id_te  = data_x2['testID'] 
    y_te   = data_x2['testLabel']
    
    x1_tr = data_x1['X1']
    x2_tr = data_x2['X2']
    id_tr  = data_x2['trainID'] 
    y_tr   = data_x2['trainLabel']

    x1_all = np.r_[x1_te,x1_tr]
    x2_all = np.r_[x2_te,x2_tr]
    id_all = np.r_[id_te,id_tr]
    y_all = np.r_[y_te,y_tr]
    
    #ids_all = [[7,16],[20,21],[23,28],[31,35]]
    # XXX I remove speaker 23 and 28, it seems they have 
    # some data isssues making the optimization unstable
    ids_all = [[1,3],[43,45],[10,13],[27,29]]
    idx_te = ids_all[idx]
    idx_tr = []
    if idx == 0:
        idx_tr.extend(ids_all[1])
        idx_tr.extend(ids_all[2])
        idx_tr.extend(ids_all[3])
    elif idx == 1:
        idx_tr.extend(ids_all[0])
        idx_tr.extend(ids_all[2])
        idx_tr.extend(ids_all[3])
    elif idx == 2:
        idx_tr.extend(ids_all[0])
        idx_tr.extend(ids_all[1])
        idx_tr.extend(ids_all[3])
    elif idx == 3:
        idx_tr.extend(ids_all[0])
        idx_tr.extend(ids_all[1])
        idx_tr.extend(ids_all[2])
                
    print 'using ids', idx_tr, 'for training'
    print 'and ids', idx_te, 'for testing'

    # boolean vector
    bool_tr = np.array([any(idx_tr==i) for i in id_all])
    bool_te = np.array([any(idx_te==i) for i in id_all])
    
    # train
    x1_tr = x1_all[bool_tr,:]
    x2_tr = x2_all[bool_tr,:]
    y_tr  = y_all[bool_tr,:]
    
    # test
    x1_te = x1_all[bool_te,:]
    x2_te = x2_all[bool_te,:]
    y_te  = y_all[bool_te,:]
    
    # reshape
    y_tr  = y_tr.reshape(-1,)
    y_te  = y_te.reshape(-1,)

    # rename val as test
    x1_val = x1_te
    x2_val = x2_te
    y_val  = y_te

    # shuffle training data
    #x1_tr,x2_tr,y_tr = shuffle(x1_tr,x2_tr,y_tr)

    all_data = (x1_tr.astype(np.float32),
                x2_tr.astype(np.float32),
                y_tr,
                x1_te.astype(np.float32),
                x2_te.astype(np.float32),
                y_te,
                x1_val.astype(np.float32),
                x2_val.astype(np.float32),
                y_val
                )

    return all_data

def load_all_xrmb(path='../data/',idx_te=None):
    from scipy.io import loadmat
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn import preprocessing
    from sklearn.utils import shuffle
    
    data_x1 = loadmat(path+'XRMB1.mat')
    data_x2 = loadmat(path+'XRMB2.mat')
    # keys x1 'XTe1', 'XV1', 'X1'
    # keys x2: 'trainID', 'trainLabel', 'XTe2', 'testLabel', 'XV2', 'testID', 'X2', 'tuneID', 'tuneLabel'
    x1_tr = data_x1['X1']
    x1_te = data_x1['XTe1']
    x1_val = data_x1['XV1']

    x2_tr = data_x2['X2']
    x2_te = data_x2['XTe2']
    x2_val = data_x2['XV2']
    
    id_tr  = data_x2['trainID']
    id_te  = data_x2['testID'] 
    id_val = data_x2['tuneID']
    
    y_tr   = data_x2['trainLabel']
    y_te   = data_x2['testLabel']
    y_val  = data_x2['tuneLabel']
    
    # merge test and tune (10 persons) and select randomly
    ids = np.r_[id_te, id_val]
    idx_tr = random.sample(np.unique(ids),8)
    #idx_te = np.setdiff1d(ids,idx_tr)
    if idx_te is None:
        idx_te = [20,28]
    #print 'adding ids ', idx_tr, ' to training data'
    print 'using  id ', idx_te, ' for testing'

    # boolean vector
    bool_tr = np.array([any(idx_tr==i) for i in ids])
    bool_te = np.array([any(idx_te==i) for i in ids])
    
    # add the selected 8 persons to training
    all_x1 = np.r_[x1_te,x1_val]
    all_x2 = np.r_[x2_te,x2_val]
    all_y  = np.r_[y_te,y_val]

    new_x1 = all_x1[bool_tr,:]
    new_x2 = all_x2[bool_tr,:]
    new_y  = all_y[bool_tr,:]

    #x1_tr = np.r_[x1_tr, new_x1]
    #x2_tr = np.r_[x2_tr, new_x2]
    #y_tr  = np.r_[y_tr, new_y]

    x1_te = all_x1[bool_te,:]
    x2_te = all_x2[bool_te,:]
    y_te  = all_y[bool_te,:]
    
    # reshape
    y_tr  = y_tr.reshape(-1,)
    y_te  = y_te.reshape(-1,)
    y_val  = y_val.reshape(-1,)

    # rename val as test
    x1_val = x1_te
    x2_val = x2_te
    y_val  = y_te

    # shuffle training data
    #x1_tr,x2_tr,y_tr = shuffle(x1_tr,x2_tr,y_tr)

    all_data = (x1_tr.astype(np.float32),
                x2_tr.astype(np.float32),
                y_tr,
                x1_te.astype(np.float32),
                x2_te.astype(np.float32),
                y_te,
                x1_val.astype(np.float32),
                x2_val.astype(np.float32),
                y_val
                )

    return all_data


if __name__ == "__main__":
    import numpy as np
    from collections import Counter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = load_mnist_3m()
    '''
    x1_tr,x2_tr,x3_tr,y_tr,x1_te,x2_te,x3_te,y_te = data
    
    seed = 28211
    print('label',y_tr[seed])
    plt.imshow(x3_tr[seed,:,:,:])
    plt.savefig('svhn.pdf')
    plt.imshow(np.reshape(x2_tr[seed,:],(28,28)))
    plt.savefig('mnist_rot.pdf')
    plt.imshow(np.reshape(x1_tr[seed,:],(28,28)))
    plt.savefig('mnist_orig.pdf')
    '''
    #print 'x1_tr x2_tr x3_tr ', x1_tr.shape, x2_tr.shape, x3_tr.shape
    #print 'x1_te x2_te x3_te ', x1_te.shape, x2_te.shape, x3_te.shape
