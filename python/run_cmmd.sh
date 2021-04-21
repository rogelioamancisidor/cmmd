# experiments icml paper
THEANO_FLAGS=device=cuda0,floatX=float32 python -u ./train_cmmd.py --omega 0.6 --hdim_enc 3000 3000 3000 --hdim_dec 1500 1500 1500 --hdim_dec2 1500 1500 1500 --zdim 70 --hdim_cls 100 100 --epochs 501 --R 1 --outfile cmmd_06 --dset xrmb_folds --fold 0 2>&1 | tee -a ../output/log.txt
