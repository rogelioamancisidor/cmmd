# mnist 
#THEANO_FLAGS=device=cuda0,floatX=float32 python -u ./train_cmmd.py --omega 0.4 --hdim_enc 2500 2500 2500 --hdim_dec 1024 1024 1024 --hdim_prior 1024 1024 1024 --zdim 50 --hdim_cls 70 70 --epochs 1001 --R 5 --outfile test --dset mnist 2>&1 | tee -a ../output/log.txt
THEANO_FLAGS=device=cuda0,floatX=float32 python -u ./test_cmmd.py 
