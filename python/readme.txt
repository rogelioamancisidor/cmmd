Full code of the proposed model will be available at github together 
with some trained models to reproduced some of the results in the 
paper. In addition, we will provide a container in hub.docker to run 
the code without problems. 

The easiest way to train CMMD is by running `run_cmmd.sh`. In there,
the model parameters can be specified. 

The file `CMMD.py` contains the computational graph of the model, 
`train_cmmd.py` loads data sets, specifies model parameters and train
the model. Finally, `test_cmmd.py` can be used to load a trained
model and test its performance.

The rest of the files contain axiliary functions and objects.
