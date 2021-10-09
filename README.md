# Discriminative Multimodal Learning via Conditional Priors in Generative Models 
Code for the framework in **Discriminative Multimodal Learning via Conditional Priors in Generative Models** ([paper](https://arxiv.org/abs/1904.11376)).

If you use this code in your research, please cite:

	@article{mancisidor2020deep,
  		title={Deep generative models for reject inference in credit scoring},
		author={Mancisidor, Rogelio A and Kampffmeyer, Michael and Aas, Kjersti and Jenssen, Robert},
		journal={Knowledge-Based Systems},
		volume={196},
		pages={105758},
		year={2020},
		publisher={Elsevier}
	}

## Requirements
The code for CMMD in the paper is developed in Theano. We suggest to run the code using `Docker`. Run the command `docker pull rogelioandrade/theano_setup:v4` to pull an image with all dependencies into your local machine or to run the code in a cluster.

Run `mkdir output data`, so the structure of the project look like this:

```
cmmd 
   │───data
   │───output
   │───python
```

Otherwise you will get error messages when loading the data, saving figures etc.

**Note**: you can run `bash build_image` to build the above image and mount all folders inside `cmmd`.

## Downloads
### MNIST 2-modalities
You can download the data set from here [data](https://biedu-my.sharepoint.com/:u:/g/personal/a1910329_nbsemp_no/EYkTm1w7pbVKieABiOHKHiIB5h8GmQGLZL5c_amRkWJGSw?e=jwsxGc), or you can get the matlab code to generate the two different modalities from here [code](https://www.google.com/url?q=https%3A%2F%2Fttic.uchicago.edu%2F~wwang5%2Fpapers%2Fdcca.tgz&sa=D&sntz=1&usg=AFQjCNF6TF3krK7GDKPX4o9bk3QbUaf5ZQ). The file is called `createMNIST.m`.

### XRMB 
You can obtain the two files for the XRMB data set here [data1](https://biedu-my.sharepoint.com/:u:/g/personal/a1910329_nbsemp_no/ET8dhlUmveRMgSkoi5cCAREBirLGU7PTPK_AX2f_r6Mp8w?e=vgc2jW) and [data2](https://biedu-my.sharepoint.com/:u:/g/personal/a1910329_nbsemp_no/EfBPGI6Ch0dGmuHBNBGcMIMBFJ2rmYI26okojFNQV9CaIA?e=Z8Ill8) or you can visit the website [link](https://home.ttic.edu/~klivescu/XRMB_data/full/README) 

Make sure to save the data files inside `data` 

### Flickr
You can obtaniened the data set here [data](http://www.cs.toronto.edu/~nitish/multimodal/index.html)

### MNIST-SHVN
See [link](https://github.com/iffsid/mmvae) for details about how to create the data set.


### Pretrained models 
Reproduce the results in table 2 and 6.

**MNIST**: Dowloaded the pretranied model from here [pretrained_mnist](https://biedu-my.sharepoint.com/:f:/g/personal/a1910329_nbsemp_no/EhqsIO9C2_hFrw2Hk2mD_aQBPFlahCG31bYAC6cWGQDqYw?e=rwnOVM), unzip the file and save it under `output`. Then run `test_cmmd.py` and choose `mnist` and `idx=0` as arguments for `test_cmmd`. 

**XRMB**: Pretrained weights for XRMB can be obtained here [pretrained_xrmb](https://biedu-my.sharepoint.com/:f:/g/personal/a1910329_nbsemp_no/ElUBycfhLjhOlcdC4FnlbQEByUd1PAoQ6dVxjljSxKiqpQ?e=YDZ6b1). In this case, choose `xrbm` and `idx=0` as arguments for `test_cmmd`.


**Note**: Weights trained on a GPU(CPU) can only be loaded again on a GPU(CPU). The pretrained weights in the above links  were trained on a GPU.

## Usage
### Training

Make sure the [requirements](#requirements) are satisfied in your environment, and relevant [datasets](#downloads) are downloaded. `cd` into `python`, and run

```bash
THEANO_FLAGS=device=cuda0,floatX=float32 python -u ./train_cmmd.py --omega 0.4 --hdim_enc 2500 2500 2500 --hdim_dec 1024 1024 1024 --hdim_prior 1024 1024 1024 --zdim 50 --hdim_cls 70 70 --epochs 1001 --R 1 --outfile mnist --dset mnist
```

To run the code on cpu replace `device=cuda0` for `device=cpu`.

You can play with the hyperparameters using arguments, e.g.:
- **`--omega`**: omega parameter controling the mutual information optimization. 
- **`--dropout_rate`** dropout probability
- **`--zdim`**: dimension of latent variable

For all arguments see all `add_argument()` functions in `train_cmmd.py`

