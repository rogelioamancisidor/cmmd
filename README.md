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

### XRBM 
You can obtain the two files for the XRMB data set here [data1](https://biedu-my.sharepoint.com/:u:/g/personal/a1910329_nbsemp_no/EYkTm1w7pbVKieABiOHKHiIB5h8GmQGLZL5c_amRkWJGSw?e=a1N8KV) and [data2](https://biedu-my.sharepoint.com/:u:/g/personal/a1910329_nbsemp_no/EfBPGI6Ch0dGmuHBNBGcMIMBFJ2rmYI26okojFNQV9CaIA?e=Z8Ill8) or you can visit the website [link](https://home.ttic.edu/~klivescu/XRMB_data/full/README) 

Make sure to save the data files inside `data` 

### Pretrained models 
Pretrained models are available in the `output` folder. To load a pretrained model use the script `test_cmmd.py`. 

**Note**: Weights trained on a GPU(CPU) can only be loaded again on a GPU(CPU). The pretrained weights in the `output` folder were trained on a GPU.

## Usage
### Training

Make sure the [requirements](#requirements) are satisfied in your environment, and relevant [datasets](#downloads) are downloaded. `cd` into `python`, and run

```bash
THEANO_FLAGS=device=cuda0,floatX=float32 python -u training_model1.py --outfile model1 --epochs 401 --n_cv 1 --beta 1.1 --dset paper --n_sup 1552 --n_unsup 30997 
```

To run the code on cpu replace `device=cuda0` for `device=cpu`.

You can play with the hyperparameters using arguments, e.g.:
- **`--n_sup`**: number of observations from the minority class, i.e. y=1
- **`--n_unsup`** number of unsupervised  observations
- **`--zdim`**: dimension of latent variable

For all arguments see all `add_argument()` functions in `training_model1.py` and `training_model2.py`

#### Stability
<p align='center'><img src="output/auc_m1.png" width="50%" height="50%"></p>
As mentioned in the paper, training M1 and M2 can be unstable. For example, the above diagram shows the test AUC during model training. In two of the runs, model training became unstable and it is reflected by a sharp and sundden drop in AUC.

### Analyzing
<p align='center'><img src="output/gmm_latent_space_m2.png"></p>

Run `model.plot_gmm_space()` to draw `z` representations and visualize them as 2D t-sne vectors. The above figure shows a mixture of two Gaussians distributions in the latent space. The representations for customers with class label y=1 lie in the upper-right cuadrant. The histograms on the side show the estimated default probability for each class label. Customers with class label y=1 have on average higher default probabilities. The scatter color is given by its estimated default probability.      

**Note**: It takes some epochs (~500 epochs) before the latent space shows a mixture model.
