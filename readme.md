# Hierarchical Inducing Point Gaussian Process for Inter-domainObservations

This repo contains the codes for the AISTATS 2021 paper [Hierarchical Inducing Point Gaussian Process for Inter-domainObservations](https://arxiv.org/pdf/2103.00393.pdf).



__Abstract__

_We examine the general problem of inter-domain Gaussian Processes (GPs): problems where the GP realization and the noisy ob-servations of that realization lie on different domains. When the mapping between those domains is linear, such as integration or differentiation, inference is still closed form.However, many of the scaling and approximation techniques that our community has developed do not apply to this setting. In this work, we introduce the hierarchical inducing point GP(HIP-GP), a scalable inter-domain GP inference method that enables us to improve the approximation accuracy by increasing the number of inducing points tothe millions. HIP-GP, which relies on inducing points with grid structure and a stationary kernel assumption, is suitable for low-dimensional problems. In developing HIP-GP, we introduce (1) a fast whitening strategy, and (2) a novel preconditioner for conjugate gradients which can be helpful in generalGP settings._



## Requirements



Create a new conda enviroment with python=3.7 as follows

`conda create -n hipgp python=3.7` 

To activate the conda environment, run: 

`conda activate hipgp` 

Ensure that you are in the root directory:

`cd hipgp`

To install the package requirements run the following

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Finally, install `ziggy` and `ziggy.misc`-- the packages for hipgp implementation, by running 

`python setup.py install`



## Datasets and Experiments

First, 

`cd experiments-hip-gp`

__1. Effect of the Preconditioner (Sec 5.1 )__

`python run_solve_kn_experiment.py  `

__2. Speedup over Cholesky Decomposition (Sec 5.2)__

`python run_pcg_vs_cholesky.py`

__3.Synthetic Derivative Observations (Sec 5.3)__

Run the jupyter notebook `GP-with-Derivatives.ipynb`

In order to run the notebook, you would probably need to create a kernel by

```bash
pip install ipykernel
python -m ipykernel install --user --name hipgp
```

__4. Spatial Analysis: UK Housing Prices (Sec 5.4)__

Download the 2018 uk housing data from the following sources:

  - UK land registry monthly price data:
    https://ckan.publishing.service.gov.uk/dataset/land-registry-monthly-price-paid-data

    This file includes POSTCODE, which are pretty granular in the UK.
    
  - UK Post Code lat/long data: https://www.freemaptools.com/download-uk-postcode-lat-lng.htm

  - UK shapefiles: https://gadm.org/download_country_v3.html

Put these data files under `hipgp/experiments-hipgp/uk-price-paid-data`

An example command to run the experiment

`python run_ukhousing_experiment.py --fit-models --mf-model --nobs=1000 --ntest=100 --num-inducing-x=10 --num-inducing-y=10` 

__5. Inferring Interstellar Dust Map (Sec 5.5)__ 

This repo contains a small sample (5k data points) of the whole dataset, 

under `hipgp/experiemnts-hipgp/domain-data`. 

An example command to run the experiment

`python run_domain_experiment.py --nobs=1000 --ntest=1000 --nx=10 --nz=5 --mf-model --fit-models --lr=1e-6 --epochs=2` 

__6. Empirical analysis on preconditioner (Appendix C.1)__

Run `preconditioner-analysis.ipynb` 

__7. UCI 3d road dataset (Appendix C.3)__

The full set of UCI regression datasets can be downloaded at https://d2hg8soec8ck9v.cloudfront.net/datasets/uci_data.tar.gz. Download the tar file and then put the `3droad.mat` file under `hipgp/experiments-hipgp/uci-data/`. 

An example command to run the experiment

`python run_3droad_experiment.py --nobs=1000 --ntest=1000 --nx=10 --nz=5 --mf-model --fit-models --lr=1e-6 --epochs=2` 

