# AssayCTX
This repository contains the code to reproduce and build on the findings from the assay context manuscript. 

## Description
Contains pipeline for building datasets, using the QSPRpred models tested for this project and further analysis.

## Environment
Scripts are compatible with QSPRPred tagged v2.1.0a0. 
Create the environment using the following steps:
    conda create -n assay_311 -c rapidsai -c conda-forge -c nvidia cuml=24.08 python=3.11 'cuda-version>=12.0,<=12.5' 'pytorch=*=*cuda*'
    pip install cupy-cuda12x
    pip install Py-Boost
    pip install -U sentence-transformers
    pip install git+https://github.com/CDDLeiden/QSPRpred.git@v2.1.0
    pip install textblob
    pip install chembl_downloader
    pip install git+https://github.com/LindeSchoenmaker/BindingType.git
    pip install polars
    pip install Signature-pywrapper CDK-pywrapper PaDEL-pywrapper Mold2-pywrapper
    pip install bertopic

## Dataset creation
Datasets are created with unique protein-compound-assay as rows, only of type B & F

## Assay descriptors
Assay descriptors and assay descsription based clusters are created using scripts in utilities

## Run models
Pipeline for training and evaluating different types of models. Options;
- create default model, model with custom assay descriptors and multitask models
- random or scaffold split (possible to run repeats with different seeds)
- different target benchmark sets

Dataset class is used to preprocess y values, get target properties, the FASTA files of the sequences (used to create MSA), calculate the descriptors and split the data.
Afterwards models can be optimized using hyperparameter optimization or default parameters can be supplied. Model performance is evaluated using cross-validation and on a separate test set.

## Performance metrics
Jupyter notebook for getting the R2 and RMSE for all the saved models.