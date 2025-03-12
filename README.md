# AssayCTX
This repository contains the code to reproduce and build on the findings from the assay context manuscript. 

## Description
Contains pipeline for building datasets, using the QSPRpred models tested for this project and further analysis.

## Environment
Scripts are compatible with QSPRPred tagged v2.1.0a0. 

### Dataset creation
Data for validation can be extracted from ChEMBL and papyrus. 

To reproduce create an environment with python=3.11:
    pip install git+https://github.com/CDDLeiden/QSPRpred.git@v2.1.0
    pip install chembl_downloader
    pip install git+https://github.com/LindeSchoenmaker/BindingType.git

And run
    python assayctx/dataset_creation/description_loader.py
    python assayctx/dataset_creation/datasets.py


### Complete environment
The following describes how to create an environment with all the packages used in this repository. Take note that some adjustments might be necessary based on your CUDA version.
Create the environment using the following steps:
    conda create --name assay_311_new python=3.11
    pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cuml-cu12==24.10.*
    pip install bertopic
    pip install -U sentence-transformers
    pip install polars
    pip install Py-Boost
    pip install git+https://github.com/CDDLeiden/QSPRpred.git@v2.1.0
    pip install textblob
    pip install chembl_downloader
    pip install git+https://github.com/LindeSchoenmaker/BindingType.git
    pip install Signature-pywrapper CDK-pywrapper PaDEL-pywrapper Mold2-pywrapper

    python -m textblob.download_corpora

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