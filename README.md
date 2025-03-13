# AssayCTX
This repository contains the code to reproduce and build on the findings from the manuscript "Towards assay-aware bioactivity model(er)s: getting a grip on biological context" (https://chemrxiv.org/engage/chemrxiv/article-details/67cb11e0fa469535b96191ae).

## Description
The main finding from the assay-aware research is that there is a large variety in the measured values for the same protein-compound pair in ChEMBL. The assay categorization developed for this work explains part of this variety and is therefore usefull to take into account. Here we will first describe an easy way to get insight into the assays you are modelling. Then we will show how to calculate assay descriptors. Lastly we included the steps to take for reproducing the results described in the manuscript.

## Obtain overview of assay context
To use the pretrained BERTopic you need to install the following packages
    conda create --name llm_agent python=3.11
    pip install bertopic
    pip install papyrus_scripts
    pip install langchain-core              #optional
    pip install langchain-experimental      #optional
    pip install langchain-community         #optional
    pip install llama-cpp-python            #optional
    pip install jupyter
    pip install pystow
    pip install pandas

For an example of how to assign your datapoints to assay topics have a look at the following notebook: assayctx/descriptors/llm_agent.ipynb

## Reproducing results
### Dataset creation
Data for validation can be extracted from ChEMBL and papyrus. 

To reproduce create an environment with python=3.11:
    pip install git+https://github.com/CDDLeiden/QSPRpred.git@v2.1.0
    pip install chembl_downloader
    pip install git+https://github.com/LindeSchoenmaker/BindingType.git

And run
    python assayctx/dataset_creation/description_loader.py
    python assayctx/dataset_creation/datasets.py

### Assay embedding and clustering
To create assay embeddings and do the assay grouping you need an environment with python=3.11 and the following packages:
    pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cuml-cu12==24.10.*
    pip install bertopic
    pip install -U sentence-transformers
    pip install polars
    pip install textblob
    python -m textblob.download_corpora
    pip install matplotlib
    pip install pystow
    pip install seaborn

The following functions are related
python assayctx/descriptors/embeddings.py           #for creating BioBERT embeddings
python assayctx/analysis/embedding_UMAP.py          #for plotting assay embeddings using UMAP
python assayctx/descriptors/topics.py               #for creating topic clusters
python assayctx/descriptors/topic_information.py    #for retrieving topic information (number of assay groups, assay describing words, assigning a description to a topic)


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