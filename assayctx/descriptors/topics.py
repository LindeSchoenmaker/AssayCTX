import os

# # Choose gpus to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import polars as pl
import pystow
from bertopic import BERTopic

# Need Rapids(https://rapids.ai/) to run this
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sentence_transformers import SentenceTransformer

# Reduce priority of process
os.nice(15)

BERT_DIR = pystow.join("AssayCTX", "bert")
DATA_DIR = pystow.join("AssayCTX", "data")

print(BERT_DIR)


def runner(transformer, output, target=None):
    """
    Runs the BERTopic model on the assay descriptions and saves the model and the topics to disk.
    transformer: the transformer to use, e.g. 'allenai/scibert_scivocab_uncased'
    output: the name of the output file, e.g. 'scibert'

    See [this](https://huggingface.co/dmis-lab/biobert-large-cased-v1.1-squad) to see how to choose own
    HuggingFace models.
    """
    print(f"Generating '{transformer}'")
    if target:
        targets = pd.read_csv(DATA_DIR / f'filtered_assays_split_{target}.csv', sep=',', usecols = ['AID']).AID.unique().tolist()
        df_descriptions = (
            pl.scan_csv(DATA_DIR / "assay_desc_mapping_fb_info.csv")
            .filter(pl.col("desc_length") <= 500) # remove rows with length > 500
            .filter(pl.col("chembl_id").is_in(targets))
            .select(["chembl_id", "description", "assay_type", "bao_format", "standard_type"])
            .drop_nulls("description")
            .unique("description")
        )
    else:
        df_descriptions = (
            pl.scan_csv(DATA_DIR / "assay_desc_mapping_fb_info.csv")
            .filter(pl.col("desc_length") <= 500) # remove rows with length > 500
            .select(["chembl_id", "description", "assay_type", "bao_format", "standard_type"])
            .drop_nulls("description")
            .unique("description")
        )

    df = df_descriptions.collect().to_pandas() # turn into pandas df
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # reorder

    # descriptions takes in a list of strings, preprocess as needed
    descriptions = df["description"].to_list()

    # Pre-calculate embeddings
    embedding_model = SentenceTransformer(transformer)
    embeddings = embedding_model.encode(descriptions, show_progress_bar=False)


    for min_cluster_size in [8, 16, 32, 64, 128]:
        umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.1, random_state=42)
        hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, min_cluster_size=min_cluster_size, max_cluster_size=1000)

        topic_model = BERTopic(
            name = f'BERTopic_mcs{min_cluster_size}',
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
        )
        for supervised in [None, "assay_type", "bao_format", "standard_type"]:
            print("fit_transform normal model")
            if supervised:
                y=df[supervised]
            else:
                y=None

            topics, _ = topic_model.fit_transform(documents=descriptions, embeddings=embeddings, y=y)
            topic_model.save(BERT_DIR / f"saved_topicmodel_{output}_{target}_{supervised}_{min_cluster_size}_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=transformer)

            # This outlier reduction step is optional,
            # but I found it to be useful for my own research
            print("Reducing outliers...")
            new_topics = topic_model.reduce_outliers(descriptions, topics)
            topic_model.update_topics(descriptions, new_topics)
            topic_model.save(BERT_DIR / f"saved_topicmodel_{output}_{target}_{supervised}_{min_cluster_size}_refit_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=transformer)
            
            df[f'cluster_{supervised}'] = topics
            df[f'olr_cluster_{supervised}'] = new_topics

        df.to_parquet(BERT_DIR / f"descriptions_{output}_{target}_{min_cluster_size}.parquet", compression="zstd")
        

if __name__ == "__main__":
    # Comment out the ones you don't want to run
    runner("dmis-lab/biobert-base-cased-v1.2", "biobert", target="slcs")
    runner("dmis-lab/biobert-base-cased-v1.2", "biobert", target="gpcrs")
    runner("dmis-lab/biobert-base-cased-v1.2", "biobert", target="kinases")
    runner("dmis-lab/biobert-base-cased-v1.2", "biobert", target=None)
