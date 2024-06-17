import os

import polars as pl
import pystow
from simpletransformers.language_representation import RepresentationModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_DIR = pystow.join("AssayCTX", "data")

if __name__ == "__main__":
    df = pl.read_csv(DATA_DIR / "assay_desc_mapping_fb.csv")
    descriptions = (
        pl.scan_csv(DATA_DIR / "assay_desc_mapping_fb.csv")
        .filter(pl.col("desc_length") <= 500) # remove rows with length > 500
        .drop("chembl_id")
        .drop("assay_type")
        .drop("desc_length")
        .drop_nulls()
        .unique()
        .collect()
        .to_series()
        .to_list()
    )

    model = RepresentationModel(
        model_type="bert", model_name="dmis-lab/biobert-base-cased-v1.2", use_cuda=True
    )

    word_vectors = model.encode_sentences(descriptions, combine_strategy="mean")

    # create a polars dataframe with the word vectors as a column
    df_emb = pl.DataFrame({"description": descriptions, "word_vectors": word_vectors})
    joined = df.join(df_emb, on='description', how="inner")
    joined.write_parquet(DATA_DIR / "sentence_vectors.parquet")
