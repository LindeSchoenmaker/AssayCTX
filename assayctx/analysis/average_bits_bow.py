import pickle

import pandas as pd
import pystow
from textblob import TextBlob

DATA_DIR = pystow.join("AssayCTX", "data")

# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

if __name__ == "__main__":
    with open(DATA_DIR / 'chembl_vectorizer.pk', 'rb') as fn:
        vectorizer = pickle.load(fn)

    df = pd.read_csv(DATA_DIR / "assay_desc_mapping_fb.csv", usecols=['description'])
    corpus = df.description
    vectors = vectorizer.transform(corpus)
    print(f'average number of bits per description{vectors.nnz/vectors.shape[0]}')