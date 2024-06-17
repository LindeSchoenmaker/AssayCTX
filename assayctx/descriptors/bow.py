import pickle

import pandas as pd
import pystow
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

DATA_DIR = pystow.join("AssayCTX", "data")

# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


if __name__ == "__main__":
    assay_stop_words = set(['treated', 'kg', 'reduction', 'model', 'production', 'day', 'residues', 'substrate', 'method', 'on', 'level', 'challenge', 'induction', 'type', 'decrease', 'infection', 'for', 'tested', 'with', 'increase', 'presence', 'dilution', 'in', 'evaluated', 'from', 'as', 'assessed', 'dosed', 'ug', 'prior', 'before', 'ml', 'nm', 'inhibitory', 'determined', 'mm', 'control', 'days', 'dose', 'residual', 'based', 'expressed', 'expressing', 'activation', 'origin', 'using', 'length', 'expression', 'concentration', 'compound', 'of', 'incubated', 'up', 'percent', 'formation', 'measuring', 'index', 'post', 'the', 'analysis', 'mins', 'mg', 'hr', 'effect', 'phase', 'test', 'time', 'mic', 'activity', 'min', 'harboring', 'at', 'stimulated', 'administration', 'addition', 'and', 'administered', 'system', 'followed', 'to', 'assay','measured', 'mediated', 'by', 'against', 'after', 'um', 'induced', 'hrs', 'stimulation', 'was', 'change'])
    assay_stop_words.update(list(ENGLISH_STOP_WORDS))

    stemmed_stop_words = set()
    for word in assay_stop_words:
        stemmed = textblob_tokenizer(word)[0]
        stemmed_stop_words.add(stemmed)

    df = pd.read_csv(DATA_DIR / 'assay_desc_mapping_fb.csv')
    corpus = df.description.unique()
    count_vectorizer = CountVectorizer(token_pattern = r'(?!\d)(?u)\b\w\w+\b', max_features=1024, stop_words = list(stemmed_stop_words), tokenizer=textblob_tokenizer)
    count_vectorizer.fit(corpus)

    with open(DATA_DIR /'chembl_vectorizer.pk', 'wb') as fin:
        pickle.dump(count_vectorizer, fin)

        