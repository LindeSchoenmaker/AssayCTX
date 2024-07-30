import pandas as pd
import pystow
from bertopic import BERTopic

BERT_DIR = pystow.join("AssayCTX", "bert")
DATA_DIR = pystow.join("AssayCTX", "data")




def retrieve_topic(chembl_ids):
    df_topic_full = pd.read_parquet(DATA_DIR / "descriptions_biobert_nomax_None_128.parquet")[['chembl_id', 'description', 'olr_cluster_None']]
    df_topic = df_topic_full[df_topic_full['chembl_id'].isin(chembl_ids)]
    print(df_topic)
    return df_topic

def assign_topic(descriptions):
    topic_model = BERTopic.load(BERT_DIR / "saved_topicmodel_biobert_nomax_None_None_128_dir")
    topics, proba = topic_model.transform(descriptions)
    if -1 in topics:
        topics = topic_model.reduce_outliers(descriptions, topics)

    df = pd.DataFrame({'description': descriptions, 'olr_cluster_None': topics, 'probability': proba})
    df.to_csv(DATA_DIR / 'topics.csv')
    return df


def topic_information(df_topic):
    topic_counts = df_topic['olr_cluster_None'].value_counts().to_frame().rename({'olr_cluster_None': 'Count in dataset'}, axis=1)
    topic_model = BERTopic.load(BERT_DIR / "saved_topicmodel_biobert_nomax_None_None_128_refit_dir")
    info = topic_model.get_topic_info()[['Topic', 'Count', 'Representation']]
    topic_info = topic_counts.merge(info, left_index=True, right_on='Topic').reset_index(drop=True)
    print(topic_info)
    topic_info.to_csv(DATA_DIR / 'topic_information.csv')

def get_topic_info(chembl_ids):
    df_topic = retrieve_topic(chembl_ids = chembl_ids)
    
    topic_information(df_topic)

if __name__ == "__main__":
    chembl_ids = ['CHEMBL1115158', 'CHEMBL1003978']
    df_topic = retrieve_topic(chembl_ids)
    topic_information(df_topic)

    descriptions = ['Inhibition of firefly luciferase activity at 10 uM']
    topics = assign_topic(descriptions)
    topic_information(topics)