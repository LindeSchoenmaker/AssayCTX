import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystow
from bertopic import BERTopic
from sklearn import metrics

BERT_DIR = pystow.join("AssayCTX", "bert")
DATA_DIR = pystow.join("AssayCTX", "data")
FIG_DIR = pystow.join("AssayCTX", "figures")

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_size(target= None):
    df_info = pd.DataFrame()
    for min_cluster_size in [16, 32, 64, 128]:
        df = pd.read_parquet(DATA_DIR / f"descriptions_biobert_{target}_{min_cluster_size}.parquet")
        dataframes = [df]
        if target:
            df_control = pd.read_parquet(DATA_DIR / f"descriptions_biobert_None_{min_cluster_size}.parquet")
            df_control = df_control[df_control['chembl_id'].isin(df.chembl_id.to_list())]
            dataframes.append(df_control)
        
        for i, topics in enumerate(dataframes):
        
            for supervised in [None, "assay_type", "standard_type"]:
                counts = topics[f'olr_cluster_{supervised}'].value_counts()
                df_info = pd.concat([df_info, pd.DataFrame({'supervised': [supervised],
                                                            'min_cluster': [min_cluster_size], 
                                                            'outlier_reduction': True,
                                                            'total':[len(counts)], 
                                                            'mean': [np.mean(counts.values)], 
                                                            'median': [np.median(counts.values)],
                                                            'max': [counts.max()],
                                                            'control':[i],
                                                            })])
                topics_filtered = topics.loc[topics[f'cluster_{supervised}'] != -1]
                counts = topics_filtered[f'cluster_{supervised}'].value_counts()
                df_info = pd.concat([df_info, pd.DataFrame({'supervised': [supervised],
                                                            'min_cluster': [min_cluster_size], 
                                                            'outlier_reduction': False,
                                                            'total':[len(counts)], 
                                                            'mean': [np.mean(counts.values)], 
                                                            'median': [np.median(counts.values)],
                                                            'max': [counts.max()],
                                                            'outliers': [len(topics)-len(topics_filtered)],
                                                            'control':[i],
                                                            })])
    df_info = df_info.pivot(index=['supervised', 'min_cluster'], columns=['outlier_reduction', 'control'])
    df_info.sort_values(by=['supervised'], inplace=True)
    df_info.to_csv(DATA_DIR / f'cluster_size_topics_{target}.csv')


def purity(target):
    df_info = pd.DataFrame()
    repeat = None
    repeats = ['0', '0_1', '0_1_2']
    for min_cluster_size in [16, 32, 64, 128]:
        for repeat in repeats:
            df = pd.read_parquet(DATA_DIR / f"descriptions_biobert_{repeat}_{target}_{min_cluster_size}.parquet")
            dataframes = [df]
            if target:
                df_control = pd.read_parquet(DATA_DIR / f"descriptions_biobert_{repeat}_None_{min_cluster_size}.parquet")
                df_control = df_control[df_control['chembl_id'].isin(df.chembl_id.to_list())]
                dataframes.append(df_control)
            
            for i, topics in enumerate(dataframes):
                for supervised in [None]: #, "assay_type", "bao_format", "standard_type"]:
                    for info in ["assay_type", "bao_format", "standard_type"]:
                        xmin = df[info].value_counts()[0]/len(df)
                        purity = purity_score(topics[info], topics[f'olr_cluster_{supervised}'])
                        normalized_purity = (purity-xmin)/(1-xmin)
                        df_info = pd.concat([df_info, pd.DataFrame({'supervised': [supervised],
                                                                    'min_cluster': [min_cluster_size], 
                                                                    'repeat': [repeat],
                                                                    'outlier_reduction': [True],
                                                                    'metric': [info],
                                                                    'info':[normalized_purity], 
                                                                    'control':[i], 
                                                                    })])
        
                        purity = purity_score(topics[info], topics[f'cluster_{supervised}'])
                        normalized_purity = (purity-xmin)/(1-xmin)
                        df_info = pd.concat([df_info, pd.DataFrame({'supervised': [supervised],
                                                                    'min_cluster': [min_cluster_size], 
                                                                    'repeat': [repeat],
                                                                    'outlier_reduction': [False],
                                                                    'metric': [info],
                                                                    'info':[normalized_purity], 
                                                                    'control':[i], 
                                                                    })])
    for info in ["assay_type", "bao_format", "standard_type"]:
        print(f'baseline {info}: {df[info].value_counts()[0]/len(df)}')            
    df_info_p = df_info.pivot(index=['supervised', 'min_cluster', 'repeat'], columns=['metric','outlier_reduction', 'control'])
    df_info_p.sort_values(by=['supervised'], inplace=True)
    df_info_p.to_csv(DATA_DIR / f'cluster_purity_topics_{target}.csv')

    mean = df_info.groupby(['min_cluster', 'metric', 'outlier_reduction']).mean()['info']
    std = df_info.groupby(['min_cluster', 'metric', 'outlier_reduction']).std()['info']

    fig = plt.figure()
    ax = mean.unstack(['min_cluster']).plot(kind='bar', yerr=std.unstack(['min_cluster']), xlabel='', color=["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])
    ax.set_xticks(ticks=range(6), labels=['no olr\nassay type', 'olr\nassay type', 'no olr\nboa format', 'olr\nboa format', 'no olr\nstandard type', 'olr\nstandard type'], size=9, rotation=0)
    ax.legend(title='Minimum size')
    ax.set_ylabel('Purity')
    plt.tight_layout()
    plt.savefig('purity_norm_barplot.png')


def topic_words(target, supervised, min_size):
    topic_model = BERTopic.load(BERT_DIR / f"saved_topicmodel_biobert_{target}_{supervised}_{min_size}_dir")
    dic1 = topic_model.get_topics()
    dic2 = {k:[x[0] for x in v] for k,v in dic1.items()}
    df_topic_words = pd.DataFrame.from_dict(dic2, orient='index')
    df_top5 = df_topic_words.loc[range(5)]
    df_top5.to_csv(f'cluster_words_topics_{target}.csv')

if __name__ == "__main__":
    cluster_size(target=None)
    purity(target=None)