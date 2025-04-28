import os

import pandas as pd
import pystow
from sklearn import metrics

os.environ['PYSTOW_HOME'] = "/zfsdata/data/linde"
DATA_DIR = pystow.join("AssayCTX", "data")

def calc_scores(df, i, assay_type, column = 'meta_target', supervised='olr_cluster_None'):
    df_topic = pd.read_parquet(DATA_DIR / f"descriptions_biobert_None_{i}.parquet")
    print(len(df_topic))
    df_topic = df_topic[df_topic['chembl_id'].isin(df.chembl_id.to_list())]

    df_topic = df_topic[['chembl_id', 'olr_cluster_None', 'olr_cluster_assay_type', 'olr_cluster_standard_type']]

    if assay_type:
        df = df.loc[df['assay_type'] == assay_type]
    df = pd.merge(df, df_topic, on='chembl_id')

    print(len(df.olr_cluster_None.unique()))
    ami = metrics.adjusted_mutual_info_score(df[column], df[supervised])
    homo = metrics.homogeneity_score(df[column], df[supervised])
    compl = metrics.completeness_score(df[column], df[supervised])
    v = metrics.v_measure_score(df[column], df[supervised])
    fm = metrics.fowlkes_mallows_score(df[column], df[supervised])

    return ami, homo, compl, v, fm

if __name__ == "__main__":   
    df = pd.read_csv(DATA_DIR / 'AR_categorized.csv')

    df_topic = pd.read_parquet(DATA_DIR / f"descriptions_biobert_None_{128}.parquet") # 64 best for gpcrs
    df_topic = df_topic[df_topic['chembl_id'].isin(df.chembl_id.to_list())]

    df_topic = df_topic[['chembl_id', 'cluster_None', 'olr_cluster_None']] #, 'olr_cluster_assay_type', 'olr_cluster_bao_format', 'olr_cluster_standard_type']]

    df = pd.merge(df, df_topic, on='chembl_id')

    df_B = df.loc[df['assay_type'] == 'B']
    df_F = df.loc[df['assay_type'] == 'F']
    print(metrics.completeness_score(df_B['standard_type'], df_B['olr_cluster_None']))
    print(metrics.completeness_score(df_F['meta_target'], df_F['olr_cluster_None']))

    df_org = pd.read_csv(DATA_DIR / 'AR_categorized.csv')

    df_scores = pd.DataFrame(columns=["AMI", 'homogeneity', 'completeness', 'V-measure', 'Fowlkes-Mallows'])
    for assay_type, key in {'F': 'meta_target', 'B': 'standard_type'}.items():
        print(assay_type)
        for i in [16, 32, 64, 128]:
            for supervised in ['olr_cluster_None', 'olr_cluster_assay_type', 'olr_cluster_standard_type']:
                print(i)
                ami, homo, compl, v, fm = calc_scores(df_org, i, assay_type, key, supervised=supervised)
                df_scores.loc[assay_type + " " + str(i)+ " " + supervised] = [ami, homo, compl, v, fm]
    df_scores.to_csv('data/AR_clustering_scores.csv')

    # for assay_type in ['F', 'B']:
    #     print(assay_type)
    #     i = 128
    #     for supervised in ['olr_cluster_None', 'olr_cluster_assay_type', 'olr_cluster_standard_type']:
    #         print(supervised)
    #         calc_scores(df_org, i, assay_type, key, supervised='olr_cluster_None')