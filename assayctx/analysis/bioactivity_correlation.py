import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystow
from sklearn.metrics import mean_absolute_error

matplotlib.rcParams.update({'font.size': 18})
DATA_DIR = pystow.join("AssayCTX", "data")
FIG_DIR = pystow.join("AssayCTX", "figures")

def aid_descriptor(data, precalculated_categorization):
    """to turn AID into unique int per AID"""
    if precalculated_categorization:
        df = pd.read_csv(DATA_DIR / precalculated_categorization)
        df = df.replace('Displacement', 'affinity')
        df = df.replace('Generic_downstream', 'other')
        data = pd.merge(df, data, left_on='chembl_id', right_on='AID')

        data['taskind'] = np.nan
        aid_list = data.Assay.unique().tolist()
        print(aid_list)

        for i, aid in enumerate(aid_list):
            data['taskind'] = np.where((data.Assay == aid), i, data.taskind)
        assert len(data.taskind.unique().tolist()) == len(aid_list)

    else:
        data['taskind'] = np.nan
        aid_list = data.AID.unique().tolist()

        for i, aid in enumerate(aid_list):
            data['taskind'] = np.where((data.AID == aid), i, data.taskind)
        assert len(data.taskind.unique().tolist()) == len(aid_list)

    return data, aid_list


def preprocessing(df, precalculated_categorization=None):
    """From exploded papyrus data (one row per compound, protein assay combination) get dataframe with assay id integers"""
    
    # turn AIDs into either unique integers or integers based on manual categorization
    df, aid_list = aid_descriptor(df, precalculated_categorization)

    df.taskind = df.taskind.astype('int')

    return df, aid_list


def overlap(df, min_compounds=0):
    """Get array denoting overlap and for which assays enough overlap to for example determine pearsons correlation, make scatter plot"""
    df_togroup = df.drop(columns=['pchembl_value'])

    # array with 0 for all assay combinations
    overlap_arr = np.zeros(
        (len(df.taskind.unique())+1, len(df.taskind.unique())+1))
    print(df.columns)

    # get dataframe with list of all AIDs for each protein-compound combination
    df_aid = df_togroup.groupby(by=['Activity_ID'])[
        'taskind'].apply(list).to_frame()
    for index, row in df_aid.iterrows():
        taskinds_unique = sorted(set(row.taskind))
        for taskind in taskinds_unique:
            # increase value of assay measured in that assay by 1
            overlap_arr[taskind, taskind] = overlap_arr[taskind, taskind] + 1
        # enumerate all combinations of unique assays a compound is measured in and increase their value by one
        for subset in itertools.combinations(taskinds_unique, 2):
            overlap_arr[subset[0], subset[1]
                        ] = overlap_arr[subset[0], subset[1]] + 1
            overlap_arr[subset[1], subset[0]
                        ] = overlap_arr[subset[1], subset[0]] + 1

    # use overlapp_arr as mask if < 10 corr = 0
    # if less than min_compounds number of compounds overlapping then correlation not really reliable

    indices = np.where(overlap_arr > min_compounds)

    # get list of unique assay combinations
    assays = set([tuple(sorted((index[0], index[1])))
                 for index in zip(indices[0], indices[1])])

    return assays, overlap_arr


def scatter_plot_combined(ax, df, assays, labels:list, name:str, use_mean:bool = True, show_errorbar:bool = True):
    df_assay_comb = pd.DataFrame()

    for assay_comb in assays:
        if assay_comb[0] == assay_comb[1]:
            continue
        else:
            assay1 = df[['connectivity', 'pchembl_value']
                        ].loc[df['taskind'] == assay_comb[0]]
            assay2 = df[['connectivity', 'pchembl_value']
                        ].loc[df['taskind'] == assay_comb[1]]

            df_merged = pd.merge(
                assay1, assay2, on='connectivity', how='inner', suffixes=('_x', '_y'))
            df_merged = df_merged.astype({'pchembl_value_x':'float','pchembl_value_y':'float'})
            df_merged = df_merged.groupby(by=['connectivity']).agg({'pchembl_value_x':['mean','std', 'median'], 'pchembl_value_y':['mean','std', 'median']})

            if len(df_merged) > 0:
                df_assay_comb = pd.concat([df_assay_comb, df_merged])

    ax.hexbin(df_assay_comb['pchembl_value_x']['median'],df_assay_comb['pchembl_value_y']['median'],cmap='Blues',bins='log')
    ax.set_xlim(3, 12)
    ax.set_ylim(3, 12)
    ax.set_title(name)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('pChEMBL value')
    ax.set_ylabel('pChEMBL value')


def MAE_scorer(df, assays):

    df_assay_comb = pd.DataFrame()

    for assay_comb in assays:
        if assay_comb[0] == assay_comb[1]:
            continue
        else:
            assay1 = df[['connectivity', 'pchembl_value']
                        ].loc[df['taskind'] == assay_comb[0]]
            assay2 = df[['connectivity', 'pchembl_value']
                        ].loc[df['taskind'] == assay_comb[1]]

            df_merged = pd.merge(
                assay1, assay2, on='connectivity', how='inner', suffixes=('_x', '_y'))
            df_merged = df_merged.astype({'pchembl_value_x':'float','pchembl_value_y':'float'})
            df_merged = df_merged.groupby(by=['connectivity']).agg({'pchembl_value_x':['mean','std', 'median'], 'pchembl_value_y':['mean','std', 'median']})

            if len(df_merged) > 0:
                df_assay_comb = pd.concat([df_assay_comb, df_merged])


    return mean_absolute_error(df_assay_comb['pchembl_value_x']['median'],df_assay_comb['pchembl_value_y']['median']), len(df_assay_comb)

if __name__ == "__main__":
    # start with file with average pchembl value for ActivityId & AID combination
    plot = True
    if plot:
        fig, ax = plt.subplots(1,3, figsize=(18, 6))
        for i, name in enumerate(['GPCRs', 'Kinases', 'SLCs']):
            file = f'filtered_assays_split_{name.lower()}.csv'
            df = pd.read_csv(DATA_DIR / file, usecols=[
                                'Activity_ID', 'accession', 'connectivity', 'AID', 'pchembl_value'])
            df, aid_list = preprocessing(df)
            unique_aids = df.AID.unique()
            
            assays, overlap_arr = overlap(df)
            scatter_plot_combined(ax[i], df, assays, None, name, show_errorbar=False)
        fig.savefig(FIG_DIR / 'scatter_compound_pchembl_comb_median_combined.png')
    else:
        # get MUE scores with and without clustering
        for name in ['gpcrs', 'kinases', 'slcs', 'FBall']:
            file = f'filtered_assays_split_{name}.csv'

            # get array with assays als column en row indices and sum of total measured on both as values
            df = pd.read_csv(DATA_DIR / file, usecols=[
                            'Activity_ID', 'accession', 'connectivity', 'AID', 'pchembl_value'])

            precalculated_categorization =  'descriptions_biobert_nomax_None_128.parquet'
            df_org = pd.read_parquet(DATA_DIR / precalculated_categorization)
            
            df = pd.merge(df_org, df, left_on='chembl_id', right_on='AID')
            scores = []
            nums = []
            topics = []
            for topic in df['olr_cluster_None'].unique().tolist():
                df_topic = df.loc[df['olr_cluster_None'] == topic]
                df_topic, aid_list = preprocessing(df_topic)
                unique_aids = df_topic.AID.unique()
                assays_topic, overlap_arr = overlap(df_topic)
                if len(assays_topic) > len(unique_aids):
                    score, num = MAE_scorer(df_topic, assays_topic)
                    scores.append(score)
                    nums.append(num)
                    topics.append(topic)

            print(np.average(scores, weights=nums))

            # without clustering
            df, aid_list = preprocessing(df)
            unique_aids = df.AID.unique()
            
            assays, overlap_arr = overlap(df)
            score, num = MAE_scorer(df, assays)
            print(f'score for {name}: {score}')