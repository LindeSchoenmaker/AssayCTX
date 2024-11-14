from qsprpred.data.sources.papyrus import Papyrus

print('imported papyrus')
import argparse
import json
import os
import pickle
from copy import deepcopy
from itertools import chain

import chembl_downloader
import cuml
import cupy
import numpy as np
import pandas as pd
import pystow
from prodec import ProteinDescriptors
from qsprpred.data.utils.datafilters import RepeatsFilter
from qsprpred.data.utils.datasplitters import ManualSplit, ScaffoldSplit
from qsprpred.data.utils.descriptorcalculator import (
    CustomDescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from qsprpred.data.utils.descriptorsets import DataFrameDescriptorSet, FingerprintSet
from qsprpred.data.utils.featurefilters import HighCorrelationFilter, LowVarianceFilter
from qsprpred.data.utils.scaffolds import Murcko
from qsprpred.extra.data.data import PCMDataSet
from qsprpred.extra.data.utils.descriptor_utils.msa_calculator import ClustalMSA
from qsprpred.extra.data.utils.descriptorcalculator import ProteinDescriptorCalculator
from qsprpred.extra.data.utils.descriptorsets import ProDec
from qsprpred.extra.gpu.models.pyboost import MSEwithNaNLoss, NaNR2Score, PyBoostModel
from qsprpred.models.assessment_methods import CrossValAssessor, TestSetAssessor
from qsprpred.models.early_stopping import EarlyStoppingMode
from qsprpred.models.hyperparam_optimization import OptunaOptimization
from qsprpred.models.tasks import TargetTasks
from scorer import NaNR2Scorer
from sklearn.model_selection import ShuffleSplit
from textblob import TextBlob

parser = argparse.ArgumentParser()
parser.add_argument("-t",
                    "--target",
                    help="which target to model",
                    type=str,
                    choices=[
                        'gpcrs', 'kinases', 'slcs', 'rtks'
                    ])
parser.add_argument("-c",
                    "--condition",
                    help="which model variants to run",
                    type=str,
                    choices=[
                        'control', 'descriptor', 'MT'
                    ],
                    nargs='+')

parser.add_argument("-s",
                    "--split",
                    help="which split",
                    type=str,
                    choices=[
                        'random', 'scaffold'
                    ],
                    nargs='+')

parser.add_argument("-r",
                    "--repeats",
                    help="which repeat to run",
                    default=None,
                    nargs='+')

DATA_DIR = pystow.join("AssayCTX", "data")
DESC_DIR = pystow.join("AssayCTX", "data")
QSPR_DIR = pystow.join("AssayCTX", "qspr")
BERT_DIR = pystow.join("AssayCTX", "bert")

pdescs = ProteinDescriptors()

# get accepted amino acids
zscales = pdescs.get_descriptor('Zscale Hellberg')
accepted = zscales._scales_values.keys()

def write_fasta(ds_seq, target):
    # use fasta's for clustal alignment online
    ds_seq_unique = ds_seq.drop_duplicates(subset=['accession'])

    seq_unsupported = []

    with open(DATA_DIR / f'fasta_{target}.txt', 'w') as f:
        for index, row in ds_seq_unique.iterrows():
            if any(ele not in accepted for ele in row['Sequence']):
                print(row['accession'])
                seq_unsupported.append(row['accession'])
                continue
            f.write(f'>sp|{row.accession}\n')
            f.write(row.Sequence)
            f.write('\n')


def get_assay_fp(df, target):
    with open(DESC_DIR / "chembl_fp_dict.txt", "r") as fp:
        # Load the dictionary from the file
        fp_dict = json.load(fp)
    columns = [[f'{key}_{i}' for i in fp_dict[key].values()] for key in fp_dict]
    columns = list(chain.from_iterable(columns))
    df_new = pd.DataFrame(columns = columns)
    df['assay_tax_id'] = df['assay_tax_id'].astype('str')
    df['src_id'] = df['src_id'].astype('str')
    df['confidence_score'] = df['confidence_score'].astype('str')
    for index, row in df.iterrows():
        for key in fp_dict:
            try:
                value = fp_dict[key][row[key]]
            except KeyError:
                pass
            df_new.loc[index, f'{key}_{value}'] = 1

    return df_new.fillna(0)


def query(chembl_aids, info='all'):
    # chembl_downloader.download_extract_sqlite()
    if info == 'all':
        sql = f"""
        SELECT
            ASSAYS.chembl_id,
            ASSAYS.assay_type,
            ASSAYS.assay_tax_id,
            ASSAYS.relationship_type,
            ASSAYS.bao_format,
            ASSAYS.confidence_score,
            ASSAYS.curated_by,
            ASSAYS.src_id,
            ACTIVITIES.standard_type
        FROM ASSAYS
        JOIN ACTIVITIES ON ASSAYS.assay_id == ACTIVITIES.assay_id 
        WHERE ASSAYS.chembl_id IN {tuple(chembl_aids)}
        """
    if info == 'description':
        sql = f"""
        SELECT
            ASSAYS.chembl_id,
            ASSAYS.description
        FROM ASSAYS
        WHERE ASSAYS.chembl_id IN {tuple(chembl_aids)}
        """

    df = chembl_downloader.query(sql)
    df = df.drop_duplicates()

    return df


def create_assay_fp(ds, target):
    df = ds.getDF()
    df_assay_fp = df['AID']

    unique_aids = df_assay_fp.unique()
    df_info = query(unique_aids)

    df_info.set_index('chembl_id', inplace=True)

    # calc fingerprints
    df_fp = get_assay_fp(df_info, target)

    # add as columns
    df_assay_fp = pd.merge(df_assay_fp, df_fp, left_on='AID', right_index=True).drop(columns=['AID'])
    df_assay_fp.to_csv(QSPR_DIR / f'data/{target}_assay_fp.csv')



# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def create_assay_bow(ds, target):
    df = ds.getDF()
    df_assay_bow = df['AID']

    unique_aids = df_assay_bow.unique()

    # load vectorizer
    with open(DESC_DIR / 'chembl_vectorizer.pk', 'rb') as fn:
        vectorizer = pickle.load(fn)

    # get list of descriptions for each AID
    df = query(unique_aids, info='description')

    # calc fingerprints
    vectors = vectorizer.transform(df.description)

    for i, col in enumerate(vectorizer.get_feature_names_out()):
        df[col] = pd.Series(
            pd.arrays.SparseArray(vectors[:, i].toarray().ravel(),
                                  fill_value=0))
    df.set_index('chembl_id', inplace=True)
    # add as columns
    df_assay_bow = pd.merge(df_assay_bow, df, left_on='AID',
                            right_index=True).drop(columns=['AID', 'description'])
    df_assay_bow.to_csv(QSPR_DIR / f'data/{target}_assay_bow.csv')


def create_assay_emb(ds, target):
    df = ds.getDF()
    df_assay_emb = df['AID']

    df_emb_org = pd.read_parquet(DESC_DIR / 'sentence_vectors.parquet')

    df_emb = df_emb_org['word_vectors'].apply(pd.Series)
    df_emb['AID'] = df_emb_org['chembl_id']
    df_emb.set_index('AID', inplace=True)

    df_assay_emb = pd.merge(df_assay_emb, df_emb, left_on='AID', right_index=True).drop(columns=['AID'])
    df_assay_emb.to_csv(QSPR_DIR / f'data/{target}_assay_emb.csv')


def create_assay_emb_umap(ds, target):
    df = ds.getDF()
    df_assay_emb = df['AID']

    df_emb_org = pd.read_parquet(DESC_DIR / 'sentence_vectors.parquet')

    # UMAP plot for ChEMBL embeddings
    reducer = cuml.UMAP(n_components=5, n_neighbors=15, min_dist=0.1, random_state=42)

    embeddings = np.array(df_emb_org["word_vectors"].to_list())
    embeddings_umap = reducer.fit_transform(embeddings)

    df_emb = pd.DataFrame(embeddings_umap)
    df_emb['AID'] = df_emb_org['chembl_id']
    df_emb.set_index('AID', inplace=True)

    df_assay_emb = pd.merge(df_assay_emb, df_emb, left_on='AID', right_index=True).drop(columns=['AID'])
    df_assay_emb.to_csv(QSPR_DIR / f'data/{target}_assay_emb_umap.csv')



class BaseDs():

    def __init__(self,
                 target,
                 condition,
                 task,
                 smiles_col,
                 target_col,
                 activity_col,
                 split='random'):
        self.target = target
        self.condition = condition
        self.task = task
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.activity_col = activity_col
        self.split = split

    def create(self, df):
        df, target_props = self.yvalue_prop(df, self.target)

        self.ds_seq = self.get_prot_seq(df)

        ds = PCMDataSet(
            name=
            f'{self.target}_{self.condition}{f"_{self.task}" if self.task else ""}',
            df=df,
            smiles_col=self.smiles_col,
            target_props=target_props,
            store_dir=str(QSPR_DIR / "data"),
            protein_col=self.target_col,
            protein_seq_provider=self.sequence_provider,
            overwrite=True)

        if os.path.isfile(QSPR_DIR /
                          f'data/{self.target}_alignment.aln-fasta.fasta'):
            self.alignment(ds, f'{self.target}_alignment.aln-fasta.fasta')
        else:
            print(
                'No MSA file supplied, allign the fasta file with sequences. Using: https://www.ebi.ac.uk/jdispatcher/msa/clustalo'
            )
            write_fasta(self.ds_seq, self.target)  # only once
            return

        calc_prot = ProteinDescriptorCalculator(
            desc_sets=[ProDec(sets=["Zscale Hellberg"])],
            msa_provider=self.msa_provider)
        calc_mol = MoleculeDescriptorsCalculator(desc_sets=[
            FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048)
        ])

        ds.prepareDataset(feature_calculators=[calc_prot, calc_mol], )

        if not os.path.isfile(DATA_DIR / f'{self.target}_random_split.json'):
            molecule_split(ds.getDF(), smiles_col, self.target)
        if not os.path.isfile(DATA_DIR / f'{self.target}_scaffold_split.json'):
            scaffold_split(ds, self.smiles_col, self.target)

        print(ds.name)
        ds.save()

        if self.condition == 'descriptor' and not os.path.isfile(
                QSPR_DIR / f'{self.target}_assay_emb.csv'):
            create_assay_fp(ds, self.target)
            create_assay_emb(ds, self.target)
            create_assay_emb_umap(ds, self.target)
        if self.condition == 'descriptor' and not os.path.isfile(
                QSPR_DIR / f'{self.target}_assay_bow.csv'):
            create_assay_bow(ds, self.target)

        return ds

    def load(self, repeats=[0]):
        ds = PCMDataSet.fromFile(
            QSPR_DIR /
            f'data/{self.target}_{self.condition}{f"_{self.task}" if self.task else ""}_df.pkl'
        )
        complete_datasets = []
        final_datasets = []
        # if assay descriptor add desscriptors
        if self.condition == 'descriptor':
            for desc in ['AFP', 'AEMB', 'ABOWS']:  # 'AFP', 'AEMB', 'ABOW', 'ABOWS'
                ds_desc = self.get_assay_desc(ds, desc)
                ds_desc.filter([RepeatsFilter(keep=False)
                                ])  # remove rows with non unique descriptors
                self.apply_splits(ds_desc)  # calculate splits
                complete_datasets.append(ds_desc)
        else:
            ds.filter([RepeatsFilter(keep=False)
                       ])  # remove rows with non unique descriptors
            self.apply_splits(ds)  # calculate random splits
            complete_datasets.append(ds)

        # apply splits
        for complete_dataset in complete_datasets:
            for i in repeats:
                final_datasets.append(
                    self.apply_feature_filter(
                        self.get_split(complete_dataset, i)))

        return final_datasets

    def get_assay_desc(self, ds, desc):
        ds_desc = deepcopy(ds)
        if desc == 'AFP':
            ds_desc.name = f'{ds.name}_AFP'
            assay_fp = DataFrameDescriptorSet(
                pd.read_csv(
                    QSPR_DIR /
                    f'data/{self.target}_assay_fp.csv').set_index('QSPRID'))
            calc_assay_fp = CustomDescriptorsCalculator(desc_sets=[assay_fp])
            ds_desc.addCustomDescriptors(calc_assay_fp, recalculate=True)
        if desc == 'AEMB':
            ds_desc.name = f'{ds.name}_AEMB'
            assay_emb = DataFrameDescriptorSet(
                pd.read_csv(
                    QSPR_DIR /
                    f'data/{self.target}_assay_emb.csv').set_index('QSPRID'))
            calc_assay_emb = CustomDescriptorsCalculator(desc_sets=[assay_emb])
            ds_desc.addCustomDescriptors(calc_assay_emb, recalculate=True)
        if desc == 'AUMAP':
            ds_desc.name = f'{ds.name}_AUMAP'
            assay_emb = DataFrameDescriptorSet(
                pd.read_csv(QSPR_DIR / f'data/{self.target}_assay_emb_umap.csv'
                            ).set_index('QSPRID'))
            calc_assay_emb = CustomDescriptorsCalculator(desc_sets=[assay_emb])
            ds_desc.addCustomDescriptors(calc_assay_emb, recalculate=True)
        if desc == 'ABOWS':
            ds_desc.name = f'{ds.name}_ABOWS'
            assay_bow = DataFrameDescriptorSet(
                pd.read_csv(
                    QSPR_DIR /
                    f'data/{self.target}_assay_bow_stemming.csv').set_index(
                        'QSPRID'))
            calc_assay_bow = CustomDescriptorsCalculator(desc_sets=[assay_bow])
            ds_desc.addCustomDescriptors(calc_assay_bow, recalculate=True)

        return ds_desc

    def apply_splits(self, ds):
        df_random = pd.DataFrame()
        repeat = 3
        with open(DATA_DIR / f"{self.target}_{self.split}_split.json",
                  "r") as fp:
            # Load the dictionary from the file
            split_dict = json.load(fp)
        for i in range(repeat):
            df = ds.getDF()
            df.loc[df[ds.smilesCol].isin(split_dict[str(i)]['train']),
                   f'split_{i}'] = True
            df.loc[df[ds.smilesCol].isin(split_dict[str(i)]['test']),
                   f'split_{i}'] = False
            df_random = pd.concat([df_random, df[f'split_{i}']], axis=1)

        df_random = df_random.reset_index()
        df_random.to_csv(
            QSPR_DIR /
            f'data/{self.target}_{self.condition}{f"_{self.task}" if self.task else ""}_splits{"_scaffold" if self.split == "scaffold" else ""}.csv',
            index=False)

    def get_split(self, ds, i):
        df_split = pd.read_csv(
            QSPR_DIR /
            f'data/{self.target}_{self.condition}{f"_{self.task}" if self.task else ""}_splits{"_scaffold" if self.split == "scaffold" else ""}.csv'
        )

        split = ManualSplit(df_split[f'split_{i}'], True, False)
        ds_split = deepcopy(ds)

        ds_split.split(split, featurize=True)

        ds_split.name = f'{ds_split.name}_{i}'

        return ds_split

    def apply_feature_filter(self, ds):

        # Remove features that have a low variance (<0.05) in the trainingset
        lv = LowVarianceFilter(0.05)

        # Remove features that have a high correlation (>0.9) in the trainingset
        hc = HighCorrelationFilter(0.8)

        feature_filters = [lv, hc]

        ds.filterFeatures(feature_filters)

        ret = ds.generateMetadata()
        path = QSPR_DIR / f"data/{ds.name}_meta.json"
        with open(path, "w") as f:
            json.dump(ret, f)

        return ds

    def yvalue_prop(self, df, target):
        if self.condition == 'MT':
            if not self.task:
                raise Exception("For multi-task, task should be defined")
            if self.task.split('_')[0] == 'topic':
                topic = 'olr_cluster_None'
                file = BERT_DIR / f'descriptions_biobert_None_{self.task.split("_")[1]}.parquet'

                df_info = pd.read_parquet(file)
                df_all = pd.merge(df,
                                  df_info,
                                  left_on='AID',
                                  right_on='chembl_id',
                                  how='left',
                                  indicator=True)
                if len(df_all[df_all['_merge'] == 'left_only']) > 0:
                    print(
                        f"{len(df_all[df_all['_merge'] == 'left_only'])} datapoints put into category 'other' because no chembl description"
                    )
                df_all[topic] = df_all[topic].fillna(value = 'other', axis = 1)
                # add topic to category if in top 100
                n = 100
                if df_all[topic].nunique() > n:
                    largest_topics = df_all[topic].value_counts().head(n).index.tolist()
                    df_all.loc[~df_all[topic].isin(largest_topics), topic] = 'other'
                # turn into string
                df_all[topic] = 'Topic' + df_all[topic].astype(str)
                self.tasks = df_all[topic].unique()
                target_props = [{
                    "name": x,
                    "task": TargetTasks.REGRESSION
                } for x in self.tasks]
                df_pivot = df_all.pivot_table(index='Activity_ID',
                                              columns=topic,
                                              values='pchembl_value',
                                              aggfunc='mean').reset_index()
            else:
                df_info = pd.read_csv(DESC_DIR / 'assay_desc_mapping_mt.csv')
                df_all = pd.merge(df,
                                  df_info,
                                  left_on='AID',
                                  right_on='chembl_id',
                                  how='left',
                                  indicator=True)
                if len(df_all[df_all['_merge'] == 'left_only']) > 0:
                    print('some AIDs not retrieved by chembl query')
                self.tasks = df_all[self.task].unique(
                )  #TODO check if okay to change this to self task instead of task
                target_props = [{
                    "name": x,
                    "task": TargetTasks.REGRESSION
                } for x in self.tasks]
                df_pivot = df_all.pivot_table(index='Activity_ID',
                                              columns=self.task,
                                              values='pchembl_value',
                                              aggfunc='mean').reset_index()
            df = pd.merge(df_all[['Activity_ID', 'SMILES',
                                  'accession']].drop_duplicates(),
                          df_pivot,
                          on='Activity_ID')
        elif self.condition == 'descriptor':
            agg_function = {
                'connectivity': pd.Series.mode,
                'accession': pd.Series.mode,
                'SMILES': pd.Series.mode,
                'pchembl_value': ['median']
            }
            agg_function.update(
                dict.fromkeys([
                    'source', 'CID', 'all_doc_ids', 'type_IC50', 'type_EC50',
                    'type_KD', 'type_Ki'
                ], lambda x: ';'.join(x)))
            agg_function.update(
                {'Activity_class': lambda x: pd.Series.mode(x).to_list()})
            df = df.groupby(['Activity_ID', 'Quality', 'relation',
                             'AID']).agg(agg_function).reset_index()
            df = df.droplevel(1, axis=1)
            target_props = [{
                "name": self.activity_col,
                "task": TargetTasks.REGRESSION
            }]
        else:
            agg_function = {
                'connectivity': pd.Series.mode,
                'accession': pd.Series.mode,
                'SMILES': pd.Series.mode,
                'pchembl_value': ['median']
            }
            agg_function.update(
                dict.fromkeys([
                    'source', 'CID', 'all_doc_ids', 'type_IC50', 'type_EC50',
                    'type_KD', 'type_Ki'
                ], lambda x: ';'.join(x)))
            agg_function.update(
                {'Activity_class': lambda x: pd.Series.mode(x).to_list()})
            df = df.groupby(['Activity_ID', 'Quality',
                             'relation']).agg(agg_function).reset_index()
            df = df.droplevel(1, axis=1)
            target_props = [{
                "name": self.activity_col,
                "task": TargetTasks.REGRESSION
            }]

        return df, target_props

    def get_prot_seq(self, df):
        acc_keys = df[self.target_col].unique()

        # load dataset with sequences
        papyrus = Papyrus(data_dir=DATA_DIR, stereo=False, descriptors=None)

        ds_seq = papyrus.getProteinData(acc_keys,
                                        name=f"{self.target}_seqs",
                                        use_existing=True)

        return ds_seq

    def alignment(self, ds, fname='alignment.aln-fasta.fasta'):
        map, info = self.sequence_provider(ds.getProteinKeys())

        msa_provider = ClustalMSA(out_dir=ds.storeDir, fname=fname)
        msa_provider.parseAlignment(map)
        # print(alignment)
        msa_provider.getFromCache(sorted(map.keys()))

        self.msa_provider = msa_provider

    def sequence_provider(self, acc_keys):
        """
        A function that provides a mapping from accession key to a protein sequence.

        Args:
            acc_keys (list): Accession keys of the protein to get a mapping of sequences for.

        Returns:
            (dict) : Mapping of accession keys to protein sequences.
            (dict) : Additional information to pass to the MSA provider (can be empty).
        """
        map = dict()
        info = dict()
        for i, row in self.ds_seq.iterrows():
            if any(ele not in accepted for ele in row['Sequence']): continue
            map[row['accession']] = row['Sequence']

            # can be omitted
            info[row['accession']] = {
                'Organism': row['Organism'],
                'UniProtID': row['UniProtID'],
            }

        return map, info


def modelling(ds, optimize = False, split='random'):
    parameters = {'loss':  MSEwithNaNLoss(), 'metric': NaNR2Score(), 'verbose': -1, 'lr': 0.1, 'min_data_in_leaf': 50, 'max_depth': 8, 'subsample': 0.8, 'ntrees': 1000}
    # parameters = {'loss':  MSEwithNaNLoss(), 'metric': NaNR2Score(), 'verbose': -1, 'lr': 0.05, 'min_data_in_leaf': 60, 'max_depth': 10, 'subsample': 0.8, 'colsample': 0.8, 'ntrees': 10000}
    if optimize:
        with open(QSPR_DIR / f'models/{ds.name}_PyBoost_base/{ds.name}_PyBoost_base_params.json') as json_file:
            param_dict = json.load(json_file)

        parameters.update(param_dict) #ntrees lower if early stopping occurred

        model = PyBoostModel(
            base_dir='qspr/models/',
            data=ds,
            name=f'{ds.name}{"_scaffold" if split == "scaffold" else ""}_PyBoost_final',
            parameters=parameters
        )

        score_func = NaNR2Scorer('NaNR2', NaNR2Score())
        search_space_trees = {'min_data_in_leaf': ['int', 30, 500], 'max_depth': ['int', 3, 16], 'subsample': ["discrete_uniform", 0.6, 0.9, 0.05], 'colsample': ["discrete_uniform", 0.5, 0.9, 0.05]}
        bayesoptimizer = OptunaOptimization(scoring = score_func, param_grid=search_space_trees, n_trials=50)
        best_params = bayesoptimizer.optimize(model)

        # train and evaluate final model using lower lr and more trees
        best_params.update({'lr': 0.05, 'ntrees': 1000})
        model.setParams(best_params)
        CrossValAssessor(mode=EarlyStoppingMode.RECORDING)(model)
        best_params.update({'ntrees': model.earlyStopping.getEpochs()})
        TestSetAssessor(mode=EarlyStoppingMode.OPTIMAL)(model)
        try:
            model.save()
        except:
            pass

        model.saveParams(best_params)

    else:
        # see if optimal can be lower than 100 using early stopping

        model_base = PyBoostModel(
            base_dir=str(QSPR_DIR / 'models/'),
            data=ds,
            name=f'{ds.name}{"_scaffold" if split == "scaffold" else ""}_PyBoost_base',
            parameters=parameters
        )

        CrossValAssessor(mode=EarlyStoppingMode.RECORDING)(model_base)
        TestSetAssessor(mode=EarlyStoppingMode.OPTIMAL)(model_base)

        model_base.fit(ds.X, ds.y)

        save_params = parameters.copy()
        save_params.update({'loss':  'MSEwithNaNLoss', 'metric': 'NaNR2Score', 'ntrees': model_base.earlyStopping.getEpochs()})
        model_base.setParams(save_params)

        try:
            model_base.save()
        except:
            pass

def molecule_split(df, smiles_col, target):
    repeat = 3
    split_dict = {}
    for i in range(repeat):
        unique_smiles = df[smiles_col].unique().tolist()
        train, test = next(ShuffleSplit(1, test_size=0.1, random_state=i).split(unique_smiles))
        train_smiles = [unique_smiles[i] for i in train]
        test_smiles = [unique_smiles[i] for i in test]
        split_dict.update({i: {'train': train_smiles, 'test': test_smiles}})

    with open(DATA_DIR / f'{target}_random_split.json', 'w') as f:
        json.dump(split_dict, f)


def scaffold_split(ds, smiles_col, target):
    repeat = 3
    split_dict = {}
    split = ScaffoldSplit(dataset=ds, scaffold=Murcko(), test_fraction=0.1, stratify = True)
    for i in range(repeat):
        ds.split(split)
        ds.saveSplit()
        train_smiles = ds.getDF().loc[ds.getDF().Split_IsTrain][smiles_col].to_list()
        test_smiles = ds.getDF().loc[~ds.getDF().Split_IsTrain][smiles_col].to_list()
        split_dict.update({i: {'train': train_smiles, 'test': test_smiles}})

    with open(DATA_DIR / f'{target}_scaffold_split.json', 'w') as f:
        json.dump(split_dict, f)


if __name__ == "__main__":
    os.makedirs(QSPR_DIR / 'data', exist_ok=True)

    args = parser.parse_args()
    smiles_col = 'SMILES'
    activity_col = 'pchembl_value'
    target_col = 'accession'

    conditions = args.condition
    for condition in conditions:
        if condition == 'MT':
            tasks = ['topic_128', 'topic_64', 'topic_32', 'topic_16'] # ['assay_type', 'curated_by', 'confidence_score', 'relationship_type']
        else:
            tasks = [None]
        for task in tasks:
            for split in args.split:
                try:
                    test_dataset = BaseDs(target=args.target,
                                        condition=condition,
                                        task=task,
                                        smiles_col='SMILES',
                                        target_col='accession',
                                        activity_col='pchembl_value',
                                        split=split)

                    if not os.path.isfile(
                            QSPR_DIR /
                            f'data/{args.target}_{condition}{f"_{task}" if task else ""}_meta.json'
                    ):
                        print('Creating dataset')
                        
                        # Load in the data
                        df = pd.read_csv(DATA_DIR / f'filtered_assays_split_{args.target}.csv',
                                        sep=',')

                        # drop columns without pchembl value
                        df = df.dropna(subset=['pchembl_value'])
                        test_dataset.create(df)

                    datasets = test_dataset.load(args.repeats)

                    for dataset in datasets:
                        modelling(dataset, optimize=False, split=split)
                except cupy.cuda.memory.OutOfMemoryError:
                    print(args.target, condition, task)


