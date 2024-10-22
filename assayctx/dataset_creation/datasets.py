"""
Module with datasets from papyrus.

Created by: Linde Schoenmaker
On: 03.05.2023 14:07
"""
import json
import os

import chembl_downloader
import pandas as pd
import pystow
from bindtype.papyrus import add_binding_type_to_papyrus

# from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import (
    consume_chunks,
    keep_accession,
    keep_protein_class,
    keep_quality,
    keep_source,
)
from papyrus_scripts.reader import read_papyrus, read_protein_set
from rdkit import Chem
from rdkit.Chem import Descriptors

DATA_DIR = pystow.join("AssayCTX", "data")
# download_papyrus(version='05.7', structures=True, only_pp=False, descriptors=None, outdir='./')


def read_accessions_txt(files:list):
    """Creates list from text file with accessions, should be formatted with 1 accession per row"""
    accessions = []
    for file in files:
        with open(DATA_DIR / f"{file}.txt", encoding = 'utf-8') as f:
            data = f.read()
            data_into_list = data.split("\n")
            accessions.extend(data_into_list)
    return accessions

def read_accessions_json(file, receptor, from_blast = True):
    """Creates list from text file with accessions, should be formatted with 1 accession per row"""
    with open(DATA_DIR / f"{file}.json") as json_file:
        data = json.load(json_file)

        # get list of accessions for specific receptor & blasted or not
        accessions = data[f'{receptor}{"_blasted" if from_blast else ""}']['accessions']

    return accessions

def get_data(accessions, protein_class, plusplus=False):
    """This is the code to read the entire dataset, filtered on accession, all targets mixed"""
    sample_data = read_papyrus(is3d=False, chunksize=1000000, plusplus=plusplus, source_path='./', version='05.7')
    if accessions:
        filter_accession = keep_accession(sample_data, accessions)
        return consume_chunks(filter_accession, progress=True, total=60)
    elif protein_class:
        protein_data = read_protein_set(source_path='./')
        filter_quality = keep_quality(data=sample_data, min_quality='medium')
        filter_class = keep_protein_class(filter_quality, protein_data, protein_class)
        filter_source = keep_source(filter_class, source=['ChEMBL34'])

        return consume_chunks(filter_source, progress=True, total=60)
    else:
        return consume_chunks(sample_data, progress=True, total=60)

def save(df, name, version='raw'):
    df.to_csv(DATA_DIR / f'{version}_{name}.csv', index=False)

def read(name):
    return pd.read_csv(DATA_DIR / f'raw_{name}.csv')

def quality_info(df):
    accessions = df["accession"].unique()
    for accession in accessions:
        if df.loc[(df['Quality'].isin(['Medium','High'])) & (df['accession'] == accession)].shape[0] == 0:
            print(f'No data for: {accession}')
        else:
            length = df.loc[(df['Quality'] == 'High') & (df['accession'] == accession)].shape[0]
            if length < 100:
                print(f'Less than 100 high quality datapoints for {accession}: {length}')
            else:
                print(f'Number of high quality datapoints for {accession}: {length}')


def split_cells(x):
    # for cells with ; turn into a list
    if isinstance(x, str):
        if ';' in x:
            x = x.split(';')

    return x


def explode(x, level):
    # use level to index list value in cell
    if isinstance(x, list):
        x = x[level]

    return x


def separate_aid(data):
    """From protein-compound row with average pchembl of multiple assays, to multiple rows of each assay with measured pchembl

    In papyrus data of a compound-receptor pair from multiple assays is averaged, for our purposes the assays need to be separated.
    """
    if len(data.index) == 0:
        return data

    # separate multiple studies into own rows
    data['AID'] = data.AID.str.split(pat = ';')
    cols = data.columns.tolist()
    cols.remove("AID")
    res = data.set_index(cols)['AID'].apply(pd.Series).stack()
    res = res.reset_index()
    res.columns = cols + ['level', 'AID']

    # if multiple values in column turn into list
    columns = ['pchembl_value', 'accession', 'SMILES', 'source', 'CID', 'Quality', 'all_doc_ids', 'Activity_class', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other']
    res[columns] = res[columns].applymap(lambda x: split_cells(x))

    # add right column value to AID (use level to index)
    for column in columns:
        res[column] = res.apply(lambda row: explode(row[column], row.level), axis=1)

    res["pchembl_value"] = pd.to_numeric(res["pchembl_value"])

    res = res.drop(labels = ['level'], axis=1)

    res = res.astype(dict.fromkeys(['type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other'], 'string'))

    # Calculate median pchembl_value for compound protein pairs with same assay id
    agg_function = {'connectivity': pd.Series.mode, 'accession': pd.Series.mode, 'SMILES': pd.Series.mode, 'pchembl_value':['median']}
    agg_function.update(dict.fromkeys(['source', 'CID', 'all_doc_ids', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki'], lambda x: ';'.join(x)))
    agg_function.update({'Activity_class': lambda x: pd.Series.mode(x).to_list()})
    unique = res.groupby(['Activity_ID', 'AID', 'Quality', 'relation']).agg(agg_function).reset_index()
    unique = unique.droplevel(1, axis=1)

    return unique


def filter_compounds(df, name):
    # max mol weight 1000
    df['MW'] = df.apply(lambda row: Descriptors.MolWt(Chem.MolFromSmiles(row['SMILES'])), axis=1)
    df = df.loc[df.MW <= 1000]

    # remove allosteric
    if name == 'gpcrs':
        df = add_binding_type_to_papyrus(df, target_type='GPCR', similarity = True)
        df = df.loc[df.BindingType.isin(['Unknown', 'Orthosteric'])]
    if name == 'kinases':
        df = add_binding_type_to_papyrus(df, target_type='Kinase', similarity=True)
        df = df.loc[df.BindingType.isin(['Unknown', 'Orthosteric'])]

    # add filter for receptors with less than 100 tested compounds
    counts = df.accession.value_counts()
    counts =  counts.loc[counts > 100]
    df = df.loc[df.accession.isin(counts.index.tolist())]

    return df

def categorization(df, name):
    if name.split('.')[-1] == 'xlsx':
        df_map = pd.read_excel(DATA_DIR / name)
        df_map = df_map.loc[:, ~df_map.columns.str.contains('^Unnamed')]
    else:
        df_map = pd.read_csv(DATA_DIR / f'manual_categorization_{name}.csv')

    df = pd.merge(df_map, df, left_on='chembl_id', right_on='AID')

    return df

def sparcity(df, name, full = False):
    spars_dict = {}
    levels = ['connectivity']# levels = ['Activity_ID', 'connectivity', 'accession']
    for level in levels:
        df_pivot = df[[level, 'Assay', 'pchembl_value']].pivot_table(index = level, columns='Assay', values='pchembl_value', aggfunc='mean')
        if not full:
            df_pivot = df_pivot[~df_pivot[['cAMP', 'Generic_downstream']].isnull().all(axis=1)]
        df_pivot.to_csv(DATA_DIR / f'pivot_{name}_{level}.csv', index = False)
        spars_dict[level] = df_pivot.count().sum() / (df_pivot.shape[0] * df_pivot.shape[1])

    print(spars_dict)
    df = df.loc[df['connectivity'].isin(df_pivot.index.values.tolist())]
    return df

def check_substructure(smiles, p):
    x = Chem.MolFromSmiles(smiles)
    return x.HasSubstructMatch(p)


def query(chembl_aids):
    chembl_downloader.download_extract_sqlite()
    sql = f"""
    SELECT
        ASSAYS.chembl_id,
        ASSAYS.assay_type
    FROM ASSAYS
    WHERE ASSAYS.chembl_id IN {tuple(chembl_aids)}
    """

    df = chembl_downloader.query(sql)
    df = df.drop_duplicates()

    return df


def filter_assays(df):
    unique_chembl_aids = df.AID.unique()

    df_type = query(unique_chembl_aids)
    df = pd.merge(df, df_type, left_on='AID', right_on='chembl_id')
    df = df.loc[df.assay_type.isin(['F', 'B'])].drop(columns=['assay_type', 'chembl_id'])

    return df

if __name__ == "__main__":
    class_dir = {'slcs': [{'l3': 'SLC superfamily of solute carriers'}], 'all': None, 'gpcrs': [{'l2': 'Family A G protein-coupled receptor'}], 'kinases': [{'l3': 'Protein Kinase'}]}
    accessions = None

    for name, protein_class in class_dir.items():
        if os.path.isfile(DATA_DIR / f'raw_{name}.csv'):
            df = read(name)
        else:
            df = get_data(accessions, protein_class, plusplus=False)
            save(df, name)

        if os.path.isfile(DATA_DIR / 'filtered_{name}.csv'):
            df = pd.read_csv(DATA_DIR / f'filtered_{name}.csv')
        else:
            df = filter_compounds(df, name)
            save(df, name, version='filtered')

        # from unique protein-compound comb rows to unique protein-compound-assay comb rows
        if os.path.isfile(DATA_DIR / f'split_{name}.csv'):
            df= pd.read_csv(DATA_DIR / f'split_{name}.csv')
        else:
            df = separate_aid(df)
            save(df, name, version='split')

        # remove datapoints that are not from binding or functional assays
        df = filter_assays(df)
        save(df, name, version='filtered_assays_split')