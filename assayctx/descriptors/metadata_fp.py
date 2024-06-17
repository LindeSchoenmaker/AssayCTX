import json

import pandas as pd
import pystow

DATA_DIR = pystow.join("AssayCTX", "data")

if __name__ == "__main__":
    name = 'all'
    df = pd.read_csv(DATA_DIR / 'assay_desc_mapping_fb_info.csv')
    df['assay_tax_id'] = df['assay_tax_id'].astype('str')
    df['src_id'] = df['src_id'].astype('str')
    df['confidence_score'] = df['confidence_score'].astype('str')

    # number of unique values per property
    df = df.loc[df['assay_type'].isin(['F', 'B'])]
    # also havw pKi ect.
    df.loc[~df['standard_type'].isin(['EC50', 'Ki', 'IC50', 'Kd']), 'standard_type'] = 'other'
    number_unique = pd.Series(df.nunique(axis=0), name = 'number_unique')
    fraction_defined = pd.Series((df.count()/len(df)), name = 'fraction_defined')
    df_info=pd.concat([number_unique,fraction_defined],axis=1).sort_values('fraction_defined', ascending = False)
    print("number of unique values per property")
    print(df_info)
    df_info.to_csv(DATA_DIR / 'chembl_fp_all_fraction_SI.csv')

    fp_dict = {}

    for column_name in ['assay_type', 'standard_type', 'relationship_type', 'src_id', 'curated_by', 'bao_format', 'confidence_score']:
        column = df[column_name]
        unique_values = column.unique()
        d1=zip(unique_values, range(len(unique_values)))
        fp_dict[column_name] = dict(d1)

    print(fp_dict)
    
    with open(DATA_DIR / "chembl_fp_dict.txt", "w") as fp:
        json.dump(fp_dict, fp)  # encode dict into JSON
