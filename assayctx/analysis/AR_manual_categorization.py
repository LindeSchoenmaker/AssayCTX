import chembl_downloader
import pandas as pd
import pystow

DATA_DIR = pystow.join("AssayCTX", "data")

def unique_assays(df):
    df_aids = [entry.split(';') for entry in df.AID.unique().tolist()]
    flat_aids = [item for sublist in df_aids for item in sublist]
    unique_aids = set(flat_aids)
    unique_chembl_aids = [item for item in unique_aids if item.startswith('CHEMBL')]

    return unique_chembl_aids

def query(name, chembl_aids):
    path = chembl_downloader.download_extract_sqlite()
    sql = f"""
    SELECT
        ASSAYS.chembl_id,
        ASSAYS.description,
        ASSAYS.assay_type,
        ASSAYS.assay_test_type,
        ASSAYS.assay_category,
        ASSAYS.assay_cell_type,
        ASSAYS.assay_organism,
        ASSAYS.assay_tax_id,
        ASSAYS.assay_strain,
        ASSAYS.assay_tissue,
        ASSAYS.assay_subcellular_fraction,
        ASSAYS.relationship_type,
        ASSAYS.bao_format,
        ASSAYS.confidence_score,
        ASSAYS.curated_by,
        ASSAYS.src_id,
        ASSAYS.src_assay_id,
        ASSAYS.cell_id,
        ASSAYS.tissue_id,
        ASSAYS.variant_id,
        ASSAYS.AIDX,
        TARGET_DICTIONARY.pref_name,
        ACTIVITIES.standard_type,
        DOCS.journal,
        DOCS.year,
        DOCS.pubmed_id,
        DOCS.doi
    FROM ASSAYS
    JOIN TARGET_DICTIONARY ON ASSAYS.tid == TARGET_DICTIONARY.tid
    JOIN ACTIVITIES ON ASSAYS.assay_id == ACTIVITIES.assay_id 
    JOIN DOCS ON ASSAYS.doc_id == DOCS.doc_id 
    WHERE ASSAYS.chembl_id IN {tuple(chembl_aids)}
    """

    df = chembl_downloader.query(sql)
    df = df.drop_duplicates()

    df.to_csv(DATA_DIR / f'assay_description_{name}.csv', index=False)

    return df

def apply_categorization(df):
    df['meta_target'] = 'not specified'

    df.loc[df['standard_type'] == 'Ki', 'meta_target'] = 'ligand displacement'

    df.loc[df['description'].str.contains('ligand displacement', case=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('displacement', case=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('dissociation', case=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('radioligand', case=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('radiolabeled', case=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('3H'), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('[35S]', regex=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('125I'), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('[125I]', regex=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('[125 I]', regex=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('14C'), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('[14C]', regex=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('scintillation', case=False), 'meta_target'] = 'ligand displacement'
    df.loc[df['description'].str.contains('proximity', case=False), 'detection_technology'] = 'scintillation proximity'
    df.loc[df['description'].str.contains('spr', case=False), ['meta_target', 'detection_technology']] = ['ligand displacement', 'surface plasmon resonance']
    df.loc[df['description'].str.contains('multiple cycle kinet', case=False), ['meta_target', 'detection_technology']] = ['ligand displacement', 'surface plasmon resonance']

    df.loc[df['description'].str.contains('cAMP', case=False), 'meta_target'] = 'cAMP'
    df.loc[df['description'].str.contains('cyclic AMP', case=False), 'meta_target'] = 'cAMP'
    df.loc[df['description'].str.contains('cyclic-AMP', case=False), 'meta_target'] = 'cAMP'
    df.loc[df['description'].str.contains('phosphorylated CREB', case=False), 'meta_target'] = 'cAMP'
    df.loc[df['description'].str.contains('phosphorylated CREB', case=False), ['detection_technology', 'meta_target_process']] = ['flow cytometry', 'gene reporter']
    df.loc[df['description'].str.contains('HTRF', case=False), 'detection_technology'] = 'HTRF'
    df.loc[df['description'].str.contains('luciferase', case=False), ['meta_target', 'detection_technology', 'meta_target_process']] = ['cAMP', 'luminescence', 'production']
    df.loc[df['description'].str.contains('galactosidase', case=False), ['meta_target', 'detection_technology', 'meta_target_process']] = ['cAMP', 'luminescence', 'gene reporter']
    df.loc[df['description'].str.contains('CREB', case=True), 'meta_target'] = 'cAMP'

    df.loc[df['description'].str.contains('current response', case=False), 'meta_target'] = 'Ca2+'
    df.loc[df['description'].str.contains('calcium', case=False), 'meta_target'] = 'Ca2+'
    df.loc[df['description'].str.contains('Ca2+'), 'meta_target'] = 'Ca2+'
    df.loc[df['description'].str.contains('Ca+'), 'meta_target'] = 'Ca2+'
    df.loc[df['description'].str.contains('FLIPR'), 'meta_target'] = 'Ca2+'
    df.loc[df['description'].str.contains('clamp', case=False), 'meta_target'] = 'Ca2+'
    df.loc[df['description'].str.contains('electrical', case=False), 'meta_target'] = 'Ca2+'

    df.loc[df['description'].str.contains('inositol phosphate', case=False), 'meta_target'] = 'IP'

    df.loc[df['description'].str.contains('adenylyl cyclase', case=False), 'meta_target'] = 'cAMP'
    df.loc[df['description'].str.contains('adenylyl cylase', case=False), 'meta_target'] = 'cAMP'
    df.loc[df['description'].str.contains('adenylate cyclase', case=False), 'meta_target'] = 'cAMP'


    df.loc[df['description'].str.contains('ERK1/2 phosphorylation'), 'meta_target'] = 'ERK1/2'
    df.loc[df['description'].str.contains('ERK'), 'meta_target'] = 'ERK1/2'

    df.loc[df['description'].str.contains('beta-arrestin2'), 'meta_target'] = 'beta-arrestin'
    df.loc[df['description'].str.contains('arrestin'), 'meta_target'] = 'beta-arrestin'

    df.loc[df['description'].str.contains('GTPPgammaS'), 'meta_target'] = 'GTPPgammaS'
    df.loc[df['description'].str.contains('GTP'), 'meta_target'] = 'GTPPgammaS'

    df.loc[df['description'].str.contains('reactive oxygen species'), 'meta_target'] = 'oxidative stress'
    df.loc[df['description'].str.contains('superoxide production'), 'meta_target'] = 'oxidative stress'


    df.loc[df['description'].str.contains('LacZ reporter'), 'meta_target'] = 'reporter gene'
    df.loc[df['description'].str.contains('SPAP'), 'meta_target'] = 'reporter gene'

    df.loc[df['description'].str.contains('IL-2', regex=False), 'meta_target'] = 'inflammation'
    df.loc[df['description'].str.contains('IL-6', regex=False), 'meta_target'] = 'inflammation'

    df.loc[df['description'].str.contains('erythrocyte morphology', regex=False), 'meta_target'] = 'morphology'
    df.loc[df['description'].str.contains('Impedance', case=False), 'meta_target'] = 'morphology'

    df.loc[df['description'].str.contains('propidium iodide'), 'meta_target'] = 'cell death'

    df.loc[df['description'].str.contains('vasodilation'), 'meta_target'] = 'in vivo fysiology'
    df.loc[df['description'].str.contains('relaxant'), 'meta_target'] = 'in vivo fysiology'

    return df


if __name__ == "__main__":
    accessions = ["P30542", "P29274", "P29275", "P0DMS8"]
    name = "AR"
    # only keep adenosine receptor information
    df = pd.read_csv(DATA_DIR / 'raw_gpcrs.csv')
    df = df.loc[df['accession'].isin(accessions)]

    # get assay AIDs and query chembl to get descriptions
    chembl_aids = unique_assays(df)
    df_descriptions = query(name, chembl_aids)

    # categorize descriptions
    df_descriptions = apply_categorization(df_descriptions)
    df_descriptions.to_csv(DATA_DIR / f'{name}_categorized.csv')
