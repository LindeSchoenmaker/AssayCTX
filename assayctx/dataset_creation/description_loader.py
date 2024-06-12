import chembl_downloader
import matplotlib.pyplot as plt
import pystow

path = chembl_downloader.download_extract_sqlite()

DATA_DIR = pystow.join("AssayCTX", "data")
FIG_DIR = pystow.join("AssayCTX", "figures")

sql = """
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
"""

df = chembl_downloader.query(sql, version='33')

# keep binding and functional assays
df = df.loc[df['assay_type'].isin(['F', 'B'])]

# plot length
df['desc_length']  = df['description'].str.len()
fig, ax = plt.subplots()
df.hist(column='desc_length', bins=100, ax=ax)
fig.savefig(FIG_DIR / 'desc_length.png')

# save data
df.to_csv(DATA_DIR / "assay_desc_mapping_fb_info.csv", index=False)

# save descriptions and assay type
df = df[['desc_length', 'chembl_id', 'description', 'assay_type']]
df.to_csv(DATA_DIR / "assay_desc_mapping_fb.csv", index=False)
