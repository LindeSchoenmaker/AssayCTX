import cuml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystow
import seaborn as sns


def UMAP_main_text():
    columns = ['assay_type', 'assay_tax_id', 'confidence_score', 'curated_by', 'standard_type', 'bao_format']

    addition[columns] = helper[columns]

    plt.figure(figsize=(10, 8), dpi=600)
    plt.subplots_adjust(hspace=0.5)


    for i, color_by in enumerate(columns):
        ax = plt.subplot(2, 3, i + 1)
        # bg points
        sns.scatterplot(
            data=addition,
            x="x",
            y="y",
            c="grey",
            alpha=0.1,
            s=4,
            ax=ax,
        )

        addition[color_by] = addition[color_by].astype(str)
        if len(addition[color_by].unique()) > 5:
            plot_df = addition.loc[addition[color_by].isin(addition[color_by].value_counts()[:5].index.tolist())]
        else:
            plot_df = addition
        plot_df[color_by] = plot_df[color_by].apply(lambda x : x.replace('assay_tax_id_', '').replace('confidence_score_', '').replace('_', ' ').replace('.0', '').replace('BAO ', ''))
        # if i > 3: continue
        
        # ax.scatter(plot_df['x'], plot_df['y'], s=4, c=plot_df[color_by], cmap='bright')
        # labeled points
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue=color_by,
            hue_order=sorted(plot_df[color_by].unique().tolist()),
            palette="bright",
            s=4,
            ax=ax,
        )
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        new_title = f"{color_by.replace('_x', '').replace('_', ' ').title()}" #, fontsize=10)
        ax.set_title(new_title, fontsize=14, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set(xticklabels=[], yticklabels=[])
        
        ax.legend(fontsize=8, handletextpad=0.1, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=ax.transAxes)
        for count, legend_handle in enumerate(ax.get_legend().legend_handles):
            legend_handle.set(markersize = 5, alpha = 0.8, markeredgewidth=0)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "embeddings_main_text.png")


def UMAP_SI():
    columns = ['assay_type', 'topic',
        'assay_test_type', 'assay_category', 'assay_cell_type',
        'assay_organism', 'assay_tax_id', 'assay_strain', 'assay_tissue',
        'assay_subcellular_fraction', 'relationship_type', 'bao_format',
        'confidence_score', 'curated_by', 'src_id', 'cell_id',
        'tissue_id', 'variant_id', 'aidx', 'pref_name', 'standard_type',
        'journal', 'year', 'pubmed_id', 'doi']

    # add helper["topic"] to addition
    addition[columns] = helper[columns]

    plt.figure(figsize=(10, 10), dpi=600)
    plt.subplots_adjust(hspace=0.5)

    for i, color_by in enumerate(columns):
        addition[color_by] = addition[color_by].astype(str)
        if len(addition[color_by].unique()) > 5:
            plot_df = addition.loc[addition[color_by].isin(addition[color_by].value_counts()[:5].index.tolist())]
        else:
            plot_df = addition
        # if i > 3: continue
        ax = plt.subplot(5, 5, i + 1)
        sns.scatterplot(
            # data=dataframe,
            x=plot_df["x"],
            y=plot_df["y"],
            hue=plot_df[color_by],
            palette="bright",
            s=4,
            ax=ax,
            legend=True,
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"{color_by.replace('_x', '').replace('_', ' ').title()}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set(xticklabels=[], yticklabels=[])
        labels = sorted([string.replace('_', ' ') for string in plot_df[color_by].unique().tolist()])
        ax.legend(fontsize=3, loc='upper left', labels=labels)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "embeddings_SI.png")

if __name__ == "__main__":
    # set data directories using pystow
    DATA_DIR = pystow.join("AssayCTX", "data")

    # set figure directory using pystow
    FIG_DIR = pystow.join("AssayCTX", "figures")

    info = pd.read_csv(DATA_DIR / "assay_desc_mapping_fb_info.csv")
    # topics = pd.read_parquet(DATA_DIR / "descriptions_biobert.parquet")
    sentence_vectors = pd.read_parquet(DATA_DIR / "sentence_vectors.parquet")[['description', 'word_vectors']]

    sentence_vectors = sentence_vectors[sentence_vectors["description"].notna()].drop_duplicates(subset=["description"])

    # merge descriptions and topics
    # helper = descriptions.merge(topics, left_on="description", how="left", right_on="description")

    # The following cells sample 10 % of assay descriptions from ChEMBL for clearer visualizations.
    helper = sentence_vectors.merge(info, left_on="description", how="left", right_on="description")
    helper = helper.sample(frac=0.1, random_state=40)
    helper = helper.dropna(subset = ["word_vectors"]).reset_index()

    # UMAP plot for ChEMBL embeddings
    reducer = cuml.UMAP(n_components=5, n_neighbors=15, min_dist=0.1, random_state=42)

    embeddings = np.array(helper["word_vectors"].to_list())
    embedding = reducer.fit_transform(embeddings)

    addition = pd.DataFrame(embedding, columns=["x", "y", "z1", "z2", "z3"])

    # UMAP_SI()
    UMAP_main_text()