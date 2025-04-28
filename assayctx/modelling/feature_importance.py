from qsprpred.data.sources.papyrus import Papyrus

print(f'imported {Papyrus}')
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pystow
from qsprpred.extra.gpu.models.pyboost import PyBoostModel

# from run import BaseDs

FIG_DIR = pystow.join("AssayCTX", "figures", 'feature_importance')


def plot_feature_importance(target,
                            condition,
                            task,
                            imp_type='split',
                            topn=20,
                            plot='barplot'):
    feature_dict = defaultdict(list)
    for repeat in range(3):
        model = PyBoostModel.fromFile(
            f'qspr/models/{target}_{condition}{f"_{task}" if task else ""}_{repeat}_PyBoost_base/{target}_{condition}{f"_{task}" if task else ""}_{repeat}_PyBoost_base_meta.json'
        )
        estimator = model.estimator
        f_imp = estimator.get_feature_importance(
            imp_type=imp_type)  #"Importance type should be 'gain' or 'split'"

        features = []
        prefixes = []
        for ds in model.featureCalculators:
            prefix = ds.getPrefix()
            prefixes.append(prefix)
            features.extend(
                [prefix + '_' + x for x in ds.getDescriptorNames()])

        for key, value in zip(features, f_imp):
            feature_dict[key].append(value)

    features = []
    f_imp = []
    f_imp_err = []
    for key, value in feature_dict.items():
        features.append(key)
        f_imp.append(sum(value) / len(value))
        f_imp_err.append(np.std(value))

    f_imp_arr = np.array(f_imp)

    ind = np.argpartition(f_imp_arr, -topn)[-topn:]
    f_imp_top10 = f_imp_arr[ind]
    features_top10 = [features[i] for i in ind]

    if plot == 'barplot':
        f_imp_err_top10 = [f_imp_err[i] for i in ind]

        fig, ax = plt.subplots()
        y_pos = np.arange(len(features_top10))
        ax.barh(y_pos, f_imp_top10, xerr=f_imp_err_top10, align='center')
        ax.set_yticks(y_pos, labels=features_top10)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(
            FIG_DIR /
            f'feature_importance_{target}_{condition}{f"_{task}" if task else ""}_{imp_type}_top{topn}.png'
        )
        return fig

    if plot == 'pie':
        counts = []
        descriptors = ['PCM', 'Custom']
        for descriptor in descriptors:
            count = sum([True for x in features_top10 if descriptor in x])
            counts.append(count)
        counts.append(topn - sum(counts))
        data = dict(zip(['protein', 'assay', 'ligand'], counts))
        color_dict = dict(zip(["#5790fc", "#f89c20", "#e42536"], counts))
        names = [key for key, value in data.items() if value != 0]
        values = [value for value in data.values() if value != 0]
        colors = [key for key, value in color_dict.items() if value != 0]
        fig, ax = plt.subplots()
        plt.pie(values, labels=names, colors=colors, textprops= {'fontsize':8})
        plt.tight_layout()
        fig.savefig(
            FIG_DIR /
            f'feature_importance_pie_{target}_{condition}{f"_{task}" if task else ""}_{imp_type}_top{topn}.png'
        )
        return fig, values, names, colors


if __name__ == "__main__":
    targets = ['SLCs', 'RTKs', 'GPCRs']
    conditions = ['control', 'descriptor', 'MT']
    imp_types = ['gain']

    split = 'random'
    topn = 30
    i = 1
    fig_final = plt.figure(figsize=(9, 6), dpi=300)
    for imp_type in imp_types:
        for target in targets:
            for condition in conditions:
                if condition == 'control':
                    tasks = [None]
                elif condition == 'descriptor':
                    tasks = ['ABOWS', 'AEMB', 'AFP']
                elif condition == 'MT':
                    tasks = ['topic_128']

                for task in tasks:
                    try:

                        fig, values, names, colors = plot_feature_importance(
                            target.lower(),
                            condition,
                            task,
                            imp_type,
                            topn,
                            plot='pie',
                            )
                        ax = fig_final.add_subplot(3, 5, i)
                        ax.pie(values, labels=names, colors=colors)
                        ax.set_title(f"{target} {task if task else 'control'}", fontsize=14)
                        i += 1
                    except FileNotFoundError:
                        pass
    fig_final.tight_layout()
    fig_final.savefig(FIG_DIR / 'SIFigure4.png')
