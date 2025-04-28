from qsprpred.data.sources.papyrus import Papyrus

print(Papyrus)
import statistics
from functools import partial

import pandas as pd
import pystow
from qsprpred.extra.gpu.models.pyboost import NaNR2Score, NaNRMSEScore
from qsprpred.models.metrics import SklearnMetric
from scipy.stats import kendalltau
from sklearn.metrics import make_scorer

BERT_DIR = pystow.join("AssayCTX", "bert")
DATA_DIR = pystow.join("AssayCTX", "data")
QSPR_DIR = pystow.join("AssayCTX", "qspr")
FIG_DIR = pystow.join("AssayCTX", "figures")

if __name__ == "__main__":
    df_full = pd.DataFrame()
    score_func = SklearnMetric(name='kendaltau',
                               func=partial(kendalltau, nan_policy='omit', method='asymptotic'),
                               scorer=make_scorer(
                                   partial(kendalltau, nan_policy='omit', method='asymptotic')))
    scorer_dict = {
        'r2': NaNR2Score(),
        'rmse': NaNRMSEScore(),
        'kt': score_func
    }
    targets = ['gpcrs']
    splits = ['random', 'scaffold']

    conditions = ["MT"]
    metrics = ['kt']

    for target in targets:
        df_target = pd.DataFrame()
        for split in splits:
            df_metrics = pd.DataFrame()
            for condition in conditions:
                tasks = [None]
                if condition == 'MT':
                    tasks = ['topic_128']
                elif condition == 'descriptor':
                    tasks = ['AFP', 'ABOWS', 'AEMB']
                for task in tasks:
                    for j, metric in enumerate(metrics):
                        scorer = scorer_dict[metric]
                        values = []
                        for i in range(3):
                            name = f'{target}_{condition}{f"_{task}" if task else ""}_{i}{"_scaffold" if split == "scaffold" else ""}_PyBoost_base'
                            assessor = 'ind'
                            df = pd.read_table(
                                QSPR_DIR /
                                f'models/{name}/{name}.{assessor}.tsv').set_index(
                                    'QSPRID')
                            if condition == 'MT':
                                df = df.dropna(axis=1, how='all')
                                print(len(df))
                                labels = [
                                    ele.replace('Label', '') for ele in df.columns
                                    if 'Label' in ele
                                ]
                                predictions = [
                                    ele.replace('Prediction', '') for ele in df.columns
                                    if 'Prediction' in ele
                                ]
                                overlap = list(set(labels) & set(predictions))
                                coef = scorer(
                                    df[[ele + 'Label' for ele in overlap]].values,
                                    df[[ele + 'Prediction' for ele in overlap]].values)
                            else:
                                property_name = 'pchembl_value'
                                coef = scorer(df[f"{property_name}_Label"],
                                            df[f"{property_name}_Prediction"])
                            if metric == 'kt':
                                coef = coef[0]
                            values.append(coef)
                        mean = "%.2f" % (sum(values) / len(values))
                        try:
                            std = "%.2f" % statistics.stdev(values)
                        except AttributeError:
                            std = 'NaN'
                        if j == 0:
                            df_metrics = pd.concat([
                                df_metrics,
                                pd.DataFrame({metric: [mean], f'{metric}_std': [std]},
                                            index=pd.MultiIndex.from_tuples([
                                                (condition, task),
                                            ]))
                            ],
                                                axis=0)
                        else:
                            df_metrics.loc[pd.MultiIndex.from_tuples([
                                (condition, task),
                            ]), metric] = mean
                            df_metrics.loc[pd.MultiIndex.from_tuples([
                                (condition, task),
                            ]), f'{metric}_std'] = std
            df_metrics['target'] = target
            df_metrics['split'] = split
            df_target = pd.concat([df_target, df_metrics])
            # fig, ax = plt.subplots(layout='constrained', figsize=(8,5))

            # x = np.arange(4)  # the label locations
            # width = 0.2  # the width of the bars
            # multiplier = 0
            # colors = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

            # for i, split in enumerate(splits):
            #     n = i
            #     offset = width * multiplier
            #     rects = ax.bar(x + offset, df_target.loc[df_target.split == split][metrics[0]].values, width, yerr=df_target.loc[df_target.split == split]['r2_std'].values, capsize=2, color=colors[i])
            #     multiplier += 1

            # plt.savefig(FIG_DIR / 'test.png')

            
        df_full = pd.concat([df_full, df_target])
    
    print(df_full)
    df_full.to_csv(DATA_DIR / 'model_performance_metrics_MT_asymptotic.csv')
