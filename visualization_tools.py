import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np
import itertools

import data_handling as dh

def plot_drug_dependency_matrix(table):
    drug_combinations = []
    for i in table['Drugs']:
        if len(i) == 1:
            drug_combinations.append((i[0], i[0]))
        elif len(i) > 1:
            for j in itertools.combinations(i, 2):
                drug_combinations.append(j)
        else:
            pass
    distinct_drugs = dh.get_distinct_drugs(table)
    conf = confusion_matrix(np.array(drug_combinations)[:, 0], np.array(drug_combinations)[:, 1], labels = distinct_drugs)
    # Make confusion matrix symmetric
    conf = conf + conf.T - np.diag(conf.diagonal())
    conf_masked = conf.astype('float')
    conf_masked[conf_masked == 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(conf_masked)

    ax.set_xticks(np.arange(len(distinct_drugs)))
    ax.set_yticks(np.arange(len(distinct_drugs)))

    ax.set_xticklabels(distinct_drugs)
    ax.set_yticklabels(distinct_drugs)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    cb = plt.colorbar(im, ax=ax, fraction=0.0235, aspect=40)
    cb.ax.tick_params(labelsize=10)

    plt.show()

def plot_timeline_clinical_trials(table, column='First Posted'):
    dates = [i.to_pydatetime() for i in table.loc[:, column]]
    distinct_dates = list(Counter(dates).keys())
    levels = list(Counter(dates).values())

    fig, ax = plt.subplots(figsize=(10, 2), constrained_layout=True)
    ax.set(title='Clinical Trials: ' + column)
    markerline, stemline, baseline = ax.stem(distinct_dates, levels,
                                             linefmt="C3-", basefmt="k-",
                                             use_line_collection=True)
    ax.set_ylabel('Number clinical trials')
    plt.setp(markerline, mec="k", mfc="w", zorder=3)
    markerline.set_ydata(np.zeros(len(distinct_dates)))
    plt.show()
