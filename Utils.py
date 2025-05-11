import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from random import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
import math
from scipy import interpolate


csfont = {'fontname':'Times New Roman'}

# plots confusion matrix
def plot_cm(cm, label, title):
    fig, ax = plt.subplots()
    im, cbar, cf_percentages, accuracy = heatmap(cm, label, label, title=title,
                                                 ax=ax,
                                                 cmap="Blues", cbarlabel="percentage [%]")
    annotate_heatmap(im, cf_percentages, cm, valfmt="{x:.1f} t")
    fig.tight_layout()


def heatmap(data, row_labels, col_labels, title, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    cf_percentages = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cf_percentages * 100, **kwargs)
    im.set_clim(0, 100)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=15, **csfont)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels, fontsize=13, **csfont)
    ax.set_yticklabels(row_labels, fontsize=13, **csfont)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.ylabel('True label', fontsize=15, **csfont)
    plt.xlabel('Predicted label', fontsize=15, **csfont)
    accuracy = np.trace(data) / float(np.sum(data))
    stats_text = "\nAccuracy={:0.1%}".format(accuracy)
    plt.title(title + stats_text,fontsize=15, **csfont)

    return im, cbar, cf_percentages, accuracy


def annotate_heatmap(im, cf_percentages, data=None, valfmt="{x:.1f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(cf_percentages)
    else:
        threshold = cf_percentages.max() / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Change the text's color depending on the data.
    cf_percentages = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    group_percentages = ["{0:.1%} \n".format(value) for value in cf_percentages.flatten()]
    group_counts = ["{0:0.0f}\n".format(value) for value in data.flatten()]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_percentages, group_counts)]
    box_labels = np.asarray(box_labels).reshape(data.shape[0], data.shape[1])

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(cf_percentages[i, j] > threshold)])
            text = im.axes.text(j, i, box_labels[i, j], **kw)
            texts.append(text)

    return texts

# plot roc curve for one classification results
def plot_roc(fpr, tpr, title):
    plt.figure(figsize=(7, 6))
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, c='r', lw=3, alpha=0.7, label=u'AUC=%.2f' % auc)
    plt.plot((0, 1), (0, 1), c='b', lw=2, ls='--', alpha=0.7, label='baseline = 0.5')
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    ax.spines['top'].set_linewidth(1.3)
    plt.xlabel('False Positive Rate', fontsize=15, **csfont)
    plt.ylabel('True Positive Rate', fontsize=15, **csfont)
    plt.grid()
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
    plt.title(title, fontsize=16, **csfont)
    return auc

# plot several roc curves
def plot_roc_compare(fpr_list, tpr_list, model_name_list, auc_list, title):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    marker_style = ['o', 'v', 's', 'X', 'd', 'P']
    plt.figure(figsize=(7, 6))
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    plt.plot((0, 1), (0, 1), c='b', lw=2, ls='--', alpha=0.7, label='baseline = 0.5')
    for i in range(len(model_name_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, alpha=0.7, c=colors[i], marker=marker_style[i],  markersize=6,
                 label=model_name_list[i] + u'=%.2f' % auc_list[i])
        L =plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=18)
        plt.setp(L.texts, **csfont)

    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1),  fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1),  fontsize=12)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    ax.spines['top'].set_linewidth(1.3)
    plt.xlabel('False Positive Rate', fontsize=22, **csfont)
    plt.ylabel('True Positive Rate', fontsize=22, **csfont)
    plt.grid()
    plt.title(title, fontsize=22, **csfont)
    plt.tight_layout()