#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

debug_with_random = False

data_dir = '/home/stone/data/numerai/tour88'
data_dir = '/home/stone/data/numerai/tour89'

def main():
    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv(data_dir + '/numerai_training_data.csv', header=0)
    print("Loaded.")

    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    y = training_data["target"]

    if debug_with_random:
        X = pd.DataFrame(randn(*X.shape))

    test_hist(X)

    print('quick look at original data')
    #quicklook(X, y)

    X_pca = cal_pca(X)

    print('quick look at PCA transformed data')
    quicklook(X_pca, y)

def test_hist(X):
    x = np.array(X.ix[:, 0])

    bins_list = ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']
    bins = 'auto'
    for i, bins in enumerate(bins_list):
        plt.subplot(3, 3, i+1)
        plt.hist(x, bins=bins)
        plt.title(bins)

    def cut_plot(i, func, ncut=100):
        plt.subplot(3, 3, len(bins_list) + i)
        cutted, bins = func(x, ncut, retbins=True, labels=False)
        #print(cutted)
        #print('bins:', bins)
        count = pd.Series(cutted).value_counts() #.sort_index()
        count_comp = np.full(ncut, np.nan)
        for i, v in enumerate(count):
            if i in count:
                count_comp[i] = count[i]
        plt.plot(bins[:-1], count_comp)
        #print()
        #print(cutted.value_counts())

    cut_plot(1, pd.cut)
    cut_plot(2, pd.qcut)

    plt.show()
    sys.exit()

def quicklook(X, y):
    X = pd.DataFrame(X)
    print(X.head())
    print(X.describe())

    boxplot(X)
    plot_hist(X, y)

    plot_corr_X(X, y)
    plot_corr_Xy(X, y)

def cal_pca(X):
    #pca = PCA(n_components=2)
    pca = PCA()
    pca.fit(X)
    print('PCA analysis with {}'.format(pca))
    log = np.log
    print('explained variance: ', pca.explained_variance_)
    plt.plot(log(pca.explained_variance_))
    plt.show()
    print('explained variance ratio: ', pca.explained_variance_ratio_)
    plt.plot(log(pca.explained_variance_ratio_))
    plt.show()
    print('singular values:', pca.singular_values_)
    plt.plot(log(pca.singular_values_))
    plt.show()
    return pca.transform(X)

def boxplot(X):
    X.boxplot()
    plt.show()

def plot_corr_Xy(X, y):
    c = X.apply(lambda c: c.corr(y))
    plt.plot(c.as_matrix())
    plt.show()

def plot_corr_X(X, y):
    #print(np.abs(X.corr()) > 0.5)
    c = X.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(c, interpolation='nearest')
    fig.colorbar(cax)
    plt.show()

def plot_hist(X, y):
    X.hist(bins=30)
    plt.show()

    y.hist()
    plt.show()
    return

if __name__ == '__main__':
    main()
