#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_dir = '/home/stone/data/numerai/tour88'

def main():
    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv(data_dir + '/numerai_training_data.csv', header=0)
    print("Loaded.")

    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    y = training_data["target"]

    print(X.head())
    #print(X.describe())

    #plot_corr_X(X, y)
    #plot_corr_Xy(X, y)

    #boxplot(X)

    cal_pca(X)

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
