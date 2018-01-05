#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    boxplot(X)

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
