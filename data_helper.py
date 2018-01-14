#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import os.path
import pandas as pd
import numpy as np
#from sklearn import metrics, preprocessing, linear_model
#from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.model_selection import KFold, RepeatedKFold
#from sklearn import svm
#from sklearn.pipeline import Pipeline

data_dir_tmplt = '/home/stone/data/numerai/tour{:d}'

def load_training_data(tour, nrows=None):
    # Set seed for reproducibility
    #np.random.seed(0)
    fpath = get_data_fpath(tour, 'numerai_training_data.csv')

    print("Loading training data from {}".format(fpath))
    # Load the data from the CSV files
    training_data = pd.read_csv(fpath, header=0, nrows=nrows)

    return training_data

def load_training_Xy(tour, nrows=None, onehot=False):
    training_data = load_training_data(tour, nrows)
    return get_Xy(training_data, onehot=onehot)

def get_Xy(ds, onehot=False):
    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(ds) if "feature" in f]

    X = ds[features].as_matrix()
    y = ds["target"].as_matrix()

    if onehot:
        y = dense_to_one_hot(y, np.max(y)+1)

    return X, y

def load_testing_data(tour, nrows=None):
    print("Loading testing data from {}".format(fpath))
    fpath = get_data_fpath(tour, 'numerai_tournament_data.csv')
    prediction_data = pd.read_csv(fpath, header=0, nrows=nrows)
    return prediction_data

def load_testing_Xy(tour, nrows=None, onehot=False):
    ds = loading_testing_data(tour, nrows)
    return get_Xy(ds, onehot=onehot)

def get_data_fpath(tour, fname):
    data_dir = data_dir_tmplt.format(tour)
    fpath = os.path.join(data_dir, fname)
    return fpath

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def eval_one(y_true, y_prob):
    print('probability predicted: min {}, max {}' \
            .format(np.min(y_prob), np.max(y_prob)))
    y_pred = y_prob > thr
    ll = metrics.log_loss(y_true, y_prob)
    print('log loss: {}'.format(ll))
    auc = metrics.roc_auc_score(y_true, y_prob)
    print('AUC: {}'.format(auc))
    acc = metrics.accuracy_score(y_true, y_pred)
    print('accuracy: {}'.format(acc))
    return (ll, auc, acc)

def write_result(results):
    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    results.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
