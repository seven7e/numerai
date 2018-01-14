#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn import svm
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from util import Cutter

data_dir = '/home/stone/data/numerai/tour88'
data_dir = '/home/stone/data/numerai/tour89'
thr = 0.5

def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv(data_dir + '/numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv(data_dir + '/numerai_tournament_data.csv', header=0)


    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]

    frac = 0.1
    frac = 0.01
    frac = 0.0001
    frac = None
    if frac is not None:
        training_data = training_data.sample(frac=frac)

    X = training_data[features].as_matrix()
    y = training_data["target"].as_matrix()

    # This is your model that will learn to predict
    clf = RandomForestClassifier(
            #max_depth=None,
            max_depth=20,
            n_estimators=20,
            max_features=1,
            min_samples_split=500,
            min_samples_leaf=50,
            n_jobs=1
            )

    #clf = AdaBoostClassifier()

    #clf = svm.SVC(kernel='poly', gamma=1, probability=True)
    #clf = svm.SVC(kernel='linear', gamma=0.1, probability=True)
    #clf = svm.SVC(kernel='rbf', gamma=1, probability=True)

    #clf = Pipeline([('poly', PolynomialFeatures(degree=1, include_bias=False)),
    #    ('logistic', linear_model.LogisticRegression(fit_intercept=True))])

    #clf = linear_model.LogisticRegression()
    #clf = linear_model.LogisticRegression(C=1000)

    clf = XGBClassifier(max_depth=5,
            n_estimators=150)

    model = (Cutter(method='qcut', nbins=100),
            OneHotEncoder(handle_unknown='ignore'))
    model += (clf,)
    model = clf

    print('------------ do cross validation ----------------')
    cross_validate(model, X, y, k=5)

    print('-------------- train and evaluate on whole dataset ------------------')
    train_model(model, X, y)
    pred_eval_one(model, X, y)

    print("---------------- Predicting ------------------")
    pred = predict(model, prediction_data, features=features)
    write_result(pred)

def cross_validate(model, X, y, k=5):
    print('Using model: {}'.format(model))

    num_samples = X.shape[0]

    #random_state = 12883823
    rkf = RepeatedKFold(n_splits=k, n_repeats=1) #, random_state=random_state)
    scores_train = []
    scores_test = []
    for iter_num, (train_index, test_index) in enumerate(rkf.split(X, y)):
        print('--------- iteration {} ---------'.format(iter_num))
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        num_train = y_train.shape[0]
        num_test = y_test.shape[0]
        print('Train/Test = {}({:.2f})/{}({:.2f})'.format(
            num_train, num_train/num_samples, num_test, num_test/num_samples))

        print("Training...")
        # Your model is trained on the training_data
        train_model(model, X_train, y_train)

        score = eval_model(model, X_train, y_train, X_test, y_test)
        scores_train.append(score[0])
        scores_test.append(score[1])

def train_model(model, X, y):
    if isinstance(model, (tuple, list)):
        for m in model[:-1]:
            #print('model:', m)
            #print('X:', X)
            print('training {}, X shape: {}'.format(m, X.shape))
            X = m.fit_transform(X, y)
            #print('out X:', X)
        m_last = model[-1]
    else:
        m_last = model
    print('training {}, X shape: {}'.format(m_last, X.shape))
    m_last.fit(X, y)

def model_predict(model, X):
    if isinstance(model, (tuple, list)):
        for m in model[:-1]:
            #print('model:', m)
            #print('X:', X)
            print('transforming {}, X shape: {}'.format(m, X.shape))
            X = m.transform(X)
            #print('out X:', X)
        m_last = model[-1]
    else:
        m_last = model
    print('predicting {}, X shape: {}'.format(m_last, X.shape))
    y_prediction = m_last.predict_proba(X)
    pos_prob = y_prediction[:, 1]
    return pos_prob

def predict(model, prediction_data, features):
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    x_prediction = prediction_data[features].as_matrix()
    results = model_predict(model, x_prediction)

    validation_index = prediction_data['data_type'] == 'validation'
    y_val = prediction_data['target'].as_matrix()[validation_index]
    y_pred_val = results[validation_index]
    print('Evaluating model on validation set')
    score_test = eval_one(y_val, y_pred_val)

    ids = prediction_data["id"]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(ids).join(results_df)

    return joined

def eval_model(model, X_train, y_train, X_test, y_test):
    print('Evaluating model...')
    print('train set:')
    score_train = pred_eval_one(model, X_train, y_train)
    print('test set:')
    score_test = pred_eval_one(model, X_test, y_test)
    return score_train, score_test

def pred_eval_one(model, X, y):
    prob_pos = model_predict(model, X)
    eval_one(y, prob_pos)

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
