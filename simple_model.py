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
from model_helper import eval_predict as eval_one
from model_helper import print_perfs
import data_helper
from hpopt import xgboost_param_convert, SEED

tour = 89
# tour = 90
thr = 0.5

def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    nrows = None
    # Load the data from the CSV files
    training_data = data_helper.load_training_data(tour, nrows=nrows)
    #training_data = pd.read_csv(data_dir + '/numerai_training_data.csv', header=0)

    #prediction_data = pd.read_csv(data_dir + '/numerai_tournament_data.csv', header=0)
    prediction_data = data_helper.load_testing_data(tour, nrows=nrows)

    features = data_helper.get_feature_names(training_data)

    # Transform the loaded CSV data into numpy arrays

    frac = 0.1
    frac = 0.01
    frac = 0.0001
    frac = None
    if frac is not None:
        training_data = training_data.sample(frac=frac)

    X, y = data_helper.get_Xy(training_data)

    # This is your model that will learn to predict
    rfc = RandomForestClassifier(
            #max_depth=None,
            max_depth=10,
            n_estimators=20,
            max_features=1,
            min_samples_split=500,
            min_samples_leaf=50,
            n_jobs=1
            )

    adbr = AdaBoostClassifier()

    #clf = svm.SVC(kernel='poly', gamma=1, probability=True)
    svc_linear = clf = svm.SVC(kernel='linear', gamma=0.1, probability=True)
    #clf = svm.SVC(kernel='rbf', gamma=1, probability=True)

    #clf = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
    #    ('logistic', linear_model.LogisticRegression(fit_intercept=True))])

    lr = linear_model.LogisticRegression()
    lr1 = linear_model.LogisticRegression(C=1000)

    xgbr = XGBClassifier(n_jobs=2)
    xgbr_linear = XGBClassifier(n_jobs=2, booster='gblinear')
    xgbr2 = XGBClassifier(max_depth=3,
            n_estimators=200)

    xgbr_opt_param = {
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': 2,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': SEED,
    }
    tmp = {'eta': 0.07500000000000001, 'n_estimators': 21.0, 'gamma': 0.75, 'colsample_bytree': 0.8500000000000001, 'min_child_weight': 4.0, 'subsample': 0.75, 'max_depth': 3}
    xgbr_opt_param.update(tmp)
    xgbr_opt = XGBClassifier(**(xgboost_param_convert(xgbr_opt_param)))

    #model = (Cutter(method='qcut', nbins=100),
    #        OneHotEncoder(handle_unknown='ignore'))
    #model += (clf,)
    models = [
            #('logistic', lr),
            #('logistic-C1000', lr1),
            #('linear SVC', svc_linear),  too slow
            #('randomforest', rfc),
            #('adaboost', adbr),
            #('xgbr-default', xgbr),
            ('xgbr-linear', xgbr_linear),
            # ('xgbr-200', xgbr2),
            # ('xgbr-opt', xgbr_opt),
            ]

    perfs = train_eval_model(models, X, y, prediction_data, features=features)
    print_perfs(models, perfs)

def train_eval_model(model, *args, **kwds):
    if isinstance(model, list):
        ret = train_eval_model_list(model, *args, **kwds)
    else:
        ret = train_eval_model_list([(model.__class__.__name__, model)], *args, **kwds)
    return ret

def train_eval_model_list(model_list, *args, **kwds):
    perfs = []
    for name, model in model_list:
        print('============= Train and test model "{}" =============='.format(name))
        print('model details: {!s:s}'.format(model))
        perf = train_eval_model_one(model, *args, model_name=name, **kwds)
        perfs.append((name, perf))
    return perfs

def train_eval_model_one(model, X, y, prediction_data, model_name='default', features=None):
    print('------------ do cross validation ----------------')
    perf_cv = cross_validate(model, X, y, k=5)

    print('-------------- train and evaluate on whole dataset ------------------')
    train_model(model, X, y)
    perf_train = pred_eval_one(model, X, y)

    print("---------------- Predicting ------------------")
    pred, perf_test = predict_test(model, prediction_data, features=features)
    write_result(pred, model_name)

    return perf_cv, perf_train, perf_test

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
    return (scores_train, scores_test)

def train_model(model, X, y):
    if isinstance(model, (tuple, list)):
        for m in model[:-1]:
            #print('model:', m)
            #print('X:', X)
            print('training {}, X shape: {}'.format(m, X.shape))
            #print('training with X shape: {}'.format(X.shape))
            X = m.fit_transform(X, y)
            #print('out X:', X)
        m_last = model[-1]
        print('training {}, X shape: {}'.format(m_last, X.shape))
    else:
        m_last = model
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
        print('predicting {}, X shape: {}'.format(m_last, X.shape))
    else:
        m_last = model
    y_prediction = m_last.predict_proba(X)
    pos_prob = y_prediction[:, 1]
    return pos_prob

def predict_test(model, prediction_data, features):
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    x_prediction = prediction_data[features].as_matrix()
    y_prob = model_predict(model, x_prediction)

    validation_index = prediction_data['data_type'] == 'validation'
    y_val = prediction_data['target'].as_matrix()[validation_index]
    y_pred_val = y_prob[validation_index]
    print('Evaluating model on validation set')
    score_test = eval_one(y_val, y_pred_val)

    ids = prediction_data["id"]
    results_df = pd.DataFrame(data={'probability':y_prob})
    joined = pd.DataFrame(ids).join(results_df)

    return joined, score_test

def eval_model(model, X_train, y_train, X_test, y_test):
    print('Evaluating model...')
    print('  train set:')
    score_train = pred_eval_one(model, X_train, y_train)
    print('  test set:')
    score_test = pred_eval_one(model, X_test, y_test)
    return score_train, score_test

def pred_eval_one(model, X, y):
    prob_pos = model_predict(model, X)
    return eval_one(y, prob_pos)

def write_result(results, suffix=''):
    fname = "predictions-{}.csv".format(suffix)
    print("Writing predictions to {}".format(fname))
    # Save the predictions out to a CSV file
    results.to_csv(fname, index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
