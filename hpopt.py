#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run an XGBoost model with hyperparmaters that are optimized using hyperopt
# The output of the script are the best hyperparmaters
# The optimization part using hyperopt is partly inspired from the following script:
# https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py

# Data wrangling
import pandas as pd

# Scientific
import numpy as np

# Machine learning
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate

# Hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import data_helper

# Some constants
SEED = 314159265
VALID_SIZE = 0.2
TARGET = 'outcome'
NJOBS = 3

def print_one(type, score):
    print('    {}\tmean: {:.5f}, std: {:.5f}'. \
        format(type, score.mean(), score.std()))

def print_score(type, score_train, score_test):
    if type == 'neg_log_loss':
        type = 'log_loss'
        score_train = -score_train
        score_test = -score_test
    print('  ' + type)
    print_one('train', score_train)
    print_one('test', score_test)

#-------------------------------------------------#
# Scoring and optimization functions
def make_scorer(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_SIZE,
                                    random_state=SEED)

    print('The training set is of length: ', X_train.shape)
    print('The validation set is of length: ', X_valid.shape)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def score(params):
        print("Training with params: ")
        print(params)
        num_round = int(params['n_estimators'])
        del params['n_estimators']
        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
        gbm_model = xgb.train(params, dtrain, num_round,
                              evals=watchlist,
                              verbose_eval=True)
        predictions = gbm_model.predict(dvalid,
            ntree_limit=gbm_model.best_iteration + 1)
        score = roc_auc_score(y_valid, predictions)
        # TODO: Add the importance for the selected features
        print("\tScore {0}\n\n".format(score))
        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {'loss': loss, 'status': STATUS_OK}
    return score

def xgboost_param_convert(params):
    p = { k:v for k, v in params.items() \
        if k not in {'eta', 'reg_alpha', 'reg_lambda'} }

    if 'eta' in params:
        p['learning_rate'] = params['eta']

    for k in ['n_estimators', 'max_depth']:
        if k in params:
            p[k] = int(params[k])

    if 'nthread' in params:
        p['n_jobs'] = int(params['nthread'])

    if 'silent' in params:
        p['silent'] = params['silent'] == 1

    return p

def make_cv_scorer(X, y, cv=5, scoring=['neg_log_loss', 'roc_auc', 'accuracy']):

    print('The dataset for cross validation is of size:', X.shape)

    def cal_score(params):
        p = xgboost_param_convert(params)

        clf = xgb.XGBClassifier(**p)
        print('validate on model:', clf)

        # scores = cross_val_score(clf, X, y, scoring='neg_log_loss',
        scores = cross_validate(clf, X, y,
            scoring=scoring,
            n_jobs=NJOBS,
            cv=cv,
            verbose=5,
            return_train_score=True)
        print('params: {}'.format(p))
        print("Scores:")
        # return score_mean
        for score_name in scoring:
            print_score(score_name, scores['train_' + score_name],
                scores['test_' + score_name])

        obj = -scores['test_neg_log_loss'].mean()
        return {'loss': obj, 'status': STATUS_OK}
    return cal_score

def optimize(score,
             trials,
             max_evals=100,
             random_state=SEED):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 300, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 10, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        # 'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                trials=trials,
                max_evals=max_evals,
                verbose=5)
    return best

#-------------------------------------------------#

def main():

    # Load processed data

    # You could use the following script to generate a well-processed train and test data sets:
    # https://www.kaggle.com/yassinealouini/predicting-red-hat-business-value/features-processing
    # I have only used the .head() of the data sets since the process takes a long time to run.
    # I have also put the act_train and act_test data sets since I don't have the processed data sets
    # loaded.

    print("Loading data...")
    tour = 89
    nrows = 1000
    nrows = None
    # Load the data from the CSV files
    training_data = data_helper.load_training_data(tour, nrows=nrows)

    # training_data = training_data.sample(frac=0.1)
    #prediction_data = data_helper.load_testing_data(tour, nrows=nrows)

    X, y = data_helper.get_Xy(training_data)

    #-------------------------------------------------#
    # Extract the train and valid (used for validation) dataframes from the train_df
    #-------------------------------------------------#

    # Run the optimization
    # Trials object where the history of search will be stored
    # For the time being, there is a bug with the following version of hyperopt.
    # You can read the error messag on the log file.
    # For the curious, you can read more about it here: https://github.com/hyperopt/hyperopt/issues/234
    # => So I am commenting it.
    trials = Trials()

    # score = make_scorer(X, y)
    score = make_cv_scorer(X, y, cv=5)
    best_hyperparams = optimize(score, trials, max_evals=50)
    print("The best hyperparameters are:")
    print(best_hyperparams)

    print('trials:')
    for t in trials.trials:
        print(t)


if __name__ == '__main__':
    main()
