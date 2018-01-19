#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing, linear_model
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import data_helper
import model_helper

def modelfit(alg, X, y,
        useTrainCV=True,
        cv_folds=5,
        eval_metric=['logloss', 'auc', 'error'],
        early_stopping_rounds=5):

    if useTrainCV:
        print('cross validating xgboost model with training data')
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain,
                num_boost_round=alg.get_params()['n_estimators'],
                nfold=cv_folds,
                #metrics='auc',
                metrics=eval_metric,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True
                )
        print('cvresult:', cvresult)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    print('fitting model: {}'.format(alg))
    alg.fit(X, y, eval_metric=eval_metric)

    #Predict training set:
    #dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]

    #Print model report:
    print("\nModel Report")
    model_helper.eval_predict(y, dtrain_predprob)
    #print "Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions)
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob)
    #if performCV:
    #    print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
    #            % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    xgb.plot_importance(alg)
    plt.show()

def tune1(xgb1, X, y):
    param_test1 = {
            'max_depth': range(3,10,2),
            'min_child_weight': range(1,6,2)
            }
    gsearch1 = GridSearchCV(estimator =
            xgb.XGBClassifier(learning_rate =0.1,
                # TODO: clone xgb1?
                #n_estimators=140,
                n_estimators=xgb1.n_estimators,
                max_depth=5,
                min_child_weight=1,
                gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27),
           param_grid = param_test1,
           #scoring='roc_auc',
           scoring='neg_log_loss',
           n_jobs=4,
           iid=False,
           verbose=5,
           cv=5)
    gsearch1.fit(X, y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    tour = 90
    nrows = None
    # Load the data from the CSV files
    training_data = data_helper.load_training_data(tour, nrows=nrows)
    #training_data = pd.read_csv(data_dir + '/numerai_training_data.csv', header=0)

    #prediction_data = pd.read_csv(data_dir + '/numerai_tournament_data.csv', header=0)
    #prediction_data = data_helper.load_testing_data(tour, nrows=nrows)

    features = data_helper.get_feature_names(training_data)

    # Transform the loaded CSV data into numpy arrays

    frac = 0.1
    frac = 0.01
    frac = 0.0001
    frac = None
    if frac is not None:
        training_data = training_data.sample(frac=frac)

    X, y = data_helper.get_Xy(training_data)

    xgb1 = xgb.XGBClassifier(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27)
    #modelfit(xgb1, X, y)
    tune1(xgb1, X, y)

if __name__ == '__main__':
    main()
