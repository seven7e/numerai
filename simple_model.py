#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, RepeatedKFold

data_dir = '/home/stone/data/numerai/tour88'
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

    X = training_data[features].as_matrix()
    y = training_data["target"].as_matrix()

    # This is your model that will learn to predict
    #model = linear_model.LogisticRegression(n_jobs=-1)
    model = RandomForestClassifier(
            max_depth=None,
            #max_depth=5,
            n_estimators=10, max_features=1,
            min_samples_split=100,
            min_samples_leaf=1)

    model = AdaBoostClassifier()

    build_model(model, X, y)

def build_model(model, X, y):
    print('Using model: {}'.format(model))

    num_samples = X.shape[0]

    random_state = 12883823
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
    scores_train = []
    scores_test = []
    for iter_num, (train_index, test_index) in enumerate(rkf.split(X, y)):
        print('-------------- iteration {} -------------'.format(iter_num))
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
        model.fit(X_train, y_train)

        score = eval_model(model, X_train, y_train, X_test, y_test)
        scores_train.append(score[0])
        scores_test.append(score[1])

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]
    y_prediction = model.predict_proba(x_prediction)
    results = y_prediction[:, 1]

def eval_model(model, X_train, y_train, X_test, y_test):
    print('Evaluating model...')
    print('train set:')
    score_train = eval_one(model, X_train, y_train)
    print('test set:')
    score_test = eval_one(model, X_test, y_test)
    return score_train, score_test

def eval_one(model, X, y):
    prob = model.predict_proba(X)
    prob_pos = prob[:, 1]
    y_pred = prob_pos > thr
    ll = metrics.log_loss(y, prob)
    print('log loss: {}'.format(ll))
    auc = metrics.roc_auc_score(y, prob_pos)
    print('AUC: {}'.format(auc))
    acc = metrics.accuracy_score(y, y_pred)
    print('accuracy: {}'.format(acc))
    return (ll, auc, acc)

def write_result(ids, results):

    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(ids).join(results_df)

    #print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    #joined.to_csv("predictions.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
