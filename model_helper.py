#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from sklearn import metrics

def eval_predict(y_true, y_prob, thr=0.5, indent='    '):
    print('{}probability predicted: min {}, max {}' \
            .format(indent, np.min(y_prob), np.max(y_prob)))
    y_pred = y_prob > thr
    ll = metrics.log_loss(y_true, y_prob)
    print('{}log loss: {:.7f}'.format(indent, ll))
    auc = metrics.roc_auc_score(y_true, y_prob)
    print('{}AUC: {:.7f}'.format(indent, auc))
    acc = metrics.accuracy_score(y_true, y_pred)
    print('{}accuracy: {:.8f}'.format(indent, acc))
    return (ll, auc, acc)

def print_perfs(models, perfs):
    def print_perf(perf):
        perf_cv, perf_train, perf_test = perf
        print('performance on {}-fold cross validation:'.format(len(perf_cv[0])))
        print_perf_cv(perf_cv)
        print('performance on whole trainning set:')
        print_metrics(perf_train)
        print('performance on testing set:')
        print_metrics(perf_test)

    def print_metrics(met):
        ll, auc, acc = met
        print('\tlogloss: {:.7f}, AUC: {:.7f}, Accuracy: {:.7f}'.format(ll, auc, acc))

    def print_perf_cv(perf_cv):
        #print(perf_cv)
        perf_train, perf_test = perf_cv
        ll_train, auc_train, acc_train = zip(*perf_train)
        ll_test, auc_test, acc_test = zip(*perf_test)
        print('  logloss:')
        print_cv_met(ll_train, 'train')
        print_cv_met(ll_test, 'test')
        print('  AUC:')
        print_cv_met(auc_train, 'train')
        print_cv_met(auc_test, 'test')
        print('  accuracy:')
        print_cv_met(acc_train, 'train')
        print_cv_met(acc_test, 'test')

    def print_cv_met(met, typ):
        #print('\t{}:\tmean {}, std {}, detail: {!s:s}'.format(typ, np.mean(met), np.std(met), met))
        print('\t{}:\tmean {:.7f}, std {:.7f}, detail: {!s:s}'.format(typ, np.mean(met), np.std(met), met))

    #for entry in zip(models, perfs):
    #for i, (name1, perf) in enumerate(perfs):
    for (name, model), (name1, perf) in zip(models, perfs):
        #print(entry)
        #name, model = models[i]
        #print(name, name1)
        assert(name == name1)
        print('====== Performance on model "{}" ======='.format(name))
        print('model details: {!s:s}'.format(model))
        print_perf(perf)


def main():
    return

if __name__ == '__main__':
    main()
