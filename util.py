#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

class Cutter(object):

    def __init__(self, method='cut', nbins=10, cutcols=None):
        self._method = method
        self._nbins = nbins
        self._cutcols = cutcols
        self._bins_list = []

        if self._method == 'cut':
            self._cut_func = pd.cut
        elif self._method == 'qcut':
            self._cut_func = pd.qcut
        else:
            raise NotImplementedError()

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        if len(X.shape) == 1:
            self._n_features = 0
            val, bins = self.cut_1d(X)
            self._bins_list.append(bins)
            return val
        elif len(X.shape) > 2:
            raise NotImplementedError()

        self._n_features = X.shape[1]
        if self._cutcols is None:
            cutcols = range(self._n_features)
        else:
            cutcols = self._cutcols

        ncols = len(cutcols)
        cutted_list = [0] * ncols
        self._bins_list = [0] * ncols
        for i, icol in enumerate(cutcols):
            val, bins = self._cut_1d(i, X[:, icol], ncols)
            cutted_list[i] = val
            self._bins_list[i] = bins
        self._levels = [c.categories.values for c in cutted_list]
        ret = np.column_stack([c.codes for c in cutted_list])
        return ret

    def _cut_1d(self, i, v, ncols):
        if isinstance(self._nbins, (list, tuple)):
            if len(self._nbins) != ncols:
                raise ValueError('nbins must be scalar or list with same size' +
                    'to number of columns of input data')
            nbins = self._nbins[i]
        else:
            nbins = self._nbins

        val, bins = self._cut_func(v, nbins, retbins=True)
        return val, bins

def test_cutter():
    import numpy.random as r

    x = r.randn(10, 2)
    cutter = Cutter(nbins=[2, 3])
    print(x)
    xc = cutter.fit(x)
    print(xc)
    print(cutter._levels)
    print(cutter._bins_list)

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    dummy = enc.fit_transform(xc)
    print('dummy:', dummy.toarray())
    #print('dummy:', type(dummy))


def main():
    return

if __name__ == '__main__':
    #main()
    test_cutter()
