#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
try:
    from tensorflow.python.framework import dtypes
except ImportError:
    pass

# modified from https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L189
class Batcher(object):

    def __init__(self,
               features,
               labels,
               dtype=dtypes.float32):

        """Construct a DataSet.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)

        assert features.shape[0] == labels.shape[0], (
          'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
        self._num_examples = features.shape[0]

        # # Convert shape from [num examples, rows, columns, depth]
        # # to [num examples, rows*columns] (assuming depth == 1)
        # if reshape:
        #     assert features.shape[3] == 1
        #     features = features.reshape(features.shape[0],
        #                         features.shape[1] * features.shape[2])

        # if dtype == dtypes.float32:
        #     # Convert from [0, 255] -> [0.0, 1.0].
        #     features = features.astype(np.float32)
        #     features = np.multiply(features, 1.0 / 255.0)

        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        # print('index({})/total({}), batch size: {}' \
        #     .format(self._index_in_epoch, self._num_examples, batch_size))
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            print('finished epoch {}'.format(self._epochs_completed))
            # print('shuffle:')
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]

class Cutter(object):

    def __init__(self, method='cut', nbins=10, cutcols=None,
            expand_left=-np.inf, expand_right=np.inf):
        self._method = method
        self._nbins = nbins
        self._cutcols = cutcols
        self._expand_left = expand_left
        self._expand_right = expand_right

        self._bins_list = []

        if self._method == 'cut':
            self._cut_func = pd.cut
        elif self._method == 'qcut':
            self._cut_func = pd.qcut
        else:
            raise NotImplementedError()

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        self._ndim = len(X.shape)
        if len(X.shape) == 1:
            #self._n_features = 0
            #val, bins = self.cut_1d(0, X, 0)
            #self._levels = [val.categories.values]
            #self._bins_list.append(bins)
            #return val.codes
            X = X.reshape((X.shape[0], 1))
        elif len(X.shape) > 2:
            raise NotImplementedError()

        self._n_features = X.shape[1]
        cutcols = self._get_cutcols()

        ncols = len(cutcols)
        cutted_list = [0] * ncols
        self._bins_list = [0] * ncols
        for i, icol in enumerate(cutcols):
            val, bins = self._cut_1d(i, X[:, icol], ncols)
            cutted_list[i] = val
            self._bins_list[i] = bins
        self._levels = [c.categories.values for c in cutted_list]
        self._expand_levels_bins()
        ret = np.column_stack([c.codes for c in cutted_list])
        return ret

    def _get_cutcols(self):
        if self._cutcols is None:
            cutcols = range(self._n_features)
        else:
            cutcols = self._cutcols
        return cutcols

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

    def _expand_levels_bins(self):
        for lev, bins in zip(self._levels, self._bins_list):
            if self._expand_left is not None:
                lev[0] = pd.Interval(left=self._expand_left,
                        right=lev[0].right,
                        closed=lev[0].closed)
                bins[0] = self._expand_left
            if self._expand_right is not None:
                lev[-1] = pd.Interval(left=lev[-1].left,
                        right=self._expand_right,
                        closed=lev[-1].closed)
                bins[-1] = self._expand_right

    def transform(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        if self._n_features == 0:
            if len(X.shape) != 1:
                raise ValueError('X must be 1d vector')
            return self._trans_1d(-1, X)
        else:
            if len(X.shape) != 2:
                raise NotImplementedError()

        cutcols = self._get_cutcols()
        ncols = len(cutcols)
        cutted_list = [0] * ncols
        for i, icol in enumerate(cutcols):
            val = self._trans_1d(i, X[:, icol])
            cutted_list[i] = val
        ret = np.column_stack([c for c in cutted_list])
        return ret

    def _trans_1d(self, i, v):
        ret = pd.cut(v, bins=self._bins_list[i], labels=False, include_lowest=True)
        return ret

def test_cutter():
    import numpy.random as r

    x = r.randn(10, 2)
    cutter = Cutter(nbins=[3, 4])
    print(x)
    xc = cutter.fit_transform(x)
    print(xc)
    print('levels:', cutter._levels)
    print(cutter._bins_list)
    xc2 = cutter.transform(x)
    print(xc2)
    assert(np.all(xc == xc2))

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
