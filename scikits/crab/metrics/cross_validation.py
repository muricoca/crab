"""Utilities for cross validation and performance evaluation"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
# License: BSD Style.

import numpy as np
from ..utils.extmath import factorial, combinations
from ..utils import check_random_state
from math import ceil


class LeaveOneOut(object):
    """Leave-One-Out cross validation iterator.

    Provides train/test indices to split user preferences in train
    and test sets. Each sample is used once as a test set (singleton)
    while the remaining samples form the training set.

    Due to the high number of test sets (which is the same as the
    number of samples) this cross validation method can be very costly.
    For large datasets one should favor KFold or ShuffleSplit.

    Parameters
    ==========
    n: int
        Total number of user preferences

    indices: boolean, optional (default False)
        Return train/test split with integer indices or boolean mask.
        Integer indices are useful when dealing with sparse matrices
        that cannot be indexed by boolean masks.

    Examples
    ========
    >>> from scikits.crab.metrics import LeaveOneOut
    >>> X = np.array(['userA', 'userB', 'userC'])
    >>> loo = LeaveOneOut(3)
    >>> len(loo)
    3
    >>> print loo
    scikits.crab.metrics.cross_validation.LeaveOneOut(n=3)
    >>> for train_index, test_index in loo:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    print X_train, X_test
    TRAIN: [False  True  True] TEST: [ True False False]
    ['userB' 'userC'] ['userA']
    TRAIN: [ True False  True] TEST: [False  True False]
    ['userA' 'userC'] ['userB']
    TRAIN: [ True  True False] TEST: [False False  True]
    ['userA' 'userB'] ['userC']

    """
    def __init__(self, n, indices=False):
        self.n = n
        self.indices = indices

    def __iter__(self):
        n = self.n
        for i in xrange(n):
            test_index = np.zeros(n, dtype=np.bool)
            test_index[i] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
        )

    def __len__(self):
        return self.n


class LeavePOut(object):
    """Leave-P-Out cross validation iterator

    Provides train/test indices to split user preferences in train test sets.
    The test set is built using p samples while the remaining samples form
    the training set.

    Due to the high number of iterations which grows with the number of
    samples this cross validation method can be very costly. For large
    datasets one should favor KFold or ShuffleSplit.

    Parameters
    ===========
    n: int
        Total number of user_profiles

    p: int
        Size of the test sets

    indices: boolean, optional (default False)
        Return train/test split with integer indices or boolean mask.
        Integer indices are useful when dealing with sparse matrices
        that cannot be indexed by boolean masks.

    Examples
    ========
    >>> from scikits.crab.metrics import LeavePOut
    >>> X = np.array(['userA', 'userB', 'userC'])
    >>> lpo = LeavePOut(3, 2)
    >>> len(lpo)
    3
    >>> print lpo
    scikits.crab.metrics.cross_validation.LeavePOut(n=3, p=2)
    >>> for train_index, test_index in lpo:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...    X_train, X_test = X[train_index], X[test_index]
    TRAIN: [False False  True] TEST: [ True  True False]
    TRAIN: [False  True False] TEST: [ True False  True]
    TRAIN: [ True False False] TEST: [False  True  True]

    """
    def __init__(self, n, p, indices=False):
        self.n = n
        self.p = p
        self.indices = indices

    def __iter__(self):
        n = self.n
        p = self.p
        comb = combinations(range(n), p)
        for idx in comb:
            test_index = np.zeros(n, dtype=np.bool)
            test_index[np.array(idx)] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, p=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.p,
        )

    def __len__(self):
        return (factorial(self.n) / factorial(self.n - self.p)
                / factorial(self.p))


class KFold(object):
    """K-Folds cross validation iterator

    Provides train/test indices to split user preferences in train test sets.
    Split dataset into k consecutive folds (without shuffling).

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    n: int
        Total number of user preferences

    k: int
        Number of folds

    indices: boolean, optional (default False)
        Return train/test split with integer indices or boolean mask.
        Integer indices are useful when dealing with sparse matrices
        that cannot be indexed by boolean masks.

    Examples
    --------
    >>> from scikits.crab.metrics import KFold
    >>> X = np.array(['userA', 'userB', 'userC', 'userD'])
    >>> kf = KFold(4, k=2)
    >>> len(kf)
    2
    >>> print kf
    scikits.crab.metrics.cross_validation.KFold(n=4, k=2)
    >>> for train_index, test_index in kf:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...    X_train, X_test = X[train_index], X[test_index]
    TRAIN: [False False  True  True] TEST: [ True  True False False]
    TRAIN: [ True  True False False] TEST: [False False  True  True]

    Notes
    -----
    All the folds have size trunc(n_samples / n_folds), the last one has the
    complementary.

    """
    def __init__(self, n, k, indices=False):
        assert k > 0, ValueError('Cannot have number of folds k below 1.')
        assert k <= n, ValueError('Cannot have number of folds k=%d, '
                                  'greater than the number '
                                  'of samples: %d.' % (k, n))
        self.n = n
        self.k = k
        self.indices = indices

    def __iter__(self):
        n = self.n
        k = self.k
        j = ceil(n / k)

        for i in xrange(k):
            test_index = np.zeros(n, dtype=np.bool)
            if i < k - 1:
                test_index[i * j:(i + 1) * j] = True
            else:
                test_index[i * j:] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                ind = np.arange(n)
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, k=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.k,
        )

    def __len__(self):
        return self.k


class ShuffleSplit(object):
    """Random permutation cross-validation iterator.

    Yields indices to split user preferences into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    n : int
        Total number of elements in the dataset.

    n_iterations : int (default 10)
        Number of re-shuffling & splitting iterations.

    test_fraction : float (default 0.1)
        Should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split.

    indices : boolean, optional (default False)
        Return train/test split with integer indices or boolean mask.
        Integer indices are useful when dealing with sparse matrices
        that cannot be indexed by boolean masks.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Examples
    ----------
    >>> from scikits.crab.metrics import ShuffleSplit
    >>> rs = ShuffleSplit(4, n_iterations=3, test_fraction=.25,
    ...                             random_state=0)
    >>> len(rs)
    3
    >>> print rs
    ... # doctest: +ELLIPSIS
    ShuffleSplit(4, n_iterations=3, test_fraction=0.25, indices=False, ...)
    >>> for train_index, test_index in rs:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...
    TRAIN: [False  True  True  True] TEST: [ True False False False]
    TRAIN: [ True  True  True False] TEST: [False False False  True]
    TRAIN: [ True False  True  True] TEST: [False  True False False]
    """
    def __init__(self, n, n_iterations=10, test_fraction=0.1,
                indices=False, random_state=None):
        self.n = n
        self.n_iterations = n_iterations
        self.test_fraction = test_fraction
        self.random_state = random_state
        self.indices = indices

    def __iter__(self):
        rng = self.random_state = check_random_state(self.random_state)
        n_test = ceil(self.test_fraction * self.n)
        for i in range(self.n_iterations):
            #random partition
            permutation = rng.permutation(self.n)
            ind_train = permutation[:-n_test]
            ind_test = permutation[-n_test:]
            if self.indices:
                yield ind_train, ind_test
            else:
                train_mask = np.zeros(self.n, dtype=np.bool)
                train_mask[ind_train] = True
                test_mask = np.zeros(self.n, dtype=np.bool)
                test_mask[ind_test] = True
                yield train_mask, test_mask

    def __repr__(self):
        return ('%s(%d, n_iterations=%d, test_fraction=%s, indices=%s, '
                'random_state=%d)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iterations,
                    str(self.test_fraction),
                    self.indices,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iterations
