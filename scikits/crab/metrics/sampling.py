"""Utilities for sampling techniques"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
# License: BSD Style.

import numpy as np
from ..utils import check_random_state
from math import ceil


class SplitSampling(object):
    """ Random Split Sampling the dataset into two sets.

    Parameters
    ----------
    n : int
        Total number of elements in the dataset.

    evaluation_fraction : float (default 0.7)
        Should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the training set.

    indices : boolean, optional (default False)
        Return  split with integer indices or boolean mask.
        Integer indices are useful when dealing with sparse matrices
        that cannot be indexed by boolean masks.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Examples
    ----------

    """
    def __init__(self, n, evaluation_fraction=0.7, indices=False,
            random_state=None):
        self.n = n
        self.evaluation_fraction = evaluation_fraction
        self.random_state = random_state
        self.indices = indices

    def split(self, evaluation_fraction=0.7, indices=False,
             random_state=None):
        """
        Random Split Sampling the dataset into two sets.

        Parameters
        ----------
        evaluation_fraction : float (default 0.1)
            Should be between 0.0 and 1.0 and represent the proportion of
            the dataset to include in the training set.

        indices : boolean, optional (default False)
            Return  split with integer indices or boolean mask.
            Integer indices are useful when dealing with sparse matrices
            that cannot be indexed by boolean masks.

        random_state : int or RandomState
            Pseudo-random number generator state used for random sampling.

        """
        self.evaluation_fraction = evaluation_fraction
        self.random_state = random_state
        self.indices = indices

        rng = self.random_state = check_random_state(self.random_state)
        n_train = ceil(self.evaluation_fraction * self.n)
        #random partition
        permutation = rng.permutation(self.n)
        ind_train = permutation[-n_train:]
        ind_ignore = permutation[:-n_train]
        if self.indices:
            return ind_train, ind_ignore
        else:
            train_mask = np.zeros(self.n, dtype=np.bool)
            train_mask[ind_train] = True
            test_mask = np.zeros(self.n, dtype=np.bool)
            test_mask[ind_ignore] = True
            return train_mask, test_mask

    def __repr__(self):
        return ('%s(%d, evaluation_fraction=%s, indices=%s, '
                'random_state=%d)' % (
                    self.__class__.__name__,
                    self.n,
                    str(self.evaluation_fraction),
                    self.indices,
                    self.random_state,
                ))
