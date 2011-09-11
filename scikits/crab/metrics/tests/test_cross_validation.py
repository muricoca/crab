import numpy as np
from ..cross_validation import LeaveOneOut, LeavePOut
from numpy.testing import assert_array_equal


def test_LeaveOneOut():
    X = np.array(['userA', 'userB', 'userC', 'userD'])
    loo = LeaveOneOut(4)
    results_train = [['userB', 'userC', 'userD'], ['userA', 'userC', 'userD'],
                ['userA', 'userB', 'userD'], ['userA', 'userB', 'userC']]
    results_test = [['userA'], ['userB'], ['userC'], ['userD']]
    for index, sample in enumerate(loo):
        assert_array_equal(X[sample[0]], results_train[index])
        assert_array_equal(X[sample[1]], results_test[index])

    loo = LeaveOneOut(4, True)
    for index, sample in enumerate(loo):
        assert_array_equal(X[sample[0]], results_train[index])
        assert_array_equal(X[sample[1]], results_test[index])


def test_LeavePOut():
    X = np.array(['userA', 'userB', 'userC'])
    loo = LeavePOut(3, 2)
    results_train = [['userC'], ['userB'], ['userA']]
    results_test = [['userA', 'userB'], ['userA', 'userC'],
                    ['userB', 'userC']]
    for index, sample in enumerate(loo):
        assert_array_equal(X[sample[0]], results_train[index])
        assert_array_equal(X[sample[1]], results_test[index])

    loo = LeavePOut(3, 2, True)
    for index, sample in enumerate(loo):
        assert_array_equal(X[sample[0]], results_train[index])
        assert_array_equal(X[sample[1]], results_test[index])
