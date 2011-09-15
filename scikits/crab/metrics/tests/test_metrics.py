import numpy as np
from nose.tools import assert_equals, assert_almost_equals
from ..metrics import root_mean_square_error, mean_absolute_error,\
                        normalized_mean_absolute_error, precision_recall_fscore, \
                        precision_score, recall_score, f1_score, evaluation_report,\
                        evaluation_error
from numpy.testing import assert_array_almost_equal


def test_root_mean_square_error():
    """Check that the metric Root Mean Squared Error (RMSE) """
    y_real = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    assert_equals(0.0, root_mean_square_error(y_real, y_pred))

    y_real = np.array([3.0, 1.0, 2.0, 1.0, 1.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    assert_almost_equals(1.8973665961, root_mean_square_error(y_real, y_pred))


def test_root_mean_absolute_error():
    """Check that the metric Mean Absolute Error (MAE) """
    y_real = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    assert_equals(0.0, mean_absolute_error(y_real, y_pred))

    y_real = np.array([3.0, 1.0, 2.0, 1.0, 1.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    assert_almost_equals(1.6, mean_absolute_error(y_real, y_pred))


def test_root_normalized_mean_absolute_error():
    """Check that the metric Normalized Mean Absolute Error (NMAE) """
    max_rating = 5.0
    min_rating = 1.0
    y_real = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    assert_equals(0.0, normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating))

    y_real = np.array([3.0, 1.0, 2.0, 1.0, 1.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    assert_almost_equals(0.4, normalized_mean_absolute_error(y_real, y_pred,
                max_rating, min_rating))


def test_evaluation_error():
    """ Check the error evaluation """
    max_rating = 5.0
    min_rating = 1.0
    y_real = np.array([0.0, 1.0, 0.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 0.0, 2.0, 3.0])

    mae, nmae, rmse = evaluation_error(y_real, y_pred, max_rating, min_rating)
    assert_equals(mae, 0.0)
    assert_equals(nmae, 0.0)
    assert_equals(rmse, 0.0)


def test_precision_recall_f1_score():
    """Test Precision Recall and F1 Score """
    y_real = np.array([['a', 'b', 'c'], ['a', 'b', 'e', 'f', 'g'], ['a', 'b']])
    y_pred = np.array([['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['e', 'f']])

    p, r, f = precision_recall_fscore(y_real, y_pred)
    assert_array_almost_equal(p, [1, 0.4, 0], 2)
    assert_array_almost_equal(r, [1., 0.5, 0], 2)
    assert_array_almost_equal(f, [1., 0.44, 0], 2)

    ps = precision_score(y_real, y_pred)
    assert_array_almost_equal(ps, 0.4666, 2)

    rs = recall_score(y_real, y_pred)
    assert_array_almost_equal(rs, 0.5, 2)

    fs = f1_score(y_real, y_pred)
    assert_array_almost_equal(fs, 0.48, 2)


def test_zero_precision_recall():
    """Check that pathological cases do not bring NaNs"""

    try:
        old_error_settings = np.seterr(all='raise')

        y_real = np.array([['a', 'b', 'c']])
        y_pred = np.array([[]])

        assert_array_almost_equal(precision_score(y_real, y_pred), 0.0, 2)
        assert_array_almost_equal(recall_score(y_real, y_pred), 0.0, 2)
        assert_array_almost_equal(f1_score(y_real, y_pred), 0.0, 2)

    finally:
        np.seterr(**old_error_settings)


def test_evaluation_report():
    """Test evaluation report"""
    y_real = np.array([['a', 'b', 'c'], ['a', 'b', 'e', 'f', 'g'], ['a', 'b']])
    y_pred = np.array([['a', 'b', 'c'], ['a', 'b', 'c', 'd'], ['e', 'f']])
    labels = np.array(['user_id1', 'user_id2', 'user_id3'])
    # print evaluation report with class names
    expected_report = """\
             precision    recall  f1-score

   user_id1       1.00      1.00      1.00
   user_id2       0.40      0.50      0.44
   user_id3       0.00      0.00      0.00

avg / total       0.47      0.50      0.48
"""
    report = evaluation_report(
        y_real, y_pred,
        target_names=labels)
    assert_equals(report, expected_report)
    # print classification report with label detection
    expected_report = """\
             precision    recall  f1-score

          0       1.00      1.00      1.00
          1       0.40      0.50      0.44
          2       0.00      0.00      0.00

avg / total       0.47      0.50      0.48
"""
    report = evaluation_report(y_real, y_pred)
    assert_equals(report, expected_report)
