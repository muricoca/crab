#-*- coding:utf-8 -*-

"""
This module contains basic implementations that encapsulate
    retrieval-related statistics about the quality of the recommender's
    recommendations.
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>

# License: BSD Style.

import numpy as np
from ..utils import check_arrays, unique_labels


def root_mean_square_error(y_real, y_pred):
    """
    It computes the root mean squared difference (RMSE)
    between predicted and actual ratings for users.

    Parameters
    ----------
    y_real : array-like

    y_pred : array-like

    Returns
    -------

    Positive floating point value: the best value is 0.0.

    return the mean square error

    """
    y_real, y_pred = check_arrays(y_real, y_pred)

    return np.sqrt((np.sum((y_pred - y_real) ** 2)) / y_real.shape[0])


def mean_absolute_error(y_real, y_pred):
    """
    It computes the average absolute difference (MAE)
    between predicted and actual ratings for users.

    Parameters
    ----------
    y_real : array-like

    y_pred : array-like

    Returns
    -------

    Positive floating point value: the best value is 0.0.

    return the mean absolute error


    """
    y_real, y_pred = check_arrays(y_real, y_pred)

    return np.sum(np.abs(y_pred - y_real)) / y_real.size


def normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating):
    """
    It computes the normalized average absolute difference (NMAE)
    between predicted and actual ratings for users.

    Parameters
    ----------
    y_real : array-like
        The real ratings.

    y_pred : array-like
        The predicted ratings.

    max_rating:
        The maximum rating of the model.

    min_rating:
        The minimum rating of the model.

    Returns
    -------

    Positive floating point value: the best value is 0.0.

    return the normalized mean absolute error


    """
    y_real, y_pred = check_arrays(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return mae / (max_rating - min_rating)


def evaluation_error(y_real, y_pred, max_rating, min_rating):
    """
    It computes the NMAE, MAE and RMSE between predicted
    and actual ratings for users.

    Parameters
    ----------
    y_real : array-like
        The real ratings.

    y_pred : array-like
        The predicted ratings.

    max_rating:
        The maximum rating of the model.

    min_rating:
        The minimum rating of the model.

    Returns
    -------
    mae: Positive floating point value: the best value is 0.0.
    nmae: Positive floating point value: the best value is 0.0.
    rmse: Positive floating point value: the best value is 0.0.

    """
    mae = mean_absolute_error(y_real, y_pred)
    nmae = normalized_mean_absolute_error(y_real, y_pred,
             max_rating, min_rating)
    rmse = root_mean_square_error(y_real, y_pred)

    return mae, nmae, rmse


def precision_score(y_real, y_pred):
    """Compute the precision

    The precision is the ratio :math:`tp / (tp + fp)` where tp is the
    number of true positives and fp the number of false positives.
    In recommendation systems the precision is the proportion of
     recommendations that are good recommendations.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    Returns
    -------
    precision : float

    """
    p, _, _ = precision_recall_fscore(y_real, y_pred)
    return np.average(p)


def recall_score(y_real, y_pred):
    """Compute the recall

    The recall is the ratio :math:`tp / (tp + fn)` where tp is the number of
    true positives and fn the number of false negatives.
    In recommendation systems the recall  is the proportion of good
    recommendations that appear in top recommendations.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    Returns
    -------
    recall : float
        ...
    """
    _, r, _ = precision_recall_fscore(y_real, y_pred)
    return np.average(r)


def f1_score(y_real, y_pred):
    """Compute f1 score

    The F1 score can be interpreted as a weighted average of the precision
    and recall, where an F1 score reaches its best value at 1 and worst
    score at 0. The relative contribution of precision and recall to the f1
    score are equal.

        F_1 = 2 * (precision * recall) / (precision + recall)

    See: http://en.wikipedia.org/wiki/F1_score

    In the recommender systems the F1-Score is considered an single value
    obtained combining both the precision and recall measures and
    indicates an overall utility of the recommendation list.


    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    Returns
    -------
    f1_score : float
        f1_score of ...

    References
    ----------
    http://en.wikipedia.org/wiki/F1_score

    """
    return fbeta_score(y_real, y_pred, 1)


def fbeta_score(y_real, y_pred, beta):
    """Compute fbeta score

    The F_beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    beta: float
        The beta parameter determines the weight of precision in the combined
        score. beta < 1 lends more weight to precision, while beta > 1 favors
        precision (beta == 0 considers only precision, beta == inf only
        recall).

    Returns
    -------
    fbeta_score : float
        fbeta_score of ...

    See also
    --------
    R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern Information Retrieval.
    Addison Wesley, pp. 327-328.

    http://en.wikipedia.org/wiki/F1_score

    """
    _, _, f = precision_recall_fscore(y_real, y_pred, beta=beta)

    return np.average(f)


def precision_recall_fscore(y_real, y_pred, beta=1.0):
    """Compute precisions, recalls, f-measures
       for recommender systems


    The precision is the ratio :math:`tp / (tp + fp)` where tp is the number of
    true positives and fp the number of false positives. In recommender systems
    ...

    The recall is the ratio :math:`tp / (tp + fn)` where tp is the number of
    true positives and fn the number of false negatives. In recommender
    systems...

    The F_beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F_beta score reaches its best
    value at 1 and worst score at 0.

    The F_beta score weights recall beta as much as precision. beta = 1.0 means
    recall and precision are as important.

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true recommended items

    y_pred : array, shape = [n_samples]
        predicted recommended items

    beta : float, 1.0 by default
        the strength of recall versus precision in the f-score

    Returns
    -------
    precision: array, shape = [n_unique_labels], dtype = np.double
    recall: array, shape = [n_unique_labels], dtype = np.double
    f1_score: array, shape = [n_unique_labels], dtype = np.double

    References
    ----------
    http://en.wikipedia.org/wiki/Precision_and_recall

    """
    y_real, y_pred = check_arrays(y_real, y_pred)
    assert(beta > 0)

    n_users = y_real.shape[0]
    precision = np.zeros(n_users, dtype=np.double)
    recall = np.zeros(n_users, dtype=np.double)
    fscore = np.zeros(n_users, dtype=np.double)

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        for i, y_items_pred in enumerate(y_pred):
            intersection_size = np.intersect1d(y_items_pred, y_real[i]).size
            precision[i] = (intersection_size / float(len(y_real[i]))) \
                                    if len(y_real[i])  else 0.0
            recall[i] = (intersection_size / float(len(y_items_pred))) \
                                    if len(y_items_pred) else 0.0

        # handle division by 0.0 in precision and recall
        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(precision)] = 0.0

        #fbeta Score
        beta2 = beta ** 2
        fscore = (1 + beta2) * (precision * recall) \
                    / (beta2 * precision + recall)

        #handle division by 0.0 in fscore
        fscore[(precision + recall) == 0.0] = 0.0

    finally:
        np.seterr(**old_err_settings)

    return precision, recall, fscore


def evaluation_report(y_real, y_pred, labels=None, target_names=None):
    """Build a text report showing the main recommender metrics

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        estimated targets

    labels : array, shape = [n_labels]
        optional list of label indices to include in the report

    target_names : list of strings
        optional display names matching the labels (same order)

    Returns
    -------
    report : string
        Text summary of the precision, recall, f1-score.

    """

    if labels is None:
        labels = unique_labels(y_real)
    else:
        labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'
    p, r, f1 = precision_recall_fscore(y_real, y_pred)
    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p),
              np.average(r),
              np.average(f1)):
        values += ["%0.2f" % float(v)]
    report += fmt % tuple(values)
    return report
