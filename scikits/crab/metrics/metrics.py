#-*- coding:utf-8 -*-

"""
This module contains basic implementations that encapsulate
    retrieval-related statistics about the quality of the recommender's
    recommendations.
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>

# License: BSD Style.

import numpy as np
from ..utils import check_arrays


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


def precision_score(y_real, y_pred, n_at):
    """Compute the precision

    The precision is the ratio :math:`tp / (tp + fp)` where tp is the
    number of true positives and fp the number of false positives.
    In recommendation systems the precision ...

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    n_at: int
        It is the 'at' value, as in 'precision at 5'. In this
        example would mean precision by removing the top 5
        preferences for a user and then finding the percentage
        of those 5 items included in the top 5 recommendations
        for that user.

    Returns
    -------
    precision : float

    """
    pass


def recall_score(y_real, y_pred, n_at):
    """Compute the recall

    The recall is the ratio :math:`tp / (tp + fn)` where tp is the number of
    true positives and fn the number of false negatives.
    In recommendation systems the precision ...

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    n_at: int
        It is the 'at' value, as in 'recall at 5'. In this
        example would mean recall by removing the top 5
        preferences for a user and then finding the percentage
        of those 5 items included in the top 5 recommendations
        for that user...

    Returns
    -------
    recall : float
        ...
    """
    pass


def f1_score(y_real, y_pred, n_at):
    """Compute f1 score

    The F1 score can be interpreted as a weighted average of the precision
    and recall, where an F1 score reaches its best value at 1 and worst
    score at 0. The relative contribution of precision and recall to the f1
    score are equal.

        F_1 = 2 * (precision * recall) / (precision + recall)

    See: http://en.wikipedia.org/wiki/F1_score

    In the recommender systems ...

    Parameters
    ----------
    y_real : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    n_at: int
        It is the 'at' value, as in 'recall at 5'. In this
        example would mean recall by removing the top 5
        preferences for a user and then finding the percentage
        of those 5 items included in the top 5 recommendations
        for that user...

    Returns
    -------
    f1_score : float
        f1_score of ...

    References
    ----------
    http://en.wikipedia.org/wiki/F1_score

    """
    return fbeta_score(y_real, y_pred, n_at, 1)


def fbeta_score(y_real, y_pred, beta, n_at):
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

    n_at: int
        It is the 'at' value, as in 'recall at 5'. In this
        example would mean recall by removing the top 5
        preferences for a user and then finding the percentage
        of those 5 items included in the top 5 recommendations
        for that user...

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
    pass


def precision_recall_fscore_support(y_real, y_pred, n_at, beta=1.0):
    """Compute precisions, recalls, f-measures and support
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
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    n_at: int
        It is the 'at' value, as in 'recall at 5'. In this
        example would mean recall by removing the top 5
        preferences for a user and then finding the percentage
        of those 5 items included in the top 5 recommendations
        for that user...

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
    pass
