#-*- coding:utf-8 -*-

"""Utilities to evaluate the predictive performance of the recommenders
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>

# License: BSD Style.


class RecommenderEvaluator(object):
    """
    Basic Interface which is responsible to evaluate the quality of Recommender
    recommendations. The range of values that may be returned depends on the
    implementation. but lower values must mean better recommendations, with 0
    being the lowest / best possible evaluation, meaning a perfect match.

    """

    def evaluate(self, **kwargs):
        """
        Evaluates the predictor

        Returns
        -------
        Returns a score representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate_online(self, **kwargs):
        """
        Online evaluation for recommendation prediction

        Returns

        -------
        Returns a score representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate_on_split(self, **kwargs):
        """
        Evaluate on the folds of a dataset split

        Returns
        -------
        Returns a score representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")
