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

    def evaluate(self, recommender, metrics=None, **kwargs):
        """
        Evaluates the predictor

        Parameters
        ----------

        recommender: The BaseRecommender instance
                The recommender instance to be evaluated.

        metrics: [None|'rmse'|'f1score'|'precision'|'recall'|'nmae'|'mae']
        If metrics is None, all metrics available will be evaluated.
        Otherwise it will return the specified metric evaluated.

        Returns
        -------
        Returns scores representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate_online(self, metrics=None, **kwargs):
        """
        Online evaluation for recommendation prediction

        Parameters
        ----------
        metrics: [None|'rmse'|'f1score'|'precision'|'recall'|'nmae'|'mae']
        If metrics is None, all metrics available will be evaluated.
        Otherwise it will return the specified metric evaluated.

        Returns
        -------

        Returns scores representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate_on_split(self, metrics=None, **kwargs):
        """
        Evaluate on the folds of a dataset split

        Parameters
        ----------
        metrics: [None|'rmse'|'f1score'|'precision'|'recall'|'nmae'|'mae']
        If metrics is None, all metrics available will be evaluated.
        Otherwise it will return the specified metric evaluated.

        Returns
        -------
        Returns scores representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")
