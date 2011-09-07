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
    def report(self, **kwargs):
        """
        Build a text report showing the main recommender metrics implemented
        in this evaluator.

        Returns
        -------
        report : string
        Text summary of the results.

        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def evaluate(self, **kwargs):
        """
        This method is the main method that will hold the specified
        evaluation implementation for this class.

        Returns
        -------
        Returns a score representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def plot_result(self, filesave=None, **kwargs):
        """
        This method will hold the optional implementation of plotting the
        results of the recommender metric in this class.

        Parameters
        ----------
        filesave : string
        The path where will be stored the plot as figure. optional,
        default = None

        Returns
        -------
        Returns a score representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")
