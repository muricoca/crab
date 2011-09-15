 #-*- coding:utf-8 -*-

"""
This module contains main implementations that encapsulate
    retrieval-related statistics about the quality of the recommender's
    recommendations.

"""
# Authors: Marcel Caraciolo <marcel@muricoca.com>
# License: BSD Style.


from base import RecommenderEvaluator
from metrics import root_mean_square_error
from metrics import mean_absolute_error
from metrics import normalized_mean_absolute_error
from metrics import precision_score
from metrics import recall_score
from metrics import f1_score


#Collaborative Filtering Evaluator
#==================================

evaluation_metrics = {
        'rmse': root_mean_square_error,
        'mae': mean_absolute_error,
        'nmae': normalized_mean_absolute_error,
        'precision': precision_score,
        'recall': recall_score,
        'f1score': f1_score
}


def check_sampling(sampling, n):
    """Input checker utility for building a
       sampling in a user friendly way.

   Parameters
   ===========
    sampling: a float, a sampling generator instance, or None
        The input specifying which sampling generator to use.
        It can be an float, in which case it is the the proportion of
        the dataset to include in the training set in SplitSampling.
        None, in which case all the elements are used,
        or another object, that will then be used as a cv generator.

    n: an integer.
        The number of elements.

    """
    if sampling is None:
        sampling = 1.0
    if operator.isNumberType(sampling):
        sampling = SplitSampling(n, evaluation_fraction=sampling)

    return sampling


class CfEvaluator(RecommenderEvaluator):
    def evaluate(self, recommender, metric=None, **kwargs):
        """
        Evaluates the predictor

        Parameters
        ----------
        recommender: The BaseRecommender instance
                The recommender instance to be evaluated.

        metric: [None|'rmse'|'f1score'|'precision'|'recall'|'nmae'|'mae']
        If metrics is None, all metrics available will be evaluated.
        Otherwise it will return the specified metric evaluated.

        sampling_users:  float or sampling, optional, default = None
        If an float is passed, it is the percentage of evaluated
        users. If sampling_users is None, all users are used in the
        evaluation. Specific sampling objects can be passed, see
        scikits.crab.metrics.sampling module for the list of possible
        objects.

        sampling_ratings:  float or sampling, optional, default = None
        If an float is passed, it is the percentage of evaluated
        ratings. If sampling_ratings is None, all ratings are used in the
        evaluation. Specific sampling objects can be passed, see
        scikits.crab.metrics.sampling module for the list of possible
        objects.

        Returns
        -------
        Returns a dictionary containing the evaluation results:
        (NMAE, MAE, RMSE, Precision, Recall, F1-Score)
        """
        sampling_users = kwargs.pop('sampling_users', None)
        sampling_ratings = kwargs.pop('sampling_ratings', None)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
              are %s' % (metric, evaluation_metrics.keys()))

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        #Select the users to be evaluated.

        #Select the ratings to be evaluated.

        #Evaluate the recommender.

        #Return the results.

    def evaluate_on_split(self, metric=None, **kwargs):
        """
        Evaluate on the folds of a dataset split

        Parameters
        ----------
        metric: [None|'rmse'|'f1score'|'precision'|'recall'|'nmae'|'mae']
        If metrics is None, all metrics available will be evaluated.
        Otherwise it will return the specified metric evaluated.

        Returns
        -------
        Returns a score representing how well the recommender estimated the
        preferences match real values.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

