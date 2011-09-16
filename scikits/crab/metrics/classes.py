 #-*- coding:utf-8 -*-

"""
This module contains main implementations that encapsulate
    retrieval-related statistics about the quality of the recommender's
    recommendations.

"""
# Authors: Marcel Caraciolo <marcel@muricoca.com>
# License: BSD Style.

import operator
import numpy as np
from base import RecommenderEvaluator
from metrics import root_mean_square_error
from metrics import mean_absolute_error
from metrics import normalized_mean_absolute_error
from metrics import evaluation_error
from metrics import precision_recall_fscore
from metrics import precision_score
from metrics import recall_score
from metrics import f1_score
from sampling import SplitSampling
from scikits.learn.base import clone


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

        at: integer, optional, default = None
            This number at is the 'at' value, as in 'precision at 5'.  For
        example this would mean precision or recall evaluated by removing
        the top 5 preferences for a user and then finding the percentage of
        those 5 items included in the top 5 recommendations for that user.
        If at is None, it will consider all the top 3 elements.

        Returns
        -------
        Returns a dictionary containing the evaluation results:
        (NMAE, MAE, RMSE, Precision, Recall, F1-Score)

        Examples
        --------

        """
        sampling_users = kwargs.pop('sampling_users', None)
        sampling_ratings = kwargs.pop('sampling_ratings', None)
        at = kwargs.pop('at', 3)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
              are %s' % (metric, evaluation_metrics.keys()))

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        users_set, _ = sampling_users.split()

        training_set = {}
        testing_set = {}

        #Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            #Select the ratings to be evaluated.
            preferences = recommender.model.preferences_from_user(user_id)
            sampling_ratings = check_sampling(sampling_ratings, \
                                             len(preferences))
            train_set, test_set = sampling_ratings.split()
            training_set[user_id] = preferences[train_set]
            testing_set[user_id] = preferences[test_set]

        #Evaluate the recommender.
        recommender_training = clone(recommender)
        recommender_training.model = training_set
        #if the recommender has the build_model implemented.
        if hasattr(recommender_training, 'build_model'):
            recommender_training.build_model()

        real_preferences = []
        user_ids = []
        item_ids = []
        for user_id, preferences in testing_set:
            user_ids.append(user_id)
            for item_id, preference in preferences:
                real_preferences.append(preference)
                item_ids.append(item_id)

        estimate_preferences = np.vectorize(recommender_training.estimate_preference)
        preferences = estimate_preferences(user_ids, item_ids)
        #Return the error results.
        if metric in ['rmse', 'mae', 'nmae']:
            eval_function = evaluation_metrics[metric]
            return {metric: eval_function(real_preferences, preferences)}

        #IR_Statistics
        training_set = {}
        relevant_arrays = []
        real_arrays = []

        #Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            preferences = recommender.model.preferences_from_user(user_id)
            if len(preferences) < 2 * at:
                # Really not enough prefs to meaningfully evaluate the user
                continue

            # List some most-preferred items that would count as most
            preferences = sorted(preferences, key=lambda x: x[1], reverse=True)
            relevant_item_ids = [item_id for item_id, preference in preferences[:at]]

            if len(relevant_item_ids) == 0:
                continue

            for other_user_id in recommender.model.user_ids():
                preferences_other_user = recommender.model.preferences_from_user(other_user_id)
                if other_user_id == user_id:
                    preferences_other_user = [pref for pref in preferences_other_user \
                            if pref[0] not in relevant_item_ids]

                    if preferences_other_user:
                        training_set[other_user_id] = dict(preferences_other_user)
            else:
                training_set[other_user_id] = dict(preferences_other_user)

            recommender_training.model = training_set
            #if the recommender has the build_model implemented.
            if hasattr(recommender_training, 'build_model'):
                recommender_training.build_model()

            try:
                preferences = recommender_training.model.preferences_from_user(user_id)
                if not preferences:
                    continue
            except:
                #Excluded all prefs for the user. move on.
                continue

            recommended_items = recommender.recommend(user_id, at)
            relevant_arrays.append(relevant_item_ids)
            real_arrays.append(recommended_items)

        relevant_arrays = np.array(relevant_arrays)
        real_arrays = np.array(real_arrays)

        #Return the IR results.
        if metric in ['precision', 'recall', 'f1score']:
            eval_function = evaluation_metrics[metric]
            return {metric: eval_function(real_arrays, relevant_arrays)}

        if metric is None:
            #Return all
            mae, nmae, rmse = evaluation_error(real_preferences, preferences,
                        recommender.model.maximum_preference_value(),
                        recommender.model.minimum_preference_value())
            p, r, f = precision_recall_fscore(real_arrays, relevant_arrays)

            return {'mae': mae, 'nmae': nmae, 'rmse': rmse,
                    'precision': p, 'recall': r, 'f1-score': f}

    def evaluate_on_split(self, recommender, metric=None, **kwargs):
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

        Examples
        --------

        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")
