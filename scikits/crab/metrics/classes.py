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
from cross_validation import KFold
from metrics import precision_score
from metrics import recall_score
from metrics import f1_score
from sampling import SplitSampling
from scikits.learn.base import clone
from ..models.utils import ItemNotFoundError, UserNotFoundError


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


def check_cv(cv, n):
    """Input checker utility for building a
       cross validation in a user friendly way.

   Parameters
   ===========
    sampling: an integer, a cv generator instance, or None
        The input specifying which cv generator to use.
        It can be an integer, in which case it is the number
        of folds in a KFold, None, in which case 3 fold is used,
        or another object, that will then be used as a cv generator.

    n: an integer.
        The number of elements.

    """
    if cv is None:
        cv = 3
    if operator.isNumberType(cv):
        cv = KFold(n, cv, indices=True)

    return cv


class CfEvaluator(RecommenderEvaluator):

    """
    Examples
    --------
    >>> from scikits.crab.similarities import UserSimilarity
    >>> from scikits.crab.metrics import  euclidean_distances
    >>> from scikits.crab.models import  MatrixPreferenceDataModel
    >>> from scikits.crab.recommenders.knn import UserBasedRecommender
    >>> from scikits.crab.metrics.classes import CfEvaluator
    >>> from scikits.crab.recommenders.knn.neighborhood_strategies import NearestNeighborsStrategy
    >>> movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, \
    'Snakes on a Plane': 3.5, \
    'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
    'The Night Listener': 3.0}, \
    'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
    'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
    'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
    'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
    'The Night Listener': 4.5, 'Superman Returns': 4.0, \
    'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
    'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
    'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
    'The Night Listener': 3.0, 'Superman Returns': 5.0, \
    'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {}}
    >>> model = MatrixPreferenceDataModel(movies)
    >>> similarity = UserSimilarity(model, euclidean_distances)
    >>> neighborhood = NearestNeighborsStrategy()
    >>> recsys = UserBasedRecommender(model, similarity, neighborhood)
    >>> evaluator = CfEvaluator()
    >>> all_scores = evaluator.evaluate(recsys, permutation=False)
    >>> all_scores
    {'rmse': 0.23590725429603751, 'recall': 1.0, 'precision': 1.0, \
    'mae': 0.21812065003607684, 'f1score': 1.0, 'nmae': 0.054530162509019209}
    >>> rmse = evaluator.evaluate_on_split(recsys, metric='rmse', permutation=False)
    >>> rmse
    ({'error': [{'rmse': 0.35355339059327379}, \
     {'rmse': 0.97109049202292397},  \
     {'rmse': 0.39418387598407179}]},  \
     {'final_error': {'avg': {'rmse': 0.57294258620008975}, \
     'stdev': {'rmse': 0.28202130565981975}}})

    """

    def _build_recommender(self, dataset, recommender):
        """
        Build a clone recommender with the given dataset
        as the training set.

        Parameters
        ----------

        dataset: dict
            The dataset with the user's preferences.

        recommender: A scikits.crab.base.BaseRecommender object.
            The given recommender to be cloned.

        """
        recommender_training = clone(recommender)
        #if the recommender's model has the build_model implemented.

        if not recommender.model.has_preference_values():
            recommender_training.model.dataset = \
                    recommender_training.model._load_dataset(dataset.copy())
        else:
            recommender_training.model.dataset = dataset

        if hasattr(recommender_training.model, 'build_model'):
            recommender_training.model.build_model()

        return recommender_training

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
        ratings. If sampling_ratings is None, 70% will be used in the
        training set and 30% in the test set. Specific sampling objects
        can be passed, see scikits.crab.metrics.sampling module
        for the list of possible objects.

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

        """
        sampling_users = kwargs.pop('sampling_users', None)
        sampling_ratings = kwargs.pop('sampling_ratings', 0.7)
        permutation = kwargs.pop('permutation', True)
        at = kwargs.pop('at', 3)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
              are %s' % (metric, evaluation_metrics.keys()))

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        users_set, _ = sampling_users.split(permutation=permutation)

        training_set = {}
        testing_set = {}

        #Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            #Select the ratings to be evaluated.
            preferences = recommender.model.preferences_from_user(user_id)

            sampling_eval = check_sampling(sampling_ratings, \
                                             len(preferences))
            train_set, test_set = sampling_eval.split(indices=True,
                                        permutation=permutation)

            preferences = list(preferences)
            if recommender.model.has_preference_values():
                training_set[user_id] = dict((preferences[idx]
                             for idx in train_set)) if preferences else {}
                testing_set[user_id] = [preferences[idx]
                             for idx in test_set] if preferences else []
            else:
                training_set[user_id] = dict(((preferences[idx], 1.0)
                             for idx in train_set)) if preferences else {}
                testing_set[user_id] = [(preferences[idx], 1.0)
                             for idx in test_set] if preferences else []

        #Evaluate the recommender.
        recommender_training = self._build_recommender(training_set, \
                                recommender)

        real_preferences = []
        estimated_preferences = []

        for user_id, preferences in testing_set.iteritems():
            for item_id, preference in preferences:
            #Estimate the preferences
                try:
                    estimated = recommender_training.estimate_preference(
                                user_id, item_id)
                    real_preferences.append(preference)
                except ItemNotFoundError:
                    # It is possible that an item exists in the test data but
                    # not training data in which case an exception will be
                    # throw. Just ignore it and move on
                    continue
                estimated_preferences.append(estimated)

        #Return the error results.
        if metric in ['rmse', 'mae', 'nmae']:
            eval_function = evaluation_metrics[metric]
            if metric == 'nmae':
                return {metric: eval_function(real_preferences,
                                          estimated_preferences,
                                recommender.model.maximum_preference_value(),
                                recommender.model.minimum_preference_value())}
            return {metric: eval_function(real_preferences,
                                          estimated_preferences)}

        #IR_Statistics
        relevant_arrays = []
        real_arrays = []

        #Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            preferences = recommender.model.preferences_from_user(user_id)
            preferences = list(preferences)
            if len(preferences) < 2 * at:
                # Really not enough prefs to meaningfully evaluate the user
                continue

            # List some most-preferred items that would count as most
            if not recommender.model.has_preference_values():
                preferences = [(preference, 1.0) for preference in preferences]

            preferences = sorted(preferences, key=lambda x: x[1], reverse=True)
            relevant_item_ids = [item_id for item_id, preference
                                    in preferences[:at]]

            if len(relevant_item_ids) == 0:
                continue

            training_set = {}
            for other_user_id in recommender.model.user_ids():
                preferences_other_user = \
                    recommender.model.preferences_from_user(other_user_id)

                if not recommender.model.has_preference_values():
                    preferences_other_user = [(preference, 1.0)
                                     for preference in preferences_other_user]
                if other_user_id == user_id:
                    preferences_other_user = \
                        [pref for pref in preferences_other_user \
                            if pref[0] not in relevant_item_ids]

                    if preferences_other_user:
                        training_set[other_user_id] = \
                            dict(preferences_other_user)
                else:
                    training_set[other_user_id] = dict(preferences_other_user)

            #Evaluate the recommender
            recommender_training = self._build_recommender(training_set, \
                        recommender)

            try:
                preferences = \
                    recommender_training.model.preferences_from_user(user_id)
                preferences = list(preferences)
                if not preferences:
                    continue
            except:
                #Excluded all prefs for the user. move on.
                continue

            recommended_items = recommender_training.recommend(user_id, at)
            relevant_arrays.append(list(relevant_item_ids))
            real_arrays.append(list(recommended_items))

        relevant_arrays = np.array(relevant_arrays)
        real_arrays = np.array(real_arrays)

        #Return the IR results.
        if metric in ['precision', 'recall', 'f1score']:
            eval_function = evaluation_metrics[metric]
            return {metric: eval_function(real_arrays, relevant_arrays)}

        if metric is None:
            #Return all
            mae, nmae, rmse = evaluation_error(real_preferences,
                        estimated_preferences,
                        recommender.model.maximum_preference_value(),
                        recommender.model.minimum_preference_value())
            f = f1_score(real_arrays, relevant_arrays)
            r = recall_score(real_arrays, relevant_arrays)
            p = precision_score(real_arrays, relevant_arrays)

            return {'mae': mae, 'nmae': nmae, 'rmse': rmse,
                    'precision': p, 'recall': r, 'f1score': f}

    def evaluate_on_split(self, recommender, metric=None, cv=None, **kwargs):
        """
        Evaluate on the folds of a dataset split

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

        cv: integer or crossvalidation, optional, default = None
            If an integer is passed, it is the number of fold (default 3).
            Specific sampling objects can be passed, see
            scikits.crab.metrics.cross_validation module for the list of
            possible objects.

        at: integer, optional, default = None
            This number at is the 'at' value, as in 'precision at 5'.  For
        example this would mean precision or recall evaluated by removing
        the top 5 preferences for a user and then finding the percentage of
        those 5 items included in the top 5 recommendations for that user.
        If at is None, it will consider all the top 3 elements.

        Returns
        -------
        score: dict
            a dictionary containing the average results over
            the different permutations on the split.

        permutation_scores : array, shape = [n_permutations]
            The scores obtained for each permutations.

        """
        sampling_users = kwargs.pop('sampling_users', 0.7)
        permutation = kwargs.pop('permutation', True)
        at = kwargs.pop('at', 3)

        if metric not in evaluation_metrics and metric is not None:
            raise ValueError('metric %s is not recognized. valid keywords \
              are %s' % (metric, evaluation_metrics.keys()))

        permutation_scores_error = []
        permutation_scores_ir = []
        final_score_error = {'avg': {}, 'stdev': {}}
        final_score_ir = {'avg': {}, 'stdev': {}}

        n_users = recommender.model.users_count()
        sampling_users = check_sampling(sampling_users, n_users)
        users_set, _ = sampling_users.split(permutation=permutation)

        total_ratings = []
        #Select the users to be evaluated.
        user_ids = recommender.model.user_ids()
        for user_id in user_ids[users_set]:
            #Select the ratings to be evaluated.
            preferences = recommender.model.preferences_from_user(user_id)
            preferences = list(preferences)
            total_ratings.extend([(user_id, preference)
                                 for preference in preferences])

        n_ratings = len(total_ratings)
        cross_val = check_cv(cv, n_ratings)
        #Defining the splits and run on the splits.
        for train_set, test_set in cross_val:

            training_set = {}
            testing_set = {}

            for idx in train_set:
                user_id, pref = total_ratings[idx]
                if recommender.model.has_preference_values():
                    training_set.setdefault(user_id, {})
                    training_set[user_id][pref[0]] = pref[1]
                else:
                    training_set.setdefault(user_id, {})
                    training_set[user_id][pref] = 1.0

            for idx in test_set:
                user_id, pref = total_ratings[idx]
                if recommender.model.has_preference_values():
                    testing_set.setdefault(user_id, [])
                    testing_set[user_id].append(pref)
                else:
                    testing_set.setdefault(user_id, [])
                    testing_set[user_id].append((pref, 1.0))

            #Evaluate the recommender.
            recommender_training = self._build_recommender(training_set, \
                                    recommender)

            real_preferences = []
            estimated_preferences = []

            for user_id, preferences in testing_set.iteritems():
                for item_id, preference in preferences:
                    #Estimate the preferences
                    try:
                        estimated = recommender_training.estimate_preference(
                                    user_id, item_id)
                        real_preferences.append(preference)
                    except:
                        # It is possible that an item exists
                        #in the test data but
                        # not training data in which case
                        #an exception will be
                        # throw. Just ignore it and move on
                        continue
                    estimated_preferences.append(estimated)

            #Return the error results.
            if metric in ['rmse', 'mae', 'nmae']:
                eval_function = evaluation_metrics[metric]
                if metric == 'nmae':
                    permutation_scores_error.append({
                                metric: eval_function(real_preferences,
                                                 estimated_preferences,
                                recommender.model.maximum_preference_value(),
                                recommender.model.minimum_preference_value())})
                else:
                    permutation_scores_error.append(
                    {metric: eval_function(real_preferences,
                                       estimated_preferences)})
            elif metric is None:
                #Return all
                mae, nmae, rmse = evaluation_error(real_preferences,
                        estimated_preferences,
                        recommender.model.maximum_preference_value(),
                        recommender.model.minimum_preference_value())
                permutation_scores_error.append({'mae': mae, 'nmae': nmae,
                                                  'rmse': rmse})

        #IR_Statistics (Precision, Recall and F1-Score)
        n_users = recommender.model.users_count()
        cross_val = check_cv(cv, n_users)

        for train_idx, test_idx in cross_val:
            relevant_arrays = []
            real_arrays = []
            for user_id in user_ids[train_idx]:
                preferences = recommender.model.preferences_from_user(user_id)
                preferences = list(preferences)
                if len(preferences) < 2 * at:
                    # Really not enough prefs to meaningfully evaluate the user
                    continue

                # List some most-preferred items that would count as most
                if not recommender.model.has_preference_values():
                    preferences = [(preference, 1.0) for preference in preferences]

                preferences = sorted(preferences, key=lambda x: x[1], reverse=True)
                relevant_item_ids = [item_id for item_id, preference
                                        in preferences[:at]]

                if len(relevant_item_ids) == 0:
                    continue

                #Build the training set.
                training_set = {}
                for other_user_id in recommender.model.user_ids():
                    preferences_other_user = \
                        recommender.model.preferences_from_user(other_user_id)

                    if not recommender.model.has_preference_values():
                        preferences_other_user = [(preference, 1.0)
                                         for preference in preferences_other_user]
                    if other_user_id == user_id:
                        preferences_other_user = \
                            [pref for pref in preferences_other_user \
                                if pref[0] not in relevant_item_ids]

                        if preferences_other_user:
                            training_set[other_user_id] = \
                                dict(preferences_other_user)
                    else:
                        training_set[other_user_id] = dict(preferences_other_user)

                #Evaluate the recommender
                recommender_training = self._build_recommender(training_set, \
                            recommender)

                try:
                    preferences = \
                        recommender_training.model.preferences_from_user(user_id)
                    preferences = list(preferences)
                    if not preferences:
                        continue
                except:
                    #Excluded all prefs for the user. move on.
                    continue

                recommended_items = recommender_training.recommend(user_id, at)
                relevant_arrays.append(list(relevant_item_ids))
                real_arrays.append(list(recommended_items))

            relevant_arrays = np.array(relevant_arrays)
            real_arrays = np.array(real_arrays)

            #Return the IR results.
            if metric in ['precision', 'recall', 'f1score']:
                eval_function = evaluation_metrics[metric]
                permutation_scores_ir.append({metric: eval_function(real_arrays,
                                          relevant_arrays)})
            elif metric is None:
                f = f1_score(real_arrays, relevant_arrays)
                r = recall_score(real_arrays, relevant_arrays)
                p = precision_score(real_arrays, relevant_arrays)
                permutation_scores_ir.append({'precision': p, 'recall': r, 'f1score': f})

        #Compute the final score for Error Statistics
        for result in permutation_scores_error:
            for key in result:
                final_score_error['avg'].setdefault(key, [])
                final_score_error['avg'][key].append(result[key])
        for key in final_score_error['avg']:
            final_score_error['stdev'][key] = np.std(final_score_error['avg'][key])
            final_score_error['avg'][key] = np.average(final_score_error['avg'][key])

        #Compute the final score for IR statistics
        for result in permutation_scores_ir:
            for key in result:
                final_score_ir['avg'].setdefault(key, [])
                final_score_ir['avg'][key].append(result[key])
        for key in final_score_ir['avg']:
            final_score_ir['stdev'][key] = np.std(final_score_ir['avg'][key])
            final_score_ir['avg'][key] = np.average(final_score_ir['avg'][key])

        permutation_scores = {}
        scores = {}
        if permutation_scores_error:
            permutation_scores['error'] = permutation_scores_error
            scores['final_error'] = final_score_error
        if permutation_scores_ir:
            permutation_scores['ir'] = permutation_scores_ir
            scores.setdefault('final_error', {})
            scores['final_error'].setdefault('avg', {})
            scores['final_error'].setdefault('stdev', {})
            scores['final_error']['avg'].update(final_score_ir['avg'])
            scores['final_error']['stdev'].update(final_score_ir['stdev'])

        return permutation_scores, scores
