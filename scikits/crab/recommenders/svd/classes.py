"""
Generalized Recommender models.

This module contains matrix factorization recommender interfaces
used throughout the whole scikit-crab package.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be subclassed).

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.
import random

from base import SVDRecommender
from ..knn.item_strategies import ItemsNeighborhoodStrategy
import numpy as np
from math import sqrt
import logging

logger = logging.getLogger('crab')


class MatrixFactorBasedRecommender(SVDRecommender):
    """
    Matrix Factorization Based Recommender using
    Expectation Maximization algorithm.

    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    items_selection_strategy: The item candidates strategy that you
     can choose for selecting the possible items to recommend.
     default = ItemsNeighborhoodStrategy

    n_features: int
            Number of latent factors. default = 10

    learning_rate: float
        Learning rate used. default =  0.01

    regularization: float
            Parameter used to prevent overfitting. default = 0.02

    init_mean: float
            Mean of the normal distribution used to initialize
            the factors. default = 0.1

    init_std: float
            Standard deviation of the normal distribution used to
            initialize the factors. default = 0.1

    n_interations: int
            Number of iterations over the training data. default = 30

    capper: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    with_preference: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `items_selection_strategy`: The item candidates strategy that you
         can choose for selecting the possible items to recommend.
         default = ItemsNeighborhoodStrategy

    `n_features`:  int
            Number of latent factors. default = 10

    `learning_rate`: float
            Learning rate used. default = 0.01

    `regularization`: float
            Parameter used to prevent overfitting. default = 0.02

    `random_noise`: float
            Parameter used to initialize the latent factors.

    `n_interations`: int
            Number of iterations over the training data

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.

    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    `user_factors`: array of shape [n_users, n_features]
             Matrix containing the latent item factors

    `item_factors`: array of shape [n_items, n_features]
            Matrix containing the latent item factors

    Examples
    -----------
    >>> from scikits.crab.models.classes import MatrixPreferenceDataModel
    >>> from scikits.crab.recommenders.svd.classes import MatrixFactorBasedRecommender
    >>> from scikits.crab.recommenders.knn.item_strategies import ItemsNeighborhoodStrategy
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
    >>> items_strategy = ItemsNeighborhoodStrategy()
    >>> recsys = MatrixFactorBasedRecommender( \
        model=model, \
        items_selection_strategy=items_strategy, \
        n_features=2)
    >>> #Return the recommendations for the given user.
    >>> recsys.recommend('Leopoldo Pires')
    array(['Just My Luck', 'You, Me and Dupree'], \
          dtype='|S18')

    Notes
    -----------
    This MatrixFactorizationRecommender does not yet provide
    suppot for rescorer functions.

    This MatrixFactorizationRecommender does not yet provide
    suppot for DictDataModels.

    References
    -----------


    """

    def __init__(self, model, items_selection_strategy=None,
            n_features=10, learning_rate=0.01, regularization=0.02, init_mean=0.1,
            init_stdev=0.1, n_interations=30, capper=True, with_preference=False):
        SVDRecommender.__init__(self, model, with_preference)
        self.capper = capper
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.init_mean = init_mean
        self.init_stdev = init_stdev
        self.n_interations = n_interations
        self._global_bias = self._get_average_preference()
        self.user_factors = None
        self.item_factors = None

        if items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

        self.factorize()

    def _init_models(self):
        num_users = self.model.users_count()
        num_items = self.model.items_count()

        self.user_factors = np.empty(shape=(num_users, self.n_features),
                    dtype=float)

        self.item_factors = np.empty(shape=(num_items, self.n_features),
                    dtype=float)

        '''
        pref_interval = self.model.max_preference() - self.model.min_preference()
        default_value = math.sqrt(global_bias - pref_interval * 0.1) / self.n_features
        interval = pref_interval * 0.1 / self.n_features

        for i in range(len(self.n_features)):
            for user_idx in self.model.num_users():
                self.user_factors[user_idx, i] = default_value + (random.random() - 0.5) * interval * 0.2

        for i in range(len(self.n_features)):
            for item_idx in self.model.num_items():
                self.item_factors[item_idx, i] = default_value + (random.random() - 0.5) * interval * 0.2
        '''
        #Initialize the matrix with normal distributed (Gaussian) Noise
        self.user_factors = self.init_mean * np.random.randn(num_users, self.n_features) + self.init_stdev ** 2
        self.item_factors = self.init_mean * np.random.randn(num_items, self.n_features) + self.init_stdev ** 2

    def _get_average_preference(self):
        if hasattr(self.model, 'index'):
            mdat = np.ma.masked_array(self.model.index, np.isnan(self.model.index))
        else:
            raise TypeError('This model is not yet supported for this recommender.')
        return np.mean(mdat)

    def _predict(self, user_index, item_index, trailing=True):
        #Compute the scalar product between two rows of two matrices
        result = self._global_bias + np.sum(self.user_factors[user_index] *
                                            self.item_factors[item_index])
        if trailing:
            max_preference = self.model.max_preference()
            min_preference = self.model.min_preference()
            if result > max_preference:
                result = max_preference
            elif result < min_preference:
                result = min_preference

        return result

    def _train(self, rating_indices, update_user, update_item):
        '''
        Iterate once over rating data and adjust corresponding factors (stochastic gradient descent)
        '''
        err_total = 0.0
        for user_idx, item_idx in rating_indices:
            p = self._predict(user_idx, item_idx, False)
            err = self.model.index[user_idx, item_idx] - p
            err_total += err

            #Adjust the factors
            u_f = self.user_factors[user_idx]
            i_f = self.item_factors[item_idx]

            #Compute factor updates
            delta_u = err * u_f - self.regularization * u_f
            delta_i = err * u_f - self.regularization * i_f
            #if necessary apply updates
            if update_user:
                self.user_factors[user_idx] += self.learning_rate * delta_u
            if update_item:
                self.item_factors[item_idx] += self.learning_rate * delta_i

        return err_total

    def _rating_indices(self):
        if hasattr(self.model, 'index'):
            rating_indices = [(idx, jdx) for idx in range(self.model.users_count())
                                for jdx in range(self.model.items_count())
                        if not np.isnan(self.model.index[idx, jdx])]
        else:
            raise TypeError('This model is not yet supported for this recommender.')

        return rating_indices

    def learn_factors(self, update_user=True, update_item=True):
        rating_indices = self._rating_indices()
        random.shuffle(rating_indices)

        for index in range(self.n_interations):
            err = self._train(rating_indices, update_user, update_item)
            rmse = sqrt((err ** 2.0) / len(rating_indices))
            logger.debug("Finished the interation %i with RMSE %f" %  \
                    (index, rmse))

    def factorize(self):
        #init factor matrices
        self._init_models()
        #Learn the model parameters
        self.learn_factors()

    def recommend(self, user_id, how_many=None, **params):
        '''
        Return a list of recommended items, ordered from most strongly
        recommend to least.

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        how_many: int
                 Desired number of recommendations (default=None ALL)

        '''
        self._set_params(**params)

        candidate_items = self.all_other_items(user_id)

        recommendable_items = self._top_matches(user_id, \
                 candidate_items, how_many)

        return recommendable_items

    def estimate_preference(self, user_id, item_id, **params):
        '''
        A preference is estimated by computing the dot-product
        of the user and item feature vectors.
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        item_id:  int or string
            ID of item for which wants to find the estimated preference.

        Returns
        -------
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''

        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference

        #How to catch the user_id and item_id from the matrix.

        user_features = self.user_factors[np.where(self.model.user_ids() == user_id)]
        item_features = self.item_factors[np.where(self.model.item_ids() == item_id)]

        estimated = self._global_bias + np.sum(user_features * item_features)

        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        Returns
        ---------
        Return items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.

        '''
        return self.items_selection_strategy.candidate_items(user_id, \
                            self.model)

    def _top_matches(self, source_id, target_ids, how_many=None, **params):
        '''
        Parameters
        ----------
        target_ids: array of shape [n_target_ids]

        source_id: int or string
                item id to compare against.

        how_many: int
            Desired number of most top items to recommend (default=None ALL)

        Returns
        --------
        Return the top N matches
        It can be user_ids or item_ids.
        '''
        #Empty target_ids
        if target_ids.size == 0:
            return np.array([])

        estimate_preferences = np.vectorize(self.estimate_preference)

        preferences = estimate_preferences(source_id, target_ids)

        preferences = preferences[~np.isnan(preferences)]
        target_ids = target_ids[~np.isnan(preferences)]

        sorted_preferences = np.lexsort((preferences,))[::-1]

        sorted_preferences = sorted_preferences[0:how_many] \
             if how_many and sorted_preferences.size > how_many \
                else sorted_preferences

        if self.with_preference:
            top_n_recs = np.array([(target_ids[ind], \
                     preferences[ind]) for ind in sorted_preferences])
        else:
            top_n_recs = np.array([target_ids[ind]
                 for ind in sorted_preferences])

        return top_n_recs
