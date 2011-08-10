"""
Generalized Recommender models.

This module contains basic memory recommender interfaces used throughout
the whole scikit-crab package.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be subclassed).

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.

from base import ItemRecommender
from item_strategies import ItemsNeighborhoodStrategy
import numpy as np


class ItemBasedRecommender(ItemRecommender):
    """
    Item Based Collaborative Filtering Recommender.


    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    similarity: The Item Similarity instance that will be used to
        score the items that will be recommended.

    iss: The item candidates strategy that you can choose
        for selecting the possible items to recommend. default =


    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The Item Similarity instance that will be used to
        score the items that will be recommended.

    `items_selection_strategy`: The item candidates strategy that you
         can choose for selecting the possible items to recommend.
         default = ItemsNeighborhoodStrategy

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Examples
    -----------

    Notes
    -----------

    References
    -----------

    """

    def __init__(self, model, similarity, items_selection_strategy=None,
                capper=True, with_preference=False):
        ItemRecommender.__init__(self, model, with_preference)
        self.similarity = similarity
        self.capper = capper
        if items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

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
        rescorer:  function, optional
                 Rescoring function to apply before final list of
                 recommendations.

        '''
        self._set_params(**params)

        candidate_items = self.all_other_items(user_id)

        recommendable_items = self._top_matches(user_id, \
                 candidate_items, how_many)

        return recommendable_items

    def estimate_preference(self, user_id, item_id, **params):
        '''
        Returns
        -------
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return preference

        #TODO: It needs optimization
        prefs = self.model.preferences_from_user(user_id)

        similarities = \
            np.array([self.similarity.get_similarity(item_id, to_item_id) \
            for to_item_id, pref in prefs if to_item_id != item_id]).flatten()

        prefs = np.array([pref for it, pref in prefs])
        prefs_sim = np.sum(prefs[~np.isnan(similarities)] *
                             similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)

        #Throw out the estimate if it was based on no data points,
        #of course, but also if based on
        #just one. This is a bit of a band-aid on the 'stock'
        #item-based algorithm for the moment.
        #The reason is that in this case the estimate is, simply,
        #the user's rating for one item
        #that happened to have a defined similarity.
        #The similarity score doesn't matter, and that
        #seems like a bad situation.
        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            return np.nan

        estimated = prefs_sim / total_similarity

        if self.capper:
            max_p = self.model.maximum_preference_value()
            min_p = self.model.minimum_preference_value()
            estimated = max_p if estimated > max_p else min_p \
                     if estimated < min_p else estimated
        return estimated

    def all_other_items(self, user_id, **params):
        '''
        Return items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
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
             if how_many and sorted_preferences.size > how_many else sorted_preferences

        if self.with_preference:
            top_n_recs = np.array([(target_ids[ind], \
                     preferences[ind]) for ind in sorted_preferences])
        else:
            top_n_recs = np.array([target_ids[ind] for ind in sorted_preferences])

        return top_n_recs

    def most_similar_items(self, item_id, how_many=None):
        '''
        Return the most similar items to the given item, ordered
        from most similar to least.

        Parameters
        -----------
        item_id:  int or string
            ID of item for which to find most similar other items

        how_many: int
            Desired number of most similar items to find default=None (ALL)
        '''
        old_how_many = self.similarity.num_best
        #+1 since it returns the identity.
        self.similarity.num_best = how_many + 1 \
                    if how_many is not None else None
        similarities = self.similarity[item_id]
        self.similarity.num_best = old_how_many

        return np.array([item for item, pref in similarities \
            if item != item_id])

    def recommended_because(self, user_id, item_id, how_many, **params):
        '''
        Returns the items that were most influential in recommending a
        given item to a given user. In most implementations, this
        method will return items that the user prefers and that
        are similar to the given item.

        Parameters
        -----------
        user_id : int or string
            ID of the user who was recommended the item

        item_id: int or string
            ID of item that was recommended

        how_many: int
            Maximum number of items to return.

        Returns
        ----------
        The list of items ordered from most influential in
        recommended the given item to least
        '''
        pass
