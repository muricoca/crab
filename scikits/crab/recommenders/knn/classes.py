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

    Examples
    -----------

    Notes
    -----------

    References
    -----------

    """

    def __init__(self, model, similarity, items_selection_strategy=None,
                capper=True):
        ItemRecommender.__init__(self, model)
        self.similarity = similarity
        self.capper = capper
        if items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = items_selection_strategy

    def recommend(self, user_id, how_many, **params):
        '''
        Return a list of recommended items, ordered from most strongly
        recommend to least.

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        how_many: int
                 Desired number of recommendations
        rescorer:  function, optional
                 Rescoring function to apply before final list of
                 recommendations.

        '''

        self._set_params(**params)

        candidate_items = self.all_other_items(user_id)

        recommendable_items = None

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

    def most_similar_items(item_id, how_many):
        '''
        Return the most similar items to the given item, ordered
        from most similar to least.

        Parameters
        -----------
        item_id:  int or string
            ID of item for which to find most similar other items

        how_many: int
            Desired number of most similar items to find
        '''
        pass

    def recommended_because(user_id, item_id, how_many, **params):
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
