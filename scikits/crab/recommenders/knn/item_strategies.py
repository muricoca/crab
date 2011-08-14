"""
Strategies for items selection to be a
possible candidate to be recommended.

Please check the base.BaseCandidateItemsStrategy before
implement your own strategy.

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.

from base import BaseCandidateItemsStrategy
import numpy as np


class AllPossibleItemsStrategy(BaseCandidateItemsStrategy):
    '''
    Returns all items that have not been rated by the user.
    This strategy is not recommended for large datasets and
    it is the dummiest one.
    '''

    def candidate_items(self, user_id, data_model, **params):
        #Get all the item_ids preferred from the user
        preferences = data_model.items_from_user(user_id)
        #Get all posible items from the data_model
        possible_items = data_model.item_ids()
        return np.setdiff1d(possible_items, preferences, assume_unique=True)


class ItemsNeighborhoodStrategy(BaseCandidateItemsStrategy):
    '''
    Returns all items that have not been rated by the user and were
    preferred by another user that has preferred at least one item that the
    current has preferred too.
    '''

    def candidate_items(self, user_id, data_model, **params):
        #Get all the item_ids preferred from the user
        preferences = data_model.items_from_user(user_id)
        possible_items = np.array([])
        for item_id in preferences:
            item_preferences = data_model.preferences_for_item(item_id)
            if data_model.has_preference_values():
                for user_id, score in item_preferences:
                    possible_items = np.append(possible_items, \
                        data_model.items_from_user(user_id))
            else:
                for user_id in item_preferences:
                    possible_items = np.append(possible_items, \
                        data_model.items_from_user(user_id))
        possible_items = np.unique(possible_items)

        return np.setdiff1d(possible_items, preferences, assume_unique=True)
