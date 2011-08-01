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


class ItemsNeighborhoodStrategy(BaseCandidateItemsStrategy):
    '''
    Returns all items that have not been rated by the user and were
    preferred by another user that has preferred at least one item that the
    current has preferred too.
    '''

    def candidate_items(user_id, preferences_from_user, data_model, **params):
        pref_item_ids = [user_id for user_id, score in preferences_from_user]
        possible_items = []
        for item_id in pref_item_ids:
            item_preferences = data_model.preferences_for_item(item_id)
            for user_id, score in item_preferences:
                possible_items.append(data_model.items_from_user(user_id))

        return possible_items - pref_item_ids
