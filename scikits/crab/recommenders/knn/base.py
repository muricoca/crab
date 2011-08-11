"""
Generalized Recommender models amd utility classes.

This module contains basic memory recommender interfaces used throughout
the whole scikit-crab package as also utility classes.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be subclassed).

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.

from ..base import MemoryBasedRecommender

#===========================
#Item-based Recommender Interface


class ItemRecommender(MemoryBasedRecommender):

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
        raise NotImplementedError("ItemRecommender is an abstract class.")

    def recommended_because(user_id, item_id, how_many, **params):
        '''
        Returns the items that were most influential in recommending a given item
        to a given user. In most implementations, this method will return items
        that the user prefers and that are similar to the given item.

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
        The list of items ordered from most influential in recommended the given item to least
        '''
        raise NotImplementedError("ItemRecommender is an abstract class.")


#===========================
#User-based Recommender Interface


class UserRecommender(MemoryBasedRecommender):

    def most_similar_users(user_id, how_many):
        '''
        Return the most similar users to the given user, ordered
        from most similar to least.

        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        how_many: int
            Desired number of most similar users to find
        '''
        raise NotImplementedError("UserRecommender is an abstract class.")



#===========================
# Base Item Candidate Strategy


class BaseCandidateItemsStrategy(object):
    '''
    Base implementation for retrieving
    all items that could possibly be recommended to the user
    '''

    def candidate_items(user_id, data_model, **params):
        '''
        Return the candidate items that could possibly be recommended to the user

        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        data_model: The data model that will be the source for the possible
            candidates
        '''
        raise NotImplementedError("BaseCandidateItemsStrategy is an abstract class.")
