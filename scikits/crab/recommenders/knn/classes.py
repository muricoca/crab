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
    `data_model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The Item Similarity instance that will be used to
        score the items that will be recommended.

    `items_selection_strategy`: The item candidates strategy that you
         can choose for selecting the possible items to recommend.
         default = ItemsNeighborhoodStrategy

    Examples
    -----------

    Notes
    -----------

    References
    -----------

    """

    def __init__(self, data_model, similarity, iss=None):
        ItemRecommender.__init__(self, data_model)
        self.similarity = similarity
        if self.items_selection_strategy is None:
            self.items_selection_strategy = ItemsNeighborhoodStrategy()
        else:
            self.items_selection_strategy = iss

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
        pass

    def estimate_preference(self, **params):
        '''
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''
        pass

    def all_other_items(self, user_id):
        '''
        Return all items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        '''
        pass

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

    def recommended_because(user_id, item_id, how_many):
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
        pass
