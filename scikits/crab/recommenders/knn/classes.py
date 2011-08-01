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
