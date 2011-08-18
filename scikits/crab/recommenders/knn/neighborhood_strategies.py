"""
Strategies for users selection to be a
possible candidate to be member of a user neighborhood.

Please check the base.BaseUserNeighborhoodStrategy before
implement your own strategy.

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.

from base import BaseUserNeighborhoodStrategy
import numpy as np


class AllNeighborsStrategy(BaseUserNeighborhoodStrategy):
    '''
    Returns
    --------
    Returns all users in the model.
    This strategy is not recommended for large datasets and
    it is the dummiest one.
    '''
    def user_neighborhood(self, user_id, data_model, **params):
        '''
        Computes a neighborhood consisting of the  n users to a given user based on the
        strategy implemented in this method.
        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        data_model: The data model that will be the source for the possible
            candidates
        '''
        user_ids = data_model.user_ids()
        return user_ids[user_ids != user_id] if user_ids.size else user_ids
