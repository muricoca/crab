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
#Matrix Factorization Recommender Interface


class SVDRecommender(MemoryBasedRecommender):

    def factorize(self):
        '''
        Factorize the ratings matrix with a factorization
         technique implemented in this method.

        Parameters
        -----------

        Returns
        -----------
        '''
        raise NotImplementedError("ItemRecommender is an abstract class.")

    def train(self):
        '''
        Train the recommender with the matrix factorization method chosen.

        Parameters
        -----------

        Returns
        ----------

        '''
        raise NotImplementedError("ItemRecommender is an abstract class.")
