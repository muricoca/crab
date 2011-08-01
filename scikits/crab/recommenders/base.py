"""
Generalized Recommender models.

This module contains basic recommender interfaces used throughout
the whole scikit-crab package.

The interfaces are realized as abstract base classes (ie., some optional
functionality is provided in the interface itself, so that the interfaces
can be subclassed).

"""

# Author: Marcel Caraciolo <marcel@muricoca.com>
#
# License: BSD Style.

from ..base import BaseRecommender

#===========================
#Memory Based Recommender


class MemoryBasedRecommender(BaseRecommender):
    pass
