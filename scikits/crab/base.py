import numpy as np

from ..base import BaseEstimator


class BaseRecommender(BaseEstimator):
    """
    Base Class for Recommenders that suggest items for users.
    
    Should not be used directly, use derived classes instead
    """
    pass