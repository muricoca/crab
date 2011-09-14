"""
Metrics module with score  functions, performance metrics and
pairwise metrics or distances computation
"""

from .pairwise import cosine_distances, euclidean_distances, pearson_correlation, \
    jaccard_coefficient, loglikehood_coefficient, manhattan_distances, \
     sorensen_coefficient, spearman_coefficient
from .cross_validation import LeaveOneOut, LeavePOut, KFold, ShuffleSplit
from .sampling import SplitSampling

