
"""
Base IO code for all datasets
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>
#          Bruno Melo <bruno@muricoca.com>
# License: BSD Style.

from os.path import dirname
from os.path import join
import numpy as np


class Bunch(dict):
    """
    Container object for datasets: dictionary-like object
    that exposes its keys and attributes. """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_movielens_r100k(load_timestamp=False):
    """ Load and return the MovieLens dataset with
        100,000 ratings (only the user ids, item ids, timestamps
        and ratings).

    Parameters
    ----------
    load_timestamp: bool, optional (default=False)
        Whether it loads the timestamp.

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    Examples
    --------
    To load the MovieLens data::

    >>> from scikits.crab.datasets import load_movielens_r100k
    >>> movies = load_movielens_r100k()
    >>> len(movies['data'])
    943
    >>> len(movies['item_ids'])
    1682

    """
    base_dir = join(dirname(__file__), 'data/')
    #Read data
    if load_timestamp:
        data_m = np.loadtxt(base_dir + 'movielens100k.data', 
                delimiter='\t', skiprows=1, dtype=int)
        data_movies = {}
        for user_id, item_id, rating, timestamp in data_m:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = (timestamp, int(rating))
    else:
        data_m = np.loadtxt(base_dir + 'movielens100k.data', 
                delimiter='\t', skiprows=1, usecols=(0, 1, 2), dtype=int)

        data_movies = {}
        for user_id, item_id, rating in data_m:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = int(rating)

    #Read the titles
    data_titles = np.loadtxt(base_dir + 'movielens100k.item',
             delimiter='|', usecols=(0, 1), dtype=str)

    data_t = []
    for item_id, label in data_titles:
        data_t.append((int(item_id), label))
    data_titles = dict(data_t)

    fdescr = open(dirname(__file__) + '/descr/README')

    return Bunch(data=data_movies, item_ids=data_titles, 
                 user_ids=None, DESCR=fdescr.read())
