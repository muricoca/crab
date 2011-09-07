import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from ....models.classes import  MatrixPreferenceDataModel
from ..item_strategies import ItemsNeighborhoodStrategy, AllPossibleItemsStrategy
from ....models.utils import UserNotFoundError


movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
 'The Night Listener': 3.0},
'Luciana Nunes': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 3.5},
'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0,
 'You, Me and Dupree': 2.5},
'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0},
'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Penny Frewman': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0},
'Maria Gabriela': {}}


def test_ItemsNeighborhoodStrategy():
    #Empty Dataset
    model = MatrixPreferenceDataModel({})
    strategy = ItemsNeighborhoodStrategy()
    assert_raises(UserNotFoundError, strategy.candidate_items, 'Lorena Abreu', model)

    #Possible candidates
    model = MatrixPreferenceDataModel(movies)
    strategy = ItemsNeighborhoodStrategy()
    assert_array_equal(np.array(['Lady in the Water']), strategy.candidate_items('Lorena Abreu', model))

    #Empty candidates
    model = MatrixPreferenceDataModel(movies)
    strategy = ItemsNeighborhoodStrategy()
    assert_array_equal(np.array([], dtype='|S'), strategy.candidate_items('Marcel Caraciolo', model))

    #Empty candidates
    model = MatrixPreferenceDataModel(movies)
    strategy = ItemsNeighborhoodStrategy()
    assert_array_equal(np.array([], dtype=bool), strategy.candidate_items('Maria Gabriela', model))


def test_AllPossibleItemsStrategy():
    #Empty Dataset
    model = MatrixPreferenceDataModel({})
    strategy = AllPossibleItemsStrategy()
    assert_raises(UserNotFoundError, strategy.candidate_items, 'Lorena Abreu', model)

    #Possible candidates
    model = MatrixPreferenceDataModel(movies)
    strategy = AllPossibleItemsStrategy()
    assert_array_equal(np.array(['Lady in the Water']), strategy.candidate_items('Lorena Abreu', model))

    #Empty candidates
    model = MatrixPreferenceDataModel(movies)
    strategy = AllPossibleItemsStrategy()
    assert_array_equal(np.array([], dtype='|S'), strategy.candidate_items('Marcel Caraciolo', model))

    #Empty candidates
    model = MatrixPreferenceDataModel(movies)
    strategy = AllPossibleItemsStrategy()
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), strategy.candidate_items('Maria Gabriela', model))
