import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from ....models.classes import DictPreferenceDataModel, MatrixPreferenceDataModel
from ..neighborhood_strategies import AllNeighborsStrategy, NearestNeighborsStrategy
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


def test_AllNeighborsStrategy():
    #Empty Dataset
    model = DictPreferenceDataModel({})
    strategy = AllNeighborsStrategy()
    assert_array_equal(np.array([]), strategy.user_neighborhood('Lorena Abreu', model))

    model = MatrixPreferenceDataModel({})
    strategy = AllNeighborsStrategy()
    assert_array_equal(np.array([]), strategy.user_neighborhood('Lorena Abreu', model))

    #Possible candidates
    model = DictPreferenceDataModel(movies)
    strategy = AllNeighborsStrategy()
    assert_array_equal(np.array(['Leopoldo Pires', 'Luciana Nunes', 'Marcel Caraciolo',
       'Maria Gabriela', 'Penny Frewman', 'Sheldom', 'Steve Gates']), strategy.user_neighborhood('Lorena Abreu', model))

    model = MatrixPreferenceDataModel(movies)
    strategy = AllNeighborsStrategy()
    assert_array_equal(np.array(['Leopoldo Pires', 'Luciana Nunes', 'Marcel Caraciolo',
       'Maria Gabriela', 'Penny Frewman', 'Sheldom', 'Steve Gates']), strategy.user_neighborhood('Lorena Abreu', model))


def test_NearestNeighborsStrategy():
    #Empty Dataset
    model = DictPreferenceDataModel({})
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array([]), strategy.user_neighborhood('Lorena Abreu', model))

    model = MatrixPreferenceDataModel({})
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array([]), strategy.user_neighborhood('Lorena Abreu', model))

    #Possible candidates
    model = DictPreferenceDataModel(movies)
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array(['Leopoldo Pires', 'Marcel Caraciolo', 'Sheldom',
       'Luciana Nunes', 'Penny Frewman', 'Steve Gates'], dtype='|S16'),
        strategy.user_neighborhood('Lorena Abreu', model))

    model = MatrixPreferenceDataModel(movies)
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array(['Leopoldo Pires', 'Marcel Caraciolo', 'Sheldom',
       'Luciana Nunes', 'Penny Frewman', 'Steve Gates'], dtype='|S16'),
       strategy.user_neighborhood('Lorena Abreu', model))

    #Empty candidates
    model = DictPreferenceDataModel(movies)
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array(['Leopoldo Pires', 'Steve Gates', 'Lorena Abreu', 'Sheldom',
       'Luciana Nunes', 'Penny Frewman'], dtype='|S14'),
        strategy.user_neighborhood('Marcel Caraciolo', model))

    model = MatrixPreferenceDataModel(movies)
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array(['Leopoldo Pires', 'Steve Gates', 'Lorena Abreu', 'Sheldom',
       'Luciana Nunes', 'Penny Frewman'], dtype='|S14'),
        strategy.user_neighborhood('Marcel Caraciolo', model))

    #Empty candidates
    model = DictPreferenceDataModel(movies)
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array([], dtype=bool), strategy.user_neighborhood('Maria Gabriela', model))

    model = MatrixPreferenceDataModel(movies)
    strategy = NearestNeighborsStrategy()
    assert_array_equal(np.array([], dtype=bool), strategy.user_neighborhood('Maria Gabriela', model))
