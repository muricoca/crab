import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_equals, assert_almost_equals
from ...knn.item_strategies import AllPossibleItemsStrategy, ItemsNeighborhoodStrategy
from ....models.classes import DictPreferenceDataModel, MatrixPreferenceDataModel, \
    DictBooleanPrefDataModel, MatrixBooleanPrefDataModel
from ..classes import MatrixFactorBasedRecommender


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

dict_model = DictPreferenceDataModel(movies)
matrix_model = MatrixPreferenceDataModel(movies)
boolean_model = DictBooleanPrefDataModel(movies)
boolean_matrix_model = MatrixBooleanPrefDataModel(movies)


def test_create_MatrixFactorBasedRecommender():
    items_strategy = AllPossibleItemsStrategy()
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_equals(recsys.items_selection_strategy, items_strategy)
    assert_equals(recsys.model, matrix_model)
    assert_equals(recsys.capper, True)
    assert_equals(recsys.learning_rate, 0.01)
    assert_equals(recsys.regularization, 0.02)
    assert_equals(recsys.init_mean, 0.1)
    assert_equals(recsys.n_interations, 30)
    assert_equals(recsys.init_stdev, 0.1)
    assert_equals(recsys.with_preference, False)
    assert_equals(recsys.user_factors.shape, (8, 2))
    assert_equals(recsys.item_factors.shape, (6, 2))
    assert_equals(recsys._global_bias, 3.2285714285714286)

    assert_raises(TypeError, MatrixFactorBasedRecommender,
        model=dict_model,
        items_selection_strategy=items_strategy,
        n_features=2)


def test_all_other_items_MatrixFactorBasedRecommender():
    items_strategy = AllPossibleItemsStrategy()
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))


def test_estimate_preference_MatrixFactorBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(3.206, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'), 1)

    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=3)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(3.21,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree'), 1)

    #With capper = False
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2, capper=False)
    assert_almost_equals(3.23, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'), 1)

    #Boolean Matrix Model
    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_almost_equals(1.0, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    assert_almost_equals(0.0,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree'))

    #With capper = False
    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2, capper=False)
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    assert_array_equal(0.0, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))


def test_recommend_MatrixFactorBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    #Empty Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array(['You, Me and Dupree', 'Just My Luck']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    assert_array_equal(np.array(['You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', n_features=2)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = MatrixFactorBasedRecommender(
    #    model=matrix_model,
    #    items_selection_strategy=items_strategy,
    #    n_features=2, with_preference=True)
    #assert_array_equal(np.array([('Just My Luck', 3.20597319063), \
    #            ('You, Me and Dupree', 3.14717875510)]), \
    #            recsys.recommend('Leopoldo Pires'))

    #Empty Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array(['You, Me and Dupree', 'Just My Luck']), \
    recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array(['You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = MatrixFactorBasedRecommender(
        model=boolean_matrix_model,
        items_selection_strategy=items_strategy,
        n_features=2)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', n_features=2)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = MatrixFactorBasedRecommender(
    #    model=boolean_matrix_model,
    #    items_selection_strategy=items_strategy,
    #    n_features=2)
    #assert_array_equal(np.array([('Just My Luck', 3.20597), \
    #            ('You, Me and Dupree', 3.1471)]), \
    #            recsys.recommend('Leopoldo Pires'))