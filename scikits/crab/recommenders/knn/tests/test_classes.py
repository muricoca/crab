import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_equals, assert_almost_equals
from ....models.classes import DictPreferenceDataModel, MatrixPreferenceDataModel, \
    DictBooleanPrefDataModel, MatrixBooleanPrefDataModel
from ..item_strategies import ItemsNeighborhoodStrategy, AllPossibleItemsStrategy
from ....similarities.basic_similarities import ItemSimilarity
from ..classes import ItemBasedRecommender
from ....models.utils import ItemNotFoundError
from ....metrics.pairwise import euclidean_distances, jaccard_coefficient


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


def test_create_ItemBasedRecommender():
    items_strategy = AllPossibleItemsStrategy()
    similarity = ItemSimilarity(dict_model, euclidean_distances)
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_equals(recsys.similarity, similarity)
    assert_equals(recsys.items_selection_strategy, items_strategy)
    assert_equals(recsys.model, dict_model)
    assert_equals(recsys.capper, True)


def test_all_other_items_ItemBasedRecommender():
    items_strategy = AllPossibleItemsStrategy()
    similarity = ItemSimilarity(dict_model, euclidean_distances)
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    similarity = ItemSimilarity(boolean_model, jaccard_coefficient)
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    similarity = ItemSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))


def test_estimate_preference_ItemBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(3.14717875510, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #With capper = False
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy, False)
    assert_almost_equals(3.14717875510, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    assert_array_equal(np.nan, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))

    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(dict_model, euclidean_distances)
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(3.14717875510, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #With capper = False
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy, False)
    assert_almost_equals(3.14717875510, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    assert_array_equal(np.nan, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))

    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(boolean_model, jaccard_coefficient)
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_almost_equals(1.0, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #With capper = False
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy, False)
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    assert_array_equal(0.0, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))

    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_almost_equals(1.0, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #With capper = False
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy, False)
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    assert_array_equal(0.0, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))


def test_most_similar_items_ItemBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    #semi items
    assert_array_equal(np.array(['Snakes on a Plane', \
        'The Night Listener', 'Lady in the Water', 'Just My Luck']), \
            recsys.most_similar_items('Superman Returns', 4))
    #all items
    assert_array_equal(np.array(['Lady in the Water', 'You, Me and Dupree', \
     'The Night Listener', 'Snakes on a Plane', 'Superman Returns']), \
            recsys.most_similar_items('Just My Luck'))
    #Non-existing
    assert_raises(ItemNotFoundError, recsys.most_similar_items, 'Back to the Future')
    #Exceed the limit
    assert_array_equal(np.array(['Lady in the Water', 'You, Me and Dupree', 'The Night Listener', \
       'Snakes on a Plane', 'Superman Returns']), \
            recsys.most_similar_items('Just My Luck', 20))
    #Empty
    assert_array_equal(np.array([]), \
            recsys.most_similar_items('Just My Luck', 0))

    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    #semi items
    assert_array_equal(np.array(['Snakes on a Plane', 'The Night Listener', \
    'You, Me and Dupree', 'Lady in the Water']), \
            recsys.most_similar_items('Superman Returns', 4))
    #all items
    assert_array_equal(np.array(['The Night Listener', 'You, Me and Dupree', \
        'Snakes on a Plane', 'Superman Returns', 'Lady in the Water']), \
            recsys.most_similar_items('Just My Luck'))
    #Non-existing
    assert_raises(ItemNotFoundError, recsys.most_similar_items, 'Back to the Future')
    #Exceed the limit
    assert_array_equal(np.array(['The Night Listener', 'You, Me and Dupree', 'Snakes on a Plane',
       'Superman Returns', 'Lady in the Water']), \
            recsys.most_similar_items('Just My Luck', 20))
    #Empty
    assert_array_equal(np.array([]), \
            recsys.most_similar_items('Just My Luck', 0))


def test_recommend_ItemBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    #Empty Recommendation
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['Just My Luck', 'You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['Just My Luck']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', similarity=similarity)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('Just My Luck', 3.20597319063), \
    #            ('You, Me and Dupree', 3.14717875510)]), \
    #            recsys.recommend('Leopoldo Pires'))

    similarity = ItemSimilarity(dict_model, euclidean_distances)
    #Empty Recommendation
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array(['Just My Luck', 'You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array(['Just My Luck']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', similarity=similarity)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = ItemBasedRecommender(dict_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('Just My Luck', 3.20597), \
    #            ('You, Me and Dupree', 3.1471)]), \
    #            recsys.recommend('Leopoldo Pires'))

    similarity = ItemSimilarity(boolean_model, jaccard_coefficient)
    #Empty Recommendation
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array(['You, Me and Dupree', 'Just My Luck']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array(['You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', similarity=similarity)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('Just My Luck', 1.0), \
    #            ('You, Me and Dupree', 1.0)]), \
    #            recsys.recommend('Leopoldo Pires'))

    similarity = ItemSimilarity(boolean_matrix_model, jaccard_coefficient)
    #Empty Recommendation
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['You, Me and Dupree', 'Just My Luck']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', similarity=similarity)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('Just My Luck', 3.20597), \
    #            ('You, Me and Dupree', 3.1471)]), \
    #            recsys.recommend('Leopoldo Pires'))


def test_recommend_because_ItemBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    #Full Recommendation Because
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck'))
    #over-items
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 20))
    #Semi
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #Non-Existing
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_array_equal(np.array([]), \
        recsys.recommended_because('Maria Gabriela', 'Just My Luck', 2))

    #with_preference
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy, True, True)
    assert_array_equal(np.array([('The Night Listener', 4.0), \
                ('Superman Returns', 3.5)]), \
                recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))
    #DictModel
    similarity = ItemSimilarity(dict_model, euclidean_distances)
    #Full Recommendation Because
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck'))
    #over-items
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 20))
    #Semi
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #Non-Existing
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy)
    assert_array_equal(np.array([]), \
        recsys.recommended_because('Maria Gabriela', 'Just My Luck', 2))

    #with_preference
    recsys = ItemBasedRecommender(dict_model, similarity, items_strategy, True, True)
    assert_array_equal(np.array([('The Night Listener', 4.0), \
                ('Superman Returns', 3.5)]), \
                recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #boolean_model
    similarity = ItemSimilarity(boolean_model, jaccard_coefficient)
    #Full Recommendation Because
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck'))
    #over-items
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 20))
    #Semi
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #Non-Existing
    recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy)
    assert_array_equal(np.array([]), \
        recsys.recommended_because('Maria Gabriela', 'Just My Luck', 2))

    #with_preference
    #recsys = ItemBasedRecommender(boolean_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('The Night Listener', 1.0), \
    #            ('Superman Returns', 1.0)]), \
    #            recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #boolean_matrix_model
    similarity = ItemSimilarity(boolean_matrix_model, jaccard_coefficient)
    #Full Recommendation Because
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck'))
    #over-items
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns', \
    'Snakes on a Plane', 'Lady in the Water']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 20))
    #Semi
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array(['The Night Listener', 'Superman Returns']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #Non-Existing
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_array_equal(np.array([]), \
        recsys.recommended_because('Maria Gabriela', 'Just My Luck', 2))

    #with_preference
    #recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('The Night Listener', 1.0), \
    #            ('Superman Returns', 1.0)]), \
    #            recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))
