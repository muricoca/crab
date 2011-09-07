import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_equals, assert_almost_equals
from ....models.classes import  MatrixPreferenceDataModel, \
     MatrixBooleanPrefDataModel
from ..item_strategies import ItemsNeighborhoodStrategy, AllPossibleItemsStrategy
from ..neighborhood_strategies import AllNeighborsStrategy, NearestNeighborsStrategy
from ....similarities.basic_similarities import ItemSimilarity, UserSimilarity
from ..classes import ItemBasedRecommender, UserBasedRecommender
from ....models.utils import ItemNotFoundError, UserNotFoundError
from ....metrics.pairwise import euclidean_distances, jaccard_coefficient, pearson_correlation


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

matrix_model = MatrixPreferenceDataModel(movies)
boolean_matrix_model = MatrixBooleanPrefDataModel(movies)


def test_create_ItemBasedRecommender():
    items_strategy = AllPossibleItemsStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_equals(recsys.similarity, similarity)
    assert_equals(recsys.items_selection_strategy, items_strategy)
    assert_equals(recsys.model, matrix_model)
    assert_equals(recsys.capper, True)


def test_create_UserBasedRecommender():
    nhood_strategy = AllNeighborsStrategy()
    similarity = UserSimilarity(matrix_model, euclidean_distances)
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_equals(recsys.similarity, similarity)
    assert_equals(recsys.neighborhood_strategy, nhood_strategy)
    assert_equals(recsys.model, matrix_model)
    assert_equals(recsys.capper, True)


def test_all_other_items_ItemBasedRecommender():
    items_strategy = AllPossibleItemsStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)

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


def test_all_other_items_UserBasedRecommender():
    nhood_strategy = AllNeighborsStrategy()
    similarity = UserSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    similarity = UserSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Lady in the Water']),
        recsys.all_other_items(user_id='Lorena Abreu', distance=pearson_correlation, nhood_size=2, minimal_similarity=0.1))
    assert_array_equal(np.array([], dtype='|S'),
         recsys.all_other_items(user_id='Marcel Caraciolo', distance=pearson_correlation, nhood_size=2, minimal_similarity=0.1))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']),
        recsys.all_other_items(user_id='Maria Gabriela', distance=pearson_correlation, nhood_size=2, minimal_similarity=0.1))

    similarity = UserSimilarity(matrix_model, euclidean_distances)
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)

    assert_array_equal(np.array(['Lady in the Water']), recsys.all_other_items('Lorena Abreu'))
    assert_array_equal(np.array([], dtype='|S'), recsys.all_other_items('Marcel Caraciolo'))
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), recsys.all_other_items('Maria Gabriela'))

    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(matrix_model, pearson_correlation)
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)

    assert_array_equal(np.array(['Lady in the Water']),
        recsys.all_other_items(user_id='Lorena Abreu', distance=pearson_correlation, nhood_size=2, minimal_similarity=0.1))
    assert_array_equal(np.array([], dtype='|S'),
         recsys.all_other_items(user_id='Marcel Caraciolo', distance=pearson_correlation, nhood_size=3))
    assert_array_equal(np.array([]),
        recsys.all_other_items(user_id='Maria Gabriela', distance=euclidean_distances, nhood_size=2))


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
    similarity = ItemSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy)
    assert_almost_equals(1.0, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #With capper = False
    recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy, False)
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    assert_array_equal(0.0, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))


def test_estimate_preference_UserBasedRecommender():
    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(matrix_model, euclidean_distances)
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(2.4533792305691886, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))

    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(2.8960083169728952,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree',
                distance=pearson_correlation, nhood_size=4, minimal_similarity=-1.0))

    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_almost_equals(2.0653946891716108,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree',
                 nhood_size=4))

    #With capper = False
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy, False)
    assert_almost_equals(2.0653946891716108, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    assert_almost_equals(2.8960083169728952,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree',
                distance=pearson_correlation, nhood_size=4, minimal_similarity=-1.0))

    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy, False)
    assert_almost_equals(2.0653946891716108,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree',
                 nhood_size=4))

    #Non-Preferences
    assert_array_equal(np.nan, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))

    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_almost_equals(1.0, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(0.0, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    assert_almost_equals(0.0,
         recsys.estimate_preference(user_id='Leopoldo Pires', item_id='You, Me and Dupree',
                distance=jaccard_coefficient, nhood_size=3))

    #With capper = False
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy, False)
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


def test_most_similar_users_UserBasedRecommender():
    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(matrix_model, euclidean_distances)
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    #semi items
    assert_array_equal(np.array(['Leopoldo Pires', 'Steve Gates', 'Lorena Abreu',
         'Penny Frewman']), \
            recsys.most_similar_users('Marcel Caraciolo', 4))
    #all items
    assert_array_equal(np.array(['Lorena Abreu', 'Marcel Caraciolo', 'Penny Frewman', \
    'Steve Gates', 'Luciana Nunes', 'Sheldom', 'Maria Gabriela']), \
            recsys.most_similar_users('Leopoldo Pires'))
    #Non-existing
    assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', 'Luciana Nunes',
       'Marcel Caraciolo', 'Penny Frewman', 'Sheldom', 'Steve Gates']), \
            recsys.most_similar_users('Maria Gabriela'))
    #Exceed the limit
    assert_array_equal(np.array(['Lorena Abreu', 'Marcel Caraciolo', 'Penny Frewman', \
    'Steve Gates', 'Luciana Nunes', 'Sheldom', 'Maria Gabriela']), \
            recsys.most_similar_users('Leopoldo Pires', 20))
    #Empty
    assert_array_equal(np.array([]), \
            recsys.most_similar_users('Sheldom', 0))

    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(boolean_matrix_model, jaccard_coefficient)
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    #semi items
    assert_array_equal(np.array(['Luciana Nunes', 'Steve Gates', \
            'Lorena Abreu', 'Sheldom']), \
            recsys.most_similar_users('Marcel Caraciolo', 4))
    #all items
    assert_array_equal(np.array(['Sheldom', 'Luciana Nunes', 'Marcel Caraciolo',
     'Steve Gates', 'Lorena Abreu', 'Penny Frewman', 'Maria Gabriela']), \
            recsys.most_similar_users('Leopoldo Pires'))
    #Non-existing
    assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', 'Luciana Nunes',
       'Marcel Caraciolo', 'Penny Frewman', 'Sheldom', 'Steve Gates']), \
            recsys.most_similar_users('Maria Gabriela'))
    #Exceed the limit
    assert_array_equal(np.array(['Sheldom', 'Luciana Nunes', 'Marcel Caraciolo',
     'Steve Gates', 'Lorena Abreu', 'Penny Frewman', 'Maria Gabriela']), \
            recsys.most_similar_users('Leopoldo Pires', 20))
    #Empty
    assert_array_equal(np.array([]), \
            recsys.most_similar_users('Sheldom', 0))


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


def test_recommend_UserBasedRecommender():
    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(matrix_model, euclidean_distances)
    #Empty Recommendation
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Just My Luck', 'You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Just My Luck']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #Test with params update
    recsys.recommend(user_id='Maria Gabriela', similarity=similarity)
    assert_array_equal(np.array([]), recsys.recommend('Maria Gabriela'))

    #with_preference
    #recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('Just My Luck', 3.20597319063), \
    #            ('You, Me and Dupree', 3.14717875510)]), \
    #            recsys.recommend('Leopoldo Pires'))

    similarity = UserSimilarity(boolean_matrix_model, jaccard_coefficient)
    #Empty Recommendation
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array([]), recsys.recommend('Marcel Caraciolo'))

    #Semi Recommendation
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['You, Me and Dupree', 'Just My Luck']), \
        recsys.recommend('Leopoldo Pires'))

    #Semi Recommendation
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['You, Me and Dupree']), \
        recsys.recommend('Leopoldo Pires', 1))

    #Empty Recommendation
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
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


def test_recommend_because_UserBasedRecommender():
    nhood_strategy = NearestNeighborsStrategy()
    similarity = UserSimilarity(matrix_model, euclidean_distances)
    #Full Recommendation Because
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Lorena Abreu', 'Marcel Caraciolo', \
        'Steve Gates', 'Luciana Nunes']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck'))
    #over-items
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Lorena Abreu', 'Marcel Caraciolo', \
        'Steve Gates', 'Luciana Nunes']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 20))
    #Semi
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Lorena Abreu', 'Marcel Caraciolo']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #Non-Existing
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array([]), \
        recsys.recommended_because('Maria Gabriela', 'Just My Luck', 2))

    #with_preference
    recsys = UserBasedRecommender(matrix_model, similarity, nhood_strategy, True, True)
    assert_array_equal(np.array([('Lorena Abreu', 3.0), \
                ('Marcel Caraciolo', 3.0)]), \
                recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #boolean_matrix_model
    similarity = UserSimilarity(boolean_matrix_model, jaccard_coefficient)
    #Full Recommendation Because
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Steve Gates', 'Marcel Caraciolo', 'Luciana Nunes', \
    'Lorena Abreu']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck'))
    #over-items
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Steve Gates', 'Marcel Caraciolo', 'Luciana Nunes', \
    'Lorena Abreu']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 20))
    #Semi
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array(['Steve Gates', 'Marcel Caraciolo']), \
        recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

    #Non-Existing
    recsys = UserBasedRecommender(boolean_matrix_model, similarity, nhood_strategy)
    assert_array_equal(np.array([]), \
        recsys.recommended_because('Maria Gabriela', 'Just My Luck', 2))

    #with_preference
    #recsys = ItemBasedRecommender(boolean_matrix_model, similarity, items_strategy, True, True)
    #assert_array_equal(np.array([('The Night Listener', 1.0), \
    #            ('Superman Returns', 1.0)]), \
    #            recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))

