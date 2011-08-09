import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_equals, assert_almost_equals
from ....models.data_models import DictPreferenceDataModel, MatrixPreferenceDataModel
from ..item_strategies import ItemsNeighborhoodStrategy, AllPossibleItemsStrategy
from ....similarities.basic_similarities import ItemSimilarity
from ..classes import ItemBasedRecommender
from ....models.utils import UserNotFoundError, ItemNotFoundError
from ....metrics.pairwise import euclidean_distances


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


def test_estimate_preference_ItemBasedRecommender():
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(matrix_model, euclidean_distances)
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy)
    assert_almost_equals(3.5, recsys.estimate_preference('Marcel Caraciolo', 'Superman Returns'))
    assert_almost_equals(3.14717875510, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #With capper = False
    recsys = ItemBasedRecommender(matrix_model, similarity, items_strategy, False)
    #assert_almost_equals(3.14717875510, recsys.estimate_preference('Leopoldo Pires', 'You, Me and Dupree'))
    #Non-Preferences
    #assert_array_equal(np.nan, recsys.estimate_preference('Maria Gabriela', 'You, Me and Dupree'))


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



'''


    def test_local_estimatePreference(self):
        userID = 'Marcel Caraciolo'
        itemID = 'Superman Returns'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertAlmostEquals(3.5,recSys.estimatePreference(userID=userID,similarity=self.similarity,itemID=itemID))


    def test_local_not_existing_estimatePreference(self):
        userID = 'Leopoldo Pires'
        itemID = 'You, Me and Dupree'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertAlmostEquals(3.14717875510,recSys.estimatePreference(userID=userID,similarity=self.similarity,itemID=itemID))


    def test_local_not_existing_capper_False_estimatePreference(self):
        userID = 'Leopoldo Pires'
        itemID = 'You, Me and Dupree'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        self.assertAlmostEquals(3.14717875510,recSys.estimatePreference(userID=userID,similarity=self.similarity,itemID=itemID))


    def test_local_not_existing_rescorer_estimatePreference(self):
        userID = 'Leopoldo Pires'
        itemID = 'You, Me and Dupree'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        scorer = TanHScorer()
        self.assertAlmostEquals(3.1471787551,recSys.estimatePreference(userID=userID,similarity=self.similarity,itemID=itemID,rescorer=scorer))


    def test_empty_recommend(self):
        userID = 'Marcel Caraciolo'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        self.assertEquals([],recSys.recommend(userID,4))


    def test_recommend(self):
        userID = 'Leopoldo Pires'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        self.assertEquals(['Just My Luck', 'You, Me and Dupree'],recSys.recommend(userID,4))


    def test_full_recommend(self):
        userID = 'Maria Gabriela'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        self.assertEquals([],recSys.recommend(userID,4))


    def test_semi_recommend(self):
        userID = 'Leopoldo Pires'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        self.assertEquals(['Just My Luck'],recSys.recommend(userID,1))


    def test_recommendedBecause(self):
        userID = 'Leopoldo Pires'
        itemID = 'Just My Luck'
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,False)
        self.assertEquals(['The Night Listener', 'Superman Returns'],recSys.recommendedBecause(userID,itemID,2))
'''
