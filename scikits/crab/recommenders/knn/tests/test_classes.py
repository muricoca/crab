import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from ....models.data_models import DictPreferenceDataModel, MatrixPreferenceDataModel
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


def test_create_ItemBasedRecommender():
    pass


'''

    def test_create_ItemBasedRecommender(self):
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertEquals(recSys.similarity,self.similarity)
        self.assertEquals(recSys.capper,True)
        self.assertEquals(recSys.strategy,self.strategy)
        self.assertEquals(recSys.model,self.model)


    def test_oneItem_mostSimilarItems(self):
        itemIDs = ['Superman Returns']
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertEquals(['Snakes on a Plane', 'The Night Listener', 'Lady in the Water', 'Just My Luck'],recSys.mostSimilarItems(itemIDs,4))

    def test_multipeItems_mostSimilarItems(self):
        itemIDs = ['Superman Returns','Just My Luck']
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertEquals(['Lady in the Water', 'Snakes on a Plane', 'The Night Listener', 'You, Me and Dupree'],recSys.mostSimilarItems(itemIDs,4))

    def test_semiItem_mostSimilarItems(self):
        itemIDs = ['Superman Returns','Just My Luck','Snakes on a Plane',  'The Night Listener',  'You, Me and Dupree']
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertEquals(['Lady in the Water'],recSys.mostSimilarItems(itemIDs,4))

    def test_allItem_mostSimilarItems(self):
        itemIDs = ['Superman Returns','Just My Luck','Snakes on a Plane',  'The Night Listener',  'You, Me and Dupree','Lady in the Water']
        recSys = ItemRecommender(self.model,self.similarity,self.strategy,True)
        self.assertEquals([],recSys.mostSimilarItems(itemIDs,4))


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
