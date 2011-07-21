import numpy as np
from numpy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal, run_module_suite, TestCase

from nose.tools import assert_raises, assert_equals

from ..basic_similarities import UserSimilarity, ItemSimilarity, find_common_elements
from ...metrics.pairwise import cosine_distances, pearson_correlation, euclidean_distances, manhattan_distances
from ...models.data_models import DictPreferenceDataModel

#Simple Movies DataSet

movies={'Marcel Caraciolo': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
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
'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0},
'Maria Gabriela': {}}

model = DictPreferenceDataModel(movies)

def test_find_common_elements():
	global model

	source_preferences = model.preferences_from_user('Marcel Caraciolo')
	target_preferences = model.preferences_from_user('Leopoldo Pires')
	assert_array_equal(np.array([[2.5,3.5,3.5,3.0]]),find_common_elements(source_preferences,target_preferences,'item_ids')[0])
	assert_array_equal(np.array([[2.5,3.0,3.5,4.0]]),find_common_elements(source_preferences,target_preferences, 'item_ids')[1])

	source_preferences = model.preferences_from_user('Marcel Caraciolo')
	target_preferences = model.preferences_from_user('Luciana Nunes')
	assert_array_equal( np.array([[ 3. ,  2.5,  3.5,  3.5,  3. ,  2.5]]),find_common_elements(source_preferences,target_preferences, 'item_ids')[0])
	assert_array_equal( np.array([[ 1.5,  3. ,  3.5,  5. ,  3. ,  3.5]]) ,find_common_elements(source_preferences,target_preferences,'item_ids')[1])

	source_preferences = model.preferences_from_user('Marcel Caraciolo')
	target_preferences = model.preferences_from_user('Maria Gabriela')
	assert_array_equal( np.array([[]]),find_common_elements(source_preferences,target_preferences,'item_ids')[0])
	assert_array_equal( np.array([[]]) ,find_common_elements(source_preferences,target_preferences, 'item_ids')[1])


	source_preferences = model.preferences_for_item('Snakes on a Plane')
	target_preferences = model.preferences_for_item('Superman Returns')
	assert_array_equal(np.array([[3. ,  3.5,  3.5,  3.5,  4.5,  4. ,  4.]]),find_common_elements(source_preferences,target_preferences,'user_ids')[0])
	assert_array_equal( np.array([[ 3.5,  4. ,  5. ,  3.5,  4. ,  5. ,  3. ]]) ,find_common_elements(source_preferences,target_preferences, 'user_ids')[1])

	model.set_preference('Maria Gabriela','Back to the Future', 3.5)

	source_preferences = model.preferences_for_item('Back to the Future')
	target_preferences = model.preferences_for_item('Superman Returns')
	assert_array_equal(np.array([[]]),find_common_elements(source_preferences,target_preferences,'user_ids')[0])
	assert_array_equal( np.array([[]]) ,find_common_elements(source_preferences,target_preferences, 'user_ids')[1])

	model = DictPreferenceDataModel(movies)


def test_get__item___UserSimilarity():
	#Cosine #With limits
	similarity = UserSimilarity(model,cosine_distances,3)

	#print similarity['Marcel Caraciolo']


	assert_array_equal(np.array([[1.]]), similarity['Marcel Caraciolo'][0][1])
	assert_equals('Marcel Caraciolo', similarity['Marcel Caraciolo'][0][0])

	assert_array_almost_equal(np.array([[0.99127583]]), similarity['Marcel Caraciolo'][1][1])
	assert_equals('Sheldom', similarity['Marcel Caraciolo'][1][0])

	assert_array_almost_equal(np.array([[0.98658676]]), similarity['Marcel Caraciolo'][2][1])
	assert_equals('Lorena Abreu', similarity['Marcel Caraciolo'][2][0])


	#Pearson Without limits
	similarity = UserSimilarity(model,pearson_correlation)

	#print similarity['Leopoldo Pires']

	assert_array_almost_equal(np.array([[1.]]), similarity['Leopoldo Pires'][0][1])
	assert_equals('Leopoldo Pires', similarity['Leopoldo Pires'][0][0])

	assert_array_almost_equal(np.array([[1.]]), similarity['Leopoldo Pires'][1][1])
	assert_equals('Lorena Abreu', similarity['Leopoldo Pires'][1][0])

	assert_array_almost_equal(np.array([[0.40451992]]), similarity['Leopoldo Pires'][2][1])
	assert_equals('Marcel Caraciolo', similarity['Leopoldo Pires'][2][0])

	assert_array_almost_equal(np.array([[ 0.2045983]]), similarity['Leopoldo Pires'][3][1])
	assert_equals('Luciana Nunes', similarity['Leopoldo Pires'][3][0])

	assert_array_almost_equal(np.array([[np.nan]]), similarity['Leopoldo Pires'][4][1])
	assert_equals('Maria Gabriela', similarity['Leopoldo Pires'][4][0])

	assert_array_almost_equal(np.array([[0.13483997]]), similarity['Leopoldo Pires'][5][1])
	assert_equals('Sheldom', similarity['Leopoldo Pires'][5][0])

	assert_array_almost_equal(np.array([[-0.25819889]]), similarity['Leopoldo Pires'][6][1])
	assert_equals('Steve Gates', similarity['Leopoldo Pires'][6][0])

	assert_array_almost_equal(np.array([[-1.]]), similarity['Leopoldo Pires'][7][1])
	assert_equals('Penny Frewman', similarity['Leopoldo Pires'][7][0])



	similarity = UserSimilarity(model,euclidean_distances)

	assert_array_equal(np.array([[1.]]), similarity['Steve Gates'][0][1])
	assert_equals('Steve Gates', similarity['Steve Gates'][0][0])

	assert_array_almost_equal(np.array([[0.41421356]]), similarity['Steve Gates'][1][1])
	assert_equals('Marcel Caraciolo', similarity['Steve Gates'][1][0])

	assert_array_almost_equal(np.array([[0.4]]), similarity['Steve Gates'][2][1])
	assert_equals('Penny Frewman', similarity['Steve Gates'][2][0])

	assert_array_almost_equal(np.array([[0.38742589]]), similarity['Steve Gates'][3][1])
	assert_equals('Leopoldo Pires', similarity['Steve Gates'][3][0])

	assert_array_almost_equal(np.array([[0.31451986]]), similarity['Steve Gates'][4][1])
	assert_equals('Lorena Abreu', similarity['Steve Gates'][4][0])

	assert_array_almost_equal(np.array([[0.2779263]]), similarity['Steve Gates'][5][1])
	assert_equals('Luciana Nunes', similarity['Steve Gates'][5][0])

	assert_array_almost_equal(np.array([[np.nan]]), similarity['Steve Gates'][6][1])
	assert_equals('Maria Gabriela', similarity['Steve Gates'][6][0])

	assert_array_almost_equal(np.array([[0.28571429]]), similarity['Steve Gates'][7][1])
	assert_equals('Sheldom', similarity['Steve Gates'][7][0])


	similarity = UserSimilarity(model,manhattan_distances,0)

	assert_equals([], similarity['Steve Gates'])

	similarity = UserSimilarity(model,manhattan_distances,20)
	
	assert_array_equal(np.array([[1.]]), similarity['Steve Gates'][0][1])
	assert_equals('Steve Gates', similarity['Steve Gates'][0][0])

	assert_array_almost_equal(np.array([[0.5]]), similarity['Steve Gates'][1][1])
	assert_equals('Marcel Caraciolo', similarity['Steve Gates'][1][0])

	assert_array_almost_equal(np.array([[0.3]]), similarity['Steve Gates'][2][1])
	assert_equals('Sheldom', similarity['Steve Gates'][2][0])

	assert_array_almost_equal(np.array([[0.25]]), similarity['Steve Gates'][3][1])
	assert_equals('Leopoldo Pires', similarity['Steve Gates'][3][0])

	assert_array_almost_equal(np.array([[0.25]]), similarity['Steve Gates'][4][1])
	assert_equals('Luciana Nunes', similarity['Steve Gates'][4][0])

	assert_array_almost_equal(np.array([[0.1]]), similarity['Steve Gates'][5][1])
	assert_equals('Lorena Abreu', similarity['Steve Gates'][5][0])

	assert_array_almost_equal(np.array([[np.nan]]), similarity['Steve Gates'][6][1])
	assert_equals('Maria Gabriela', similarity['Steve Gates'][6][0])

	assert_array_almost_equal(np.array([[0.16666667]]), similarity['Steve Gates'][7][1])
	assert_equals('Penny Frewman', similarity['Steve Gates'][7][0])


	similarity = UserSimilarity(model,euclidean_distances)

	assert_array_equal(np.array([[1.]]), similarity['Steve Gates'][0][1])
	assert_equals('Steve Gates', similarity['Steve Gates'][0][0])

	assert_array_almost_equal(np.array([[0.41421356]]), similarity['Steve Gates'][1][1])
	assert_equals('Marcel Caraciolo', similarity['Steve Gates'][1][0])

	assert_array_almost_equal(np.array([[0.4]]), similarity['Steve Gates'][2][1])
	assert_equals('Penny Frewman', similarity['Steve Gates'][2][0])

	assert_array_almost_equal(np.array([[0.38742589]]), similarity['Steve Gates'][3][1])
	assert_equals('Leopoldo Pires', similarity['Steve Gates'][3][0])

	assert_array_almost_equal(np.array([[0.31451986]]), similarity['Steve Gates'][4][1])
	assert_equals('Lorena Abreu', similarity['Steve Gates'][4][0])

	assert_array_almost_equal(np.array([[0.2779263]]), similarity['Steve Gates'][5][1])
	assert_equals('Luciana Nunes', similarity['Steve Gates'][5][0])

	assert_array_almost_equal(np.array([[np.nan]]), similarity['Steve Gates'][6][1])
	assert_equals('Maria Gabriela', similarity['Steve Gates'][6][0])

	assert_array_almost_equal(np.array([[0.28571429]]), similarity['Steve Gates'][7][1])
	assert_equals('Sheldom', similarity['Steve Gates'][7][0])


def test_get_similarities__UserSimilarity():

	similarity = UserSimilarity(model,cosine_distances,3)

	sim =  similarity.get_similarities('Marcel Caraciolo')

	assert_equals(len(sim), model.users_count())


	#Pearson Without limits
	similarity = UserSimilarity(model,pearson_correlation)

	sim =  similarity.get_similarities('Leopoldo Pires')

	assert_equals(len(sim), model.users_count())



	similarity = UserSimilarity(model,euclidean_distances)

	sim =  similarity.get_similarities('Steve Gates')

	assert_equals(len(sim), model.users_count())



	similarity = UserSimilarity(model,manhattan_distances,0)

	sim =  similarity.get_similarities('Steve Gates')

	assert_equals(len(sim), model.users_count())


	similarity = UserSimilarity(model,manhattan_distances,20)

	sim =  similarity.get_similarities('Steve Gates')

	assert_equals(len(sim), model.users_count())


def test__iter__UserSimilarity():
	similarity = UserSimilarity(model,cosine_distances,3)

	source_ids = []
	prefs = []
	for source_id,preferences in similarity:
		source_ids.append(source_id)
		prefs.append(preferences)
	assert_equals(len(source_ids),model.users_count())

	for pref in prefs:
		assert_equals(len(pref),3)


	similarity = UserSimilarity(model,pearson_correlation)

	source_ids = []
	prefs = []
	for source_id,preferences in similarity:
		source_ids.append(source_id)
		prefs.append(preferences)
	assert_equals(len(source_ids),model.users_count())

	for pref in prefs:
		assert_equals(len(pref),model.users_count())


	similarity = UserSimilarity(model,manhattan_distances,0)

	source_ids = []
	prefs = []
	for source_id,preferences in similarity:
		source_ids.append(source_id)
		prefs.append(preferences)
	assert_equals(len(source_ids),model.users_count())

	for pref in prefs:
		assert_equals(len(pref),0)


	similarity = UserSimilarity(model,manhattan_distances,20)

	source_ids = []
	prefs = []
	for source_id,preferences in similarity:
		source_ids.append(source_id)
		prefs.append(preferences)
	assert_equals(len(source_ids),model.users_count())

	for pref in prefs:
		assert_equals(len(pref),model.users_count())




def test_get__item___ItemSimilarity():
	#Cosine #With limits
	similarity = ItemSimilarity(model,cosine_distances,3)

	#assert_array_equal(np.array([[1.]]), similarity['Snakes on a Plane'][0][1])
	#assert_equals('Marcel Caraciolo', similarity['Snakes on a Plane'][0][0])

	assert_array_almost_equal(np.array([[1.]]), similarity['Snakes on a Plane'][1][1])
	assert_equals('Snakes on a Plane', similarity['Snakes on a Plane'][1][0])

	assert_array_almost_equal(np.array([[0.99773877]]), similarity['Snakes on a Plane'][2][1])
	assert_equals('Lady in the Water', similarity['Snakes on a Plane'][2][0])

	#Pearson Without limits
	similarity = ItemSimilarity(model,pearson_correlation)

	#assert_array_equal(np.array([[1.]]), similarity['The Night Listener'][0][1])
	#assert_equals('Leopoldo Pires', similarity['The Night Listener'][0][0])

	assert_array_almost_equal(np.array([[1.]]), similarity['The Night Listener'][1][1])
	assert_equals('The Night Listener', similarity['The Night Listener'][1][0])

	assert_array_almost_equal(np.array([[0.55555556]]), similarity['The Night Listener'][2][1])
	assert_equals('Just My Luck', similarity['The Night Listener'][2][0])

	assert_array_almost_equal(np.array([[ -0.17984719]]), similarity['The Night Listener'][3][1])
	assert_equals('Superman Returns', similarity['The Night Listener'][3][0])

	assert_array_almost_equal(np.array([[-0.25]]), similarity['The Night Listener'][4][1])
	assert_equals('You, Me and Dupree', similarity['The Night Listener'][4][0])

	assert_array_almost_equal(np.array([[-0.56635211]]), similarity['The Night Listener'][5][1])
	assert_equals('Snakes on a Plane', similarity['The Night Listener'][5][0])

	assert_array_almost_equal(np.array([[-0.61237244]]), similarity['The Night Listener'][6][1])
	assert_equals('Lady in the Water', similarity['The Night Listener'][6][0])


	similarity = ItemSimilarity(model,euclidean_distances)

	#assert_array_equal(np.array([[1.]]), similarity['The Night Listener'][0][1])
	#assert_equals('Steve Gates', similarity['The Night Listener'][0][0])

	assert_array_almost_equal(np.array([[1.]]), similarity['The Night Listener'][1][1])
	assert_equals('The Night Listener', similarity['The Night Listener'][1][0])

	assert_array_almost_equal(np.array([[0.38742589]]), similarity['The Night Listener'][2][1])
	assert_equals('Lady in the Water', similarity['The Night Listener'][2][0])

	assert_array_almost_equal(np.array([[0.32037724]]), similarity['The Night Listener'][3][1])
	assert_equals('Snakes on a Plane', similarity['The Night Listener'][3][0])

	assert_array_almost_equal(np.array([[0.29893508]]), similarity['The Night Listener'][4][1])
	assert_equals('Just My Luck', similarity['The Night Listener'][4][0])

	assert_array_almost_equal(np.array([[0.29429806]]), similarity['The Night Listener'][5][1])
	assert_equals('You, Me and Dupree', similarity['The Night Listener'][5][0])

	assert_array_almost_equal(np.array([[0.25265031]]), similarity['The Night Listener'][6][1])
	assert_equals('Superman Returns', similarity['The Night Listener'][6][0])


	similarity = ItemSimilarity(model,manhattan_distances,0)

	assert_equals([], similarity['Lady in the Water'])

	similarity = ItemSimilarity(model,manhattan_distances,20)

	assert_array_almost_equal(np.array([[1.]]), similarity['Snakes on a Plane'][1][1])
	assert_equals('Snakes on a Plane', similarity['Snakes on a Plane'][1][0])

	assert_array_almost_equal(np.array([[0.28571429]]), similarity['Snakes on a Plane'][2][1])
	assert_equals('Superman Returns', similarity['Snakes on a Plane'][2][0])

	assert_array_almost_equal(np.array([[0.2]]), similarity['Snakes on a Plane'][3][1])
	assert_equals('Lady in the Water', similarity['Snakes on a Plane'][3][0])

	assert_array_almost_equal(np.array([[0.16666667]]), similarity['Snakes on a Plane'][4][1])
	assert_equals('The Night Listener', similarity['Snakes on a Plane'][4][0])

	assert_array_almost_equal(np.array([[-0.25]]), similarity['Snakes on a Plane'][5][1])
	assert_equals('Just My Luck', similarity['Snakes on a Plane'][5][0])

	assert_array_almost_equal(np.array([[-0.33333333]]), similarity['Snakes on a Plane'][6][1])
	assert_equals('You, Me and Dupree', similarity['Snakes on a Plane'][6][0])



def test_get_similarities__ItemSimilarity():

	similarity = ItemSimilarity(model,cosine_distances,3)

	sim =  similarity.get_similarities('Snakes on a Plane')

	assert_equals(len(sim), model.items_count())

	#Pearson Without limits
	similarity = ItemSimilarity(model,pearson_correlation)

	sim =  similarity.get_similarities('Lady in the Water')

	assert_equals(len(sim), model.items_count())


	similarity = ItemSimilarity(model,euclidean_distances)

	sim =  similarity.get_similarities('Lady in the Water')

	assert_equals(len(sim), model.items_count())


	similarity = ItemSimilarity(model,manhattan_distances,0)

	sim =  similarity.get_similarities('Lady in the Water')

	assert_equals(len(sim), model.items_count())


	similarity = ItemSimilarity(model,manhattan_distances,20)

	sim =  similarity.get_similarities('Lady in the Water')

	assert_equals(len(sim), model.items_count())



def test__iter__ItemSimilarity():
	similarity = ItemSimilarity(model,cosine_distances,3)

	item_ids = []
	prefs = []
	for item_id,preferences in similarity:
		item_ids.append(item_id)
		prefs.append(preferences)
	assert_equals(len(item_ids),model.items_count())

	for pref in prefs:
		assert_equals(len(pref),3)


	similarity = ItemSimilarity(model,pearson_correlation)

	item_ids = []
	prefs = []
	for item_id,preferences in similarity:
		item_ids.append(item_id)
		prefs.append(preferences)
	assert_equals(len(item_ids),model.items_count())

	for pref in prefs:
		assert_equals(len(pref),model.items_count())


	similarity = ItemSimilarity(model,manhattan_distances,0)

	item_ids = []
	prefs = []
	for item_id,preferences in similarity:
		item_ids.append(item_id)
		prefs.append(preferences)
	assert_equals(len(item_ids),model.items_count())

	for pref in prefs:
		assert_equals(len(pref),0)


	similarity = ItemSimilarity(model,manhattan_distances,20)

	item_ids = []
	prefs = []
	for item_id,preferences in similarity:
		item_ids.append(item_id)
		prefs.append(preferences)
	assert_equals(len(item_ids),model.items_count())

	for pref in prefs:
		assert_equals(len(pref),model.items_count())
