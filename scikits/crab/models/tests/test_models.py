import numpy as np
from numpy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal, run_module_suite, TestCase

from nose.tools import assert_raises, assert_equals

from ..basic_models import DictPreferenceDataModel

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


def test_create_DictPreferenceDataModel():
	#Empty Dataset
	model = DictPreferenceDataModel({})
	assert_equals(model.dataset,{})
	
	assert_array_equal(np.array([]),model.user_ids())
	assert_array_equal(np.array([]),model.item_ids())
	assert_equals(True,model.has_preference_values())
	assert_equals(0,model.users_count())
	assert_equals(0,model.items_count())
	assert_equals(-np.inf,model.maximum_preference_value())
	assert_equals(np.inf,model.minimum_preference_value())
	
	
	#DataSet
	model = DictPreferenceDataModel(movies)
	assert_equals(model.dataset,movies)	
	assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', 'Luciana Nunes','Marcel Caraciolo', 'Maria Gabriela', 'Penny Frewman', 'Sheldom',
	       'Steve Gates'],  dtype='|S16'),model.user_ids())
	assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
		       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']),model.item_ids())
	assert_equals(True,model.has_preference_values())
	assert_equals(8,model.users_count())
	assert_equals(6,model.items_count())
	assert_equals(5.0,model.maximum_preference_value())
	assert_equals(1.0,model.minimum_preference_value())
	
