import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises, assert_equals

from ..classes import  MatrixPreferenceDataModel,  \
             MatrixBooleanPrefDataModel
from ..utils import UserNotFoundError, ItemNotFoundError

#Simple Movies DataSet
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


def test_basic_methods_MatrixPreferenceDataModel():
    #Empty Dataset
    model = MatrixPreferenceDataModel({})
    assert_equals(model.dataset, {})
    assert_array_equal(np.array([]), model.user_ids())
    assert_array_equal(np.array([]), model.item_ids())
    assert_equals(True, model.has_preference_values())
    assert_equals(0, model.users_count())
    assert_equals(0, model.items_count())
    assert_equals(-np.inf, model.maximum_preference_value())
    assert_equals(np.inf, model.minimum_preference_value())

    assert("MatrixPreferenceDataModel (0 by 0)" in model.__str__())

    #Simple DataSet
    model = MatrixPreferenceDataModel(movies)
    assert_equals(model.dataset, movies)
    assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', 'Luciana Nunes',
      'Marcel Caraciolo', 'Maria Gabriela', 'Penny Frewman', 'Sheldom', 'Steve Gates']),
      model.user_ids())
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
               'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), model.item_ids())
    assert_equals(True, model.has_preference_values())
    assert_equals(8, model.users_count())
    assert_equals(6, model.items_count())
    assert_equals(5.0, model.maximum_preference_value())
    assert_equals(1.0, model.minimum_preference_value())
    assert_equals([('Just My Luck', 3.0), ('Lady in the Water', 2.5),
             ('Snakes on a Plane', 3.5), ('Superman Returns', 3.5),
             ('The Night Listener', 3.0), ('You, Me and Dupree', 2.5)], model['Marcel Caraciolo'])
    elements = [pref  for pref in model]
    assert_array_equal([('Lady in the Water', 2.5), ('Snakes on a Plane', 3.0), \
         ('Superman Returns', 3.5), ('The Night Listener', 4.0)], elements[0][1])

    assert("MatrixPreferenceDataModel (8 by 6)" in model.__str__())


def test_preferences_from_user_exists_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    #ordered by item_id
    assert_array_equal(np.array([('Just My Luck', 3.0), ('Snakes on a Plane', 3.5),
       ('Superman Returns', 4.0), ('The Night Listener', 4.5), ('You, Me and Dupree', 2.5)]),
        model.preferences_from_user('Lorena Abreu'))

    #ordered by rating (reverse)
    assert_array_equal(np.array([('The Night Listener', 4.5), ('Superman Returns', 4.0), \
       ('Snakes on a Plane', 3.5), ('Just My Luck', 3.0), ('You, Me and Dupree', 2.5)]), \
          model.preferences_from_user('Lorena Abreu', order_by_id=False))


def test_preferences_from_user_exists_no_preferences_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_array_equal(np.array([]), model.preferences_from_user('Maria Gabriela'))


def test_preferences_from_user_non_existing_user_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_raises(UserNotFoundError, model.preferences_from_user, 'Flavia')


def test_item_ids_from_user_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
           'Superman Returns', 'The Night Listener', 'You, Me and Dupree']),
      model.items_from_user('Marcel Caraciolo'))


def test_preferences_for_item_existing_item_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    #ordered by item_id
    assert_array_equal(np.array([('Leopoldo Pires', 3.5), ('Lorena Abreu', 4.0), \
           ('Luciana Nunes', 5.0), ('Marcel Caraciolo', 3.5), \
           ('Penny Frewman', 4.0), ('Sheldom', 5.0), ('Steve Gates', 3.0)]),
       model.preferences_for_item('Superman Returns'))
    #ordered by rating (reverse)
    assert_array_equal(np.array([('Luciana Nunes', 5.0), ('Sheldom', 5.0), ('Lorena Abreu', 4.0), \
           ('Penny Frewman', 4.0), ('Leopoldo Pires', 3.5), \
           ('Marcel Caraciolo', 3.5), ('Steve Gates', 3.0)]),
           model.preferences_for_item('Superman Returns', order_by_id=False))


def test_preferences_for_item_existing_item_no_preferences_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_array_equal(np.array([]), model.preferences_for_item, 'The Night Listener')


def test_preferences_for_item_non_existing_item_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_raises(ItemNotFoundError, model.preferences_for_item, 'Back to the future')


def test_preference_value_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_equals(3.5, model.preference_value('Marcel Caraciolo', 'Superman Returns'))


def test_preference_value__invalid_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    assert_raises(UserNotFoundError, model.preference_value, 'Flavia', 'Superman Returns')
    assert_raises(ItemNotFoundError, model.preference_value, 'Marcel Caraciolo', 'Back to the future')
    assert_array_equal(np.nan, model.preference_value('Maria Gabriela', 'The Night Listener'))


def test_set_preference_value_MatrixPreferenceDataModel():
    #Add
    model = MatrixPreferenceDataModel(movies)
    model.set_preference('Maria Gabriela', 'Superman Returns', 2.0)
    assert_equals(2.0, model.preference_value('Maria Gabriela', 'Superman Returns'))
    #Edit
    model = MatrixPreferenceDataModel(movies)
    model.set_preference('Marcel Caraciolo', 'Superman Returns', 1.0)
    assert_equals(1.0, model.preference_value('Marcel Caraciolo', 'Superman Returns'))
    #invalid
    assert_raises(UserNotFoundError, model.set_preference, 'Carlos', 'Superman Returns', 2.0)
    #assert_raises(ItemNotFoundError,model.set_preference,'Marcel Caraciolo','Indiana Jones', 1.0)


def test_remove_preference_value_MatrixPreferenceDataModel():
    model = MatrixPreferenceDataModel(movies)
    model.remove_preference('Maria Gabriela', 'Superman Returns')
    assert_array_equal(np.nan, model.preference_value('Maria Gabriela', 'Superman Returns'))
    assert_raises(ItemNotFoundError, model.remove_preference, 'Marcel Caraciolo', 'Indiana Jones')

movies_boolean = {
'Marcel Caraciolo': ['Lady in the Water', 'Snakes on a Plane',
 'Just My Luck', 'Superman Returns', 'You, Me and Dupree',
 'The Night Listener'],
'Luciana Nunes': ['Lady in the Water', 'Snakes on a Plane',
 'Just My Luck', 'Superman Returns', 'The Night Listener',
 'You, Me and Dupree'],
'Leopoldo Pires': ['Lady in the Water', 'Snakes on a Plane',
 'Superman Returns', 'The Night Listener'],
'Lorena Abreu': ['Snakes on a Plane', 'Just My Luck',
 'The Night Listener', 'Superman Returns',
 'You, Me and Dupree'],
'Steve Gates': ['Lady in the Water', 'Snakes on a Plane',
 'Just My Luck', 'Superman Returns', 'The Night Listener',
 'You, Me and Dupree'],
'Sheldom': ['Lady in the Water', 'Snakes on a Plane',
 'The Night Listener', 'Superman Returns', 'You, Me and Dupree'],
'Penny Frewman': ['Snakes on a Plane', 'You, Me and Dupree', 'Superman Returns'],
'Maria Gabriela': []
}


def test_basic_methods_MatrixBooleanPrefDataModel():
    #Empty Dataset
    model = MatrixBooleanPrefDataModel({})
    assert_equals(model.dataset, {})
    assert_array_equal(np.array([]), model.user_ids())
    assert_array_equal(np.array([]), model.item_ids())
    assert_equals(False, model.has_preference_values())
    assert_equals(0, model.users_count())
    assert_equals(0, model.items_count())
    assert("MatrixBooleanPrefDataModel (0 by 0)" in model.__str__())

    #Simple DataSet
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_equals(model.dataset, movies_boolean)
    assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', 'Luciana Nunes',
      'Marcel Caraciolo', 'Maria Gabriela', 'Penny Frewman', 'Sheldom', 'Steve Gates']),
      model.user_ids())
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
               'Superman Returns', 'The Night Listener', 'You, Me and Dupree']), model.item_ids())
    assert_equals(False, model.has_preference_values())
    assert_equals(8, model.users_count())
    assert_equals(6, model.items_count())
    assert_array_equal(['Just My Luck', 'Lady in the Water',
             'Snakes on a Plane', 'Superman Returns',
             'The Night Listener', 'You, Me and Dupree'], model['Marcel Caraciolo'])
    elements = [pref  for pref in model]
    assert_array_equal(['Lady in the Water', 'Snakes on a Plane', \
         'Superman Returns', 'The Night Listener'], elements[0][1])
    assert("MatrixBooleanPrefDataModel (8 by 6)" in model.__str__())


def test_preferences_from_user_exists_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    #ordered by item_id
    assert_array_equal(np.array(['Just My Luck', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']),
        model.preferences_from_user('Lorena Abreu'))

    #ordered by rating (reverse)
    assert_array_equal(np.array(['Just My Luck', 'Snakes on a Plane',
       'Superman Returns', 'The Night Listener', 'You, Me and Dupree']),
        model.preferences_from_user('Lorena Abreu', order_by_id=False))


def test_preferences_from_user_exists_no_preferences_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_array_equal(np.array([],
      dtype='|S18'), model.preferences_from_user('Maria Gabriela'))


def test_preferences_from_user_non_existing_user_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_raises(UserNotFoundError, model.preferences_from_user, 'Flavia')


def test_item_ids_from_user_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_array_equal(np.array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
           'Superman Returns', 'The Night Listener', 'You, Me and Dupree']),
      model.items_from_user('Marcel Caraciolo'))


def test_preferences_for_item_existing_item_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    #ordered by item_id
    assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', \
           'Luciana Nunes', 'Marcel Caraciolo', \
           'Penny Frewman', 'Sheldom', 'Steve Gates']),
       model.preferences_for_item('Superman Returns'))
    #ordered by rating (reverse)
    assert_array_equal(np.array(['Leopoldo Pires', 'Lorena Abreu', \
           'Luciana Nunes', 'Marcel Caraciolo', \
           'Penny Frewman', 'Sheldom', 'Steve Gates']),
                model.preferences_for_item('Superman Returns', order_by_id=False))


def test_preferences_for_item_existing_item_no_preferences_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_array_equal(np.array([]), model.preferences_for_item, 'The Night Listener')


def test_preferences_for_item_non_existing_item_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_raises(ItemNotFoundError, model.preferences_for_item, 'Back to the future')


def test_preference_value_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_equals(1.0, model.preference_value('Marcel Caraciolo', 'Superman Returns'))


def test_preference_value__invalid_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    assert_raises(UserNotFoundError, model.preference_value, 'Flavia', 'Superman Returns')
    assert_raises(ItemNotFoundError, model.preference_value, 'Marcel Caraciolo', 'Back to the future')
    assert_array_equal(0.0, model.preference_value('Maria Gabriela', 'The Night Listener'))


def test_set_preference_value_MatrixBooleanPrefDataModel():
    #Add
    model = MatrixBooleanPrefDataModel(movies_boolean)
    model.set_preference('Maria Gabriela', 'Superman Returns')
    assert_equals(1.0, model.preference_value('Maria Gabriela', 'Superman Returns'))
    #Edit
    model = MatrixBooleanPrefDataModel(movies_boolean)
    model.set_preference('Marcel Caraciolo', 'Superman Returns')
    assert_equals(1.0, model.preference_value('Marcel Caraciolo', 'Superman Returns'))
    #invalid
    assert_raises(UserNotFoundError, model.set_preference, 'Carlos', 'Superman Returns')
    #assert_raises(ItemNotFoundError,model.set_preference,'Marcel Caraciolo','Indiana Jones', 1.0)


def test_remove_preference_value_MatrixBooleanPrefDataModel():
    model = MatrixBooleanPrefDataModel(movies_boolean)
    model.remove_preference('Maria Gabriela', 'Superman Returns')
    assert_array_equal(0.0, model.preference_value('Maria Gabriela', 'Superman Returns'))
    assert_raises(ItemNotFoundError, model.remove_preference, 'Marcel Caraciolo', 'Indiana Jones')
