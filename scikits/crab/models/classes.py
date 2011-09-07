#-*- coding:utf-8 -*-

"""
Several Basic Data models.

"""
# Authors: Marcel Caraciolo <marcel@muricoca.com>
# License: BSD Style


import numpy as np
from .base import BaseDataModel
from .utils import UserNotFoundError, ItemNotFoundError
import logging

logger = logging.getLogger('crab')


###############################################################################
# MatrixDataModel
class MatrixPreferenceDataModel(BaseDataModel):
    '''
    Matrix with preferences based Data model
    A DataModel backed by a python dict structured data.
    This class expects a simple dictionary where each
    element contains a userID, followed by itemID,
    followed by preference value and optional timestamp.

    {userID:{itemID:preference, itemID2:preference2},
       userID2:{itemID:preference3,itemID4:preference5}}

    Preference value is the parameter that the user simply
     expresses the degree of preference for an item.

    Parameters
    ----------
    dataset dict, shape  = {userID:{itemID:preference, itemID2:preference2},
              userID2:{itemID:preference3,itemID4:preference5}}

    Examples
    ---------
    >>> from scikits.crab.models.classes import MatrixPreferenceDataModel
    >>> model = MatrixPreferenceDataModel({})
    >>> #empty dataset
    >>> model.user_ids()
    array([], dtype=float64)
    >>> model.item_ids()
    array([], dtype=float64)
    >>> movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, \
     'Snakes on a Plane': 3.5, \
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
     'The Night Listener': 3.0}, \
     'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
     'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
     'The Night Listener': 4.5, 'Superman Returns': 4.0, \
     'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'The Night Listener': 3.0, 'Superman Returns': 5.0, \
     'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {}}
    >>> model = MatrixPreferenceDataModel(movies)
    >>> #non-empty dataset
    >>> model.user_ids()
    array(['Leopoldo Pires', 'Lorena Abreu', 'Marcel Caraciolo',
               'Maria Gabriela', 'Paola Pow', 'Penny Frewman', 'Sheldom',
               'Steve Gates'],
              dtype='|S16')
    >>> model.item_ids()
    array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
               'Superman Returns', 'The Night Listener', 'You, Me and Dupree'],
              dtype='|S18')
    >>> model.preferences_from_user('Sheldom')
    [('Lady in the Water', 3.0), ('Snakes on a Plane', 4.0), ('Superman Returns', 5.0),
        ('The Night Listener', 3.0), ('You, Me and Dupree', 3.5)]
    '''
    def __init__(self, dataset):
        BaseDataModel.__init__(self)
        self.dataset = dataset
        self.build_model()

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self):
        '''
        Returns
        -------
        self:
             Build the data model
        '''
        #Is it important to store as numpy array ?
        self._user_ids = np.asanyarray(self.dataset.keys())
        self._user_ids.sort()

        #Is it important to store as numpy array ?
        self._item_ids = []
        for items in self.dataset.itervalues():
            self._item_ids.extend(items.keys())

        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()

        self.max_pref = -np.inf
        self.min_pref = np.inf

        logger.info("creating matrix for %d users and %d items" % \
                    (self._user_ids.size, self._item_ids.size))

        self.index = np.empty(shape=(self._user_ids.size, self._item_ids.size))
        for userno, user_id in enumerate(self._user_ids):
            if userno % 2 == 0:
                logger.debug("PROGRESS: at user_id #%i/%i" %  \
                    (userno, self._user_ids.size))
            for itemno, item_id in enumerate(self._item_ids):
                r = self.dataset[user_id].get(item_id, np.NaN) #Is it to be np.NaN or 0 ?!!
                self.index[userno, itemno] = r

        if self.index.size:
            self.max_pref = np.nanmax(self.index)
            self.min_pref = np.nanmin(self.index)

    def user_ids(self):
        '''
        Returns
        -------
        self.user_ids:  numpy array of shape [n_user_ids]
                        Return all user ids in the model, in order
        '''
        return self._user_ids

    def item_ids(self):
        '''
        Returns
        -------
        self.item_ids:  numpy array of shape [n_item_ids]
                    Return all item ids in the model, in order
        '''
        return self._item_ids

    def preference_values_from_user(self, user_id):
        '''
        Returns
        --------
        Return user's preferences values as an array

        Notes
        --------
        This method is a particular method in MatrixDataModel
        '''
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            #user_id not found
            raise UserNotFoundError

        preferences = self.index[user_id_loc]

        return preferences

    def preferences_from_user(self, user_id, order_by_id=True):
        '''
        Returns
        -------
        self.user_preferences :  list [(item_id,preference)]
         Return user's preferences, ordered by user ID (if order_by_id is True)
         or by the preference values (if order_by_id is False), as an array.

        '''
        preferences = self.preference_values_from_user(user_id)

        #think in a way to return as numpy array and how to remove the nan values efficiently.
        data = zip(self._item_ids, preferences.flatten())

        if order_by_id:
            return [(item_id, preference)  for item_id, preference in data \
                         if not np.isnan(preference)]
        else:
            return sorted([(item_id, preference)  for item_id, preference in data \
                         if not np.isnan(preference)], key=lambda item: - item[1])

    def has_preference_values(self):
        '''
        Returns
        -------
        True/False:  bool
                     Return True if this implementation actually
                     it is not a 'boolean' data model, otherwise returns False.
        '''
        return True

    def maximum_preference_value(self):
        '''
        Returns
        ---------
        self.max_preference:  float
                Return the maximum preference value that is possible in the
                 current problem domain being evaluated.
        '''
        return self.max_pref

    def minimum_preference_value(self):
        '''
        Returns
        ---------
        self.min_preference:  float
                Returns the minimum preference value that is possible in the
                current problem domain being evaluated
        '''
        return self.min_pref

    def users_count(self):
        '''
        Returns
        --------
        n_users:  int
                  Return total number of users known to the model.
        '''
        return self._user_ids.size

    def items_count(self):
        '''
        Returns
        --------
        n_items:  int
                  Return total number of items known to the model.
        '''
        return self._item_ids.size

    def items_from_user(self, user_id):
        '''
        Returns
        -------
        items_from_user : numpy array of shape [item_id,..]
                 Return IDs of items user expresses a preference for
        '''
        preferences = self.preferences_from_user(user_id)
        return [key for key, value in preferences]

    def preferences_for_item(self, item_id, order_by_id=True):
        '''
        Returns
        -------
        preferences: numpy array of shape [(item_id,preference)]
                     Return all existing Preferences expressed for that item,
        '''
        item_id_loc = np.where(self._item_ids == item_id)
        if not item_id_loc[0].size:
            #item_id not found
            raise ItemNotFoundError('Item not found')
        preferences = self.index[:, item_id_loc]

        #think in a way to return as numpy array and how to remove the nan values efficiently.
        data = zip(self._user_ids, preferences.flatten())
        if order_by_id:
            return [(user_id, preference)  for user_id, preference in data \
                         if not np.isnan(preference)]
        else:
            return sorted([(user_id, preference)  for user_id, preference in data \
                         if not np.isnan(preference)], key=lambda user: - user[1])

    def preference_value(self, user_id, item_id):
        '''
        Returns
        -------
        preference:  float
                     Retrieves the preference value for a single user and item.
        '''
        item_id_loc = np.where(self._item_ids == item_id)
        user_id_loc = np.where(self._user_ids == user_id)

        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')

        return self.index[user_id_loc, item_id_loc].flatten()[0]

    def set_preference(self, user_id, item_id, value):
        '''
        Returns
        --------
        self
            Sets a particular preference (item plus rating) for a user.
        '''
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        #ALLOW NEW ITEMS
        #if not item_id_loc[0].size:
        #    raise ItemNotFoundError('item_id in the model not found')

        #How not use the dataset in memory ?!
        self.dataset[user_id][item_id] = value
        self.build_model()

    def remove_preference(self, user_id, item_id):
        '''
        Returns
        --------
        self
            Removes a particular preference for a user.
        '''
        user_id_loc = np.where(self._user_ids == user_id)
        item_id_loc = np.where(self._item_ids == item_id)

        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')

        del self.dataset[user_id][item_id]
        self.build_model()

    def __repr__(self):
        return "<MatrixPreferenceDataModel (%d by %d)>" % (self.index.shape[0],
                        self.index.shape[1])

    def _repr_matrix(self, matrix):
        s = ""
        cellWidth = 11
        shape = matrix.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    s += "---".center(cellWidth)
                else:
                    exp = np.log(abs(v))
                    if abs(exp) <= 4:
                        if exp < 0:
                            s += ("%9.6f" % v).ljust(cellWidth)
                        else:
                            s += ("%9.*f" % (6, v)).ljust(cellWidth)
                    else:
                        s += ("%9.2e" % v).ljust(cellWidth)
            s += "\n"
        return s[:-1]

    def __unicode__(self):
        """
        Write out a representative picture of this matrix.

        The upper left corner of the matrix will be shown, with up to 20x5
        entries, and the rows and columns will be labeled with up to 8
        characters.
        """
        matrix = self._repr_matrix(self.index[:20, :5])
        lines = matrix.split('\n')
        headers = [repr(self)[1:-1]]
        if self._item_ids.size:
            col_headers = [('%-8s' % item[:8]) for item in self._item_ids[:5]]
            headers.append(' ' + ('   '.join(col_headers)))

        if self._user_ids.size:
            for (i, line) in enumerate(lines):
                lines[i] = ('%-8s' % self._user_ids[i][:8]) + line
            for (i, line) in enumerate(headers):
                if i > 0:
                    headers[i] = ' ' * 8 + line
        lines = headers + lines
        if self.index.shape[1] > 5 and self.index.shape[0] > 0:
            lines[1] += ' ...'
        if self.index.shape[0] > 20:
            lines.append('...')

        return '\n'.join(line.rstrip() for line in lines)

    def __str__(self):
        return unicode(self).encode('utf-8')


###############################################################################
# MatrixBooleanDataModel
class MatrixBooleanPrefDataModel(BaseDataModel):
    '''
    Matrix with preferences based Boolean Data model
    This class expects a simple dictionary where each
    element contains a userID, followed by the itemIDs
    where the itemIDs represents the preference
    for that item and optional timestamp. It also can
    receive the dict with the preference values used
    at DictPreferenceDataModel.

    Preference value is the presence of the item in the list of
    preferences for that user.

    Parameters
    ----------
    dataset dict, shape  = {userID:{itemID:preference, itemID2:preference2},
              userID2:{itemID:preference3,itemID4:preference5}} or
                  {userID:[itemID,itemID2,itemID3], userID2:[itemID1, itemID2,...]...}

    Examples
    ---------
    >>> from scikits.crab.models.classes import MatrixBooleanPrefDataModel
    >>> model = MatrixBooleanPrefDataModel({})
    >>> #empty dataset
    >>> model.user_ids()
    array([], dtype=float64)
    >>> model.item_ids()
    array([], dtype=float64)
    >>> movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, \
     'Snakes on a Plane': 3.5, \
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
     'The Night Listener': 3.0}, \
     'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
     'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
     'The Night Listener': 4.5, 'Superman Returns': 4.0, \
     'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'The Night Listener': 3.0, 'Superman Returns': 5.0, \
     'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {}}
    >>> model = MatrixBooleanPrefDataModel(movies)
    >>> #non-empty dataset
    >>> model.user_ids()
    array(['Leopoldo Pires', 'Lorena Abreu', 'Marcel Caraciolo',
               'Maria Gabriela', 'Paola Pow', 'Penny Frewman', 'Sheldom',
               'Steve Gates'],
              dtype='|S16')
    >>> model.item_ids()
    array(['Just My Luck', 'Lady in the Water', 'Snakes on a Plane',
               'Superman Returns', 'The Night Listener', 'You, Me and Dupree'],
              dtype='|S18')
    >>> model.preferences_from_user('Sheldom')
    array(['Lady in the Water', 'Snakes on a Plane', 'Superman Returns',
           'The Night Listener', 'You, Me and Dupree'],
          dtype='|S18')
    '''
    def __init__(self, dataset):
        BaseDataModel.__init__(self)
        self.dataset = self._load_dataset(dataset.copy())
        self.build_model()

    def _load_dataset(self, dataset):
        '''
        Returns
        -------
        dataset: dict of shape {user_id:[item_id,item_id2,...]}

        Load the dataset which the input can be the
        {user_id:{item_id:preference,...},...}
        or the {user_id:[item_id,item_id2,...],...}
        '''
        if dataset:
            key = dataset.keys()[0]
            if isinstance(dataset[key], dict):
                for key in dataset:
                    dataset[key] = dataset[key].keys()

        return dataset

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self):
        '''
        Returns
        -------
        self:
             Build the data model
        '''

        self._user_ids = np.asanyarray(self.dataset.keys())
        self._user_ids.sort()

        self._item_ids = np.array([])
        for items in self.dataset.itervalues():
            self._item_ids = np.append(self._item_ids, items)

        self._item_ids = np.unique(self._item_ids)
        self._item_ids.sort()

        logger.info("creating matrix for %d users and %d items" % \
                    (self._user_ids.size, self._item_ids.size))

        self.index = np.empty(shape=(self._user_ids.size, self._item_ids.size), dtype=bool)
        for userno, user_id in enumerate(self._user_ids):
            if userno % 2 == 0:
                logger.debug("PROGRESS: at user_id #%i/%i" %  \
                    (userno, self._user_ids.size))
            for itemno, item_id in enumerate(self._item_ids):
                r = True if item_id in self.dataset[user_id] else False
                self.index[userno, itemno] = r

    def user_ids(self):
        '''
        Returns
        -------
        self.user_ids:  numpy array of shape [n_user_ids]
                        Return all user ids in the model, in order
        '''
        return self._user_ids

    def item_ids(self):
        '''
        Returns
        -------
        self.item_ids:  numpy array of shape [n_item_ids]
                    Return all item ids in the model, in order
        '''
        return self._item_ids

    def preference_values_from_user(self, user_id):
        '''
        Returns
        --------
        Return user's preferences values as an array

        Notes
        --------
        This method is a particular method in MatrixDataModel
        '''
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            #user_id not found
            raise UserNotFoundError

        preferences = self.index[user_id_loc]

        return preferences

    def preferences_from_user(self, user_id, order_by_id=True):
        '''
        Returns
        -------
        self.user_preferences :  list [(item_id,preference)]
         Return user's preferences, ordered by user ID (if order_by_id is True)
         or by the preference values (if order_by_id is False), as an array.

        '''
        preferences = self.preference_values_from_user(user_id)

        preferences = preferences.flatten()

        return self._item_ids[preferences]

    def has_preference_values(self):
        '''
        Returns
        -------
        True/False:  bool
                     Return True if this implementation actually
                     it is not a 'boolean' data model, otherwise returns False.
        '''
        return False

    def users_count(self):
        '''
        Returns
        --------
        n_users:  int
                  Return total number of users known to the model.
        '''
        return self._user_ids.size

    def items_count(self):
        '''
        Returns
        --------
        n_items:  int
                  Return total number of items known to the model.
        '''
        return self._item_ids.size

    def items_from_user(self, user_id):
        '''
        Returns
        -------
        items_from_user : numpy array of shape [item_id,..]
                 Return IDs of items user expresses a preference for
        '''
        preferences = self.preferences_from_user(user_id)
        return preferences

    def preferences_for_item(self, item_id, order_by_id=True):
        '''
        Returns
        -------
        preferences: numpy array of shape [(item_id,preference)]
                     Return all existing Preferences expressed for that item,
        '''
        item_id_loc = np.where(self._item_ids == item_id)
        if not item_id_loc[0].size:
            #item_id not found
            raise ItemNotFoundError('Item not found')
        preferences = self.index[:, item_id_loc]

        preferences = preferences.flatten()

        return self._user_ids[preferences]

    def preference_value(self, user_id, item_id):
        '''
        Returns
        -------
        preference:  float
                     Retrieves the preference value for a single user and item.
        '''
        item_id_loc = np.where(self._item_ids == item_id)
        user_id_loc = np.where(self._user_ids == user_id)

        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')

        return 1.0 if self.index[user_id_loc, item_id_loc].flatten()[0] else 0.0

    def set_preference(self, user_id, item_id, value=None):
        '''
        Returns
        --------
        self
            Sets a particular preference (item plus rating) for a user.
        '''
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        #ALLOW NEW ITEMS
        #if not item_id_loc[0].size:
        #    raise ItemNotFoundError('item_id in the model not found')

        #How not use the dataset in memory ?!
        self.dataset[user_id].append(item_id)
        self.build_model()

    def remove_preference(self, user_id, item_id):
        '''
        Returns
        --------
        self
            Removes a particular preference for a user.
        '''
        user_id_loc = np.where(self._user_ids == user_id)
        item_id_loc = np.where(self._item_ids == item_id)

        if not user_id_loc[0].size:
            raise UserNotFoundError('user_id in the model not found')

        if not item_id_loc[0].size:
            raise ItemNotFoundError('item_id in the model not found')

        self.dataset[user_id].remove(item_id)
        self.build_model()

    def __repr__(self):
        return "<MatrixBooleanPrefDataModel (%d by %d)>" % (self.index.shape[0],
                        self.index.shape[1])

    def _repr_matrix(self, matrix):
        s = ""
        cellWidth = 11
        shape = matrix.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = matrix[i, j]
                if not v:
                    s += "---".center(cellWidth)
                else:
                    exp = np.log(abs(v))
                    if abs(exp) <= 4:
                        if exp < 0:
                            s += ("%9.6f" % v).ljust(cellWidth)
                        else:
                            s += ("%9.*f" % (6, v)).ljust(cellWidth)
                    else:
                        s += ("%9.2e" % v).ljust(cellWidth)
            s += "\n"
        return s[:-1]

    def __unicode__(self):
        """
        Write out a representative picture of this matrix.

        The upper left corner of the matrix will be shown, with up to 20x5
        entries, and the rows and columns will be labeled with up to 8
        characters.
        """
        matrix = self._repr_matrix(self.index[:20, :5])
        lines = matrix.split('\n')
        headers = [repr(self)[1:-1]]
        if self._item_ids.size:
            col_headers = [('%-8s' % item[:8]) for item in self._item_ids[:5]]
            headers.append(' ' + ('   '.join(col_headers)))

        if self._user_ids.size:
            for (i, line) in enumerate(lines):
                lines[i] = ('%-8s' % self._user_ids[i][:8]) + line
            for (i, line) in enumerate(headers):
                if i > 0:
                    headers[i] = ' ' * 8 + line
        lines = headers + lines
        if self.index.shape[1] > 5 and self.index.shape[0] > 0:
            lines[1] += ' ...'
        if self.index.shape[0] > 20:
            lines.append('...')

        return '\n'.join(line.rstrip() for line in lines)

    def __str__(self):
        return unicode(self).encode('utf-8')
