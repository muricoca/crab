#-*- coding:utf-8 -*-

"""
Base Data Models.
"""
# Authors: Bruno Melo <bruno@muricoca.com>
# License: BSD Style


class BaseDataModel(object):

    def user_ids(self):
        '''
        Returns
        --------
        Return all user ids in the model, in order
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def item_ids(self):
        '''
        Returns
        -------
        Return a iterator of all item ids in the model, in order
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preferences_values_from_user(self, user_id, order_by_id=True):
        '''
        Parameters
        ----------
        user_id: user id in the model
                 int or string

        order_by_id: bool
                If True order by user_id otherwise by the preference values.
                default = True
        Returns
        ---------
        Return user's preferences only the values.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preferences_from_user(self, user_id, order_by_id=True):
        '''
        Parameters
        ----------
        user_id: user id in the model
                 int or string

        order_by_id: bool
                If True order by user_id otherwise by the preference values.
                default = True
        Returns
        ---------
        Return user's preferences, ordered by user id (if order_by_id is True)
        or by the preference values (if order_by_id is False), as an array.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def items_from_user(self, user_id):
        '''
        Parameters
        ----------
        user_id: user id in the model
                int or string

        Returns
        -------
        Return ids of items user expresses a preference for
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preferences_for_item(self, item_id, order_by_id=True):
        '''
        Parameters
        ----------
        item_id: id of the item in the model
                string or int

        order_by_id: bool
                If True order by user_id otherwise by the preference values.
                default = True
        Returns
        ----------
        Return all existing Preferences expressed for that item,
        ordered by user id (if order_by_id is True) or by the preference values
        (if order_by_id is False), as an array.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preference_value(self, user_id, item_id):
        '''
        Parameters
        ----------
        user_id: user id in the model
                int or string

        item_id: id of the item in the model
                string or int

        Returns
        ---------
        Retrieves the preference value for a single user and item.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def preference_time(self, user_id, item_id):
        '''
        Parameters
        ----------
        user_id: user id in the model
                int or string

        item_id: id of the item in the model
                string or int

        Returns
        ---------
        Retrieves the time at which a preference value from a user and item
        was set, if known. Time is expressed in the usual way, as a number
        of milliseconds since the epoch.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def users_count(self):
        '''
        Returns
        -------
        Return total number of users known to the model.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def items_count(self):
        '''
        Returns
        --------
        Return total number of items known to the model.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def set_preference(self, user_id, item_id, value=None):
        '''
        Parameters
        ----------
        user_id: user id in the model
                int or string

        item_id: id of the item in the model
                string or int

        value:  the preference
                bool or float

        Sets a particular preference (item plus rating) for a user.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def remove_preference(self, user_id, item_id):
        '''
        Parameters
        ----------
        user_id: user id in the model
                int or string

        item_id: id of the item in the model
                string or int

        Removes a particular preference for a user.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def has_preference_values(self):
        '''
        Returns
        -------
        Return True if this implementation actually it is not a 'boolean'
        data model, otherwise returns False.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def maximum_preference_value(self):
        '''
        Returns
        --------
        Return the maximum preference value that is possible in the current
        problem domain being evaluated.
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def minimum_preference_value(self):
        '''
        Returns
        --------
        Returns the minimum preference value that is possible in the current
        problem domain being evaluated
        '''
        raise NotImplementedError("cannot instantiate Abstract Base Class")
