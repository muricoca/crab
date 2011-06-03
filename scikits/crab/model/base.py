#-*- coding:utf-8 -*-

""" 
Base Data Models.
"""


class BaseDataModel(object):
    
    def user_ids(self):
        '''
        Return all user ids in the model, in order
        '''
        raise NotImplementedError

    def item_ids(self):
        '''
        Return a iterator of all item ids in the model, in order
        '''
        raise NotImplementedError

    def preferences_from_user(self, user_id, order_by_id=True):
        '''
        Return user's preferences, ordered by user id (if order_by_id is True) 
        or by the preference values (if order_by_id is False), as an array.
        '''
        raise NotImplementedError

    def items_from_user(self, user_id):
        '''
        Return ids of items user expresses a preference for 
        '''
        raise NotImplementedError
    
    def preferences_for_item(self, item_id, order_by_id=True):
        '''
        Return all existing Preferences expressed for that item, 
        ordered by user id (if order_by_id is True) or by the preference values 
        (if order_by_id is False), as an array.
        '''
        raise NotImplementedError
    
    def preference_value(self, user_id, item_id):
        '''
        Retrieves the preference value for a single user and item.
        '''
        raise NotImplementedError
    
    def preference_time(self, user_id, item_id):
        '''
        Retrieves the time at which a preference value from a user and item was set, if known.
        Time is expressed in the usual way, as a number of milliseconds since the epoch.
        '''
        raise NotImplementedError
    
    def users_count(self):
        '''
        Return total number of users known to the model.
        '''
        raise NotImplementedError
    
    def items_count(self):
        '''
        Return total number of items known to the model.
        '''
        raise NotImplementedError
    
    def set_preference(self, user_id, item_id, value):
        '''
        Sets a particular preference (item plus rating) for a user.
        '''
        raise NotImplementedError
    
    def remove_preference(self, user_id, item_id):
        '''
        Removes a particular preference for a user.
        '''
        raise NotImplementedError
    
    def has_preference_values(self):
        '''
        Return True if this implementation actually it is not a 'boolean' data model, otherwise returns False.
        '''
        raise NotImplementedError
    
    def maximum_preference_value(self):
        '''
        Return the maximum preference value that is possible in the current problem domain being evaluated.
        '''
        raise NotImplementedError
    
    def minimum_preference_value(self):
        '''
        Returns the minimum preference value that is possible in the current problem domain being evaluated
        '''
        raise NotImplementedError
    