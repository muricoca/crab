#-*- coding:utf-8 -*-

""" 
Base Data Models.
"""
from exceptions import NotImplementedError


class BaseDataModel(object):
    
    def user_IDs(self):
        '''
        Return all user IDs in the model, in order
        '''
        raise NotImplementedError

    def item_IDs(self):
        '''
        Return a iterator of all item IDs in the model, in order
        '''
        raise NotImplementedError

    def preferences_from_user(self, user_ID, order_by_ID=True):
        '''
        Return user's preferences, ordered by user ID (if order_by_id is True) 
        or by the preference values (if order_by_id is False), as an array.
        '''
        raise NotImplementedError

    def items_from_user(self, user_ID):
        '''
        Return IDs of items user expresses a preference for 
        '''
        raise NotImplementedError
    
    def preferences_for_item(self, item_ID, order_by_id=True):
        '''
        Return all existing Preferences expressed for that item, 
        ordered by user ID (if order_by_id is True) or by the preference values 
        (if order_by_id is False), as an array.
        '''
        raise NotImplementedError
    
    def preference_value(self, user_ID, item_ID):
        '''
        Retrieves the preference value for a single user and item.
        '''
        raise NotImplementedError
    
    def preference_time(self, user_ID, item_ID):
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
    
    def set_preference(self, user_ID, item_ID,value):
        '''
        Sets a particular preference (item plus rating) for a user.
        '''
        raise NotImplementedError
    
    def remove_preference(self, user_ID, item_ID):
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
    