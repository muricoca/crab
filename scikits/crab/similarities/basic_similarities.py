#-*- coding:utf-8 -*-

""" 
This module contains functions and classes for computing similarities across
a collection of vectors.
"""
#Authors: Marcel Caraciolo <marcel@muricoca.com>
#License: BSD Style


import numpy as np
from base import BaseSimilarity

###############################################################################
# User Similarity

class UserSimilarity(BaseSimilarity):
   '''
    Returns the degree of similarity, of two users, based on the their preferences.
    Implementations of this class define a notion of similarity between two users. 
    Implementations should  return values in the range 0.0 to 1.0, with 1.0 representing 
    perfect similarity.
    
    
    Parameters
    ----------
    
    `model`:  DataModel
         Defines the data model where data is fetched.
    `distance`: Function 
         Pairwise Function between two vectors.
     `num_best`: int
         If it is left unspecified, similarity queries return a full list (one
         float for every item in the model, including the query item).

         If `num_best` is set, queries return `num_best` most similar items, as a
         sorted list.              

    Methods
    ---------
    
    get_similarity() 
    Return similarity of the `source_id` to a specific `target_id` in the model.        
    
    get_similarities()
    Return similarity of the `source_id` to all sources in the model.
    
   '''
   def __init__(self,model,distance,num_best=None):
       BaseSimilarity.__init__(self,model,distance,num_best)
   
   def get_similarity(self,source_id,target_id):
       source_preferences = self.model.preferences_from_user(source_id)
       target_preferences = self.model.preferences_from_user(target_id)
       print source_preferences, target_preferences
       #evaluate the similarity between the two users vectors.
       return self.distance(source_preferences,target_preferences)

   def get_similarities(self,source_id):
       return [ (other_id,self.get_similarity(source_id,other_id))  for other_id,v in self.model]
       
   def __iter__(self):
       """
       For each object in model, compute the similarity function against all other objects and yield the result. 
       """
       for source_id,preferences in self.model:
           yield self[source_id]

###############################################################################
# Item Similarity

class ItemSimilarity(BaseSimilarity):
    '''
    Returns the degree of similarity, of two items, based on its preferences by the users.
    Implementations of this class define a notion of similarity between two items. 
    Implementations should  return values in the range 0.0 to 1.0, with 1.0 representing 
    perfect similarity.
    
    Parameters
    ----------
    
    `model`:  DataModel
         Defines the data model where data is fetched.
    `distance`: Function 
         Pairwise Function between two vectors.
     `num_best`: int
         If it is left unspecified, similarity queries return a full list (one
         float for every item in the model, including the query item).

         If `num_best` is set, queries return `num_best` most similar items, as a
         sorted list.              

    Methods
    ---------
    
    get_similarity() 
    Return similarity of the `source_id` to a specific `target_id` in the model.        
    
    get_similarities()
    Return similarity of the `source_id` to all sources in the model.
    
    '''

    def __init__ (self,model,distance,numBest=None):
        BaseSimilarity.__init__(self,model,distance,numBest)

    def get_similarity(self,source_id,target_id):
        source_preferences = self.model.preferences_for_item(source_id)
        target_preferences = self.model.preferences_for_item(target_id)

        #Evaluate the similarity between the two users vectors. 
        return self.distance(source_preferences,target_preferences)

    def get_similarities(self,source_id):
        return [ (other_id,self.get_similarity(source_id,other_id))  for other_id,v in self.model.item_ids()]

    def __iter__(self):
        """
        For each object in model, compute the similarity function against all other objects and yield the result. 
        """
        for item_id,preferences in self.model.item_ids():
            yield item_id, self.model.preferences_for_item(item_id)
