#-*- coding:utf-8 -*-

"""
Base Similarity Models.

"""
#Authors: Marcel Caraciolo <marcel@muricoca.com>
#License: BSD Style

import numpy as np

class BaseSimilarity(object):
    """
    Base Class for similarity that searches over a set of items/users.
    
    In all instances, there is a data model against which we want to perform the
    similarity search.
    
    For each similarity search, the input is a item/user and the output are its
    similarities to individual items/users.
    
    Similarity queries are realized by calling ``self[query_item]``.
    There is also a convenience wrapper, where iterating over `self` yields
    similarities of each object in the model against the whole data model (ie.,
    the query is each item/user in turn).

    Should not be used directly, use derived classes instead
    
    Attributes
    ----------

     `model`:  DataModel
          Defines the data model where data is fetched.
     `distance`: Function 
          Pairwise Function between two vectors.
      `num_best': int
          If it is left unspecified, similarity queries return a full list (one
          float for every item in the model, including the query item).

          If `num_best` is set, queries return `num_best` most similar items, as a
          sorted list.

    """ 
    def __init__(self,model,distance,num_best=None):        
        self.model = model
        self.distance = distance
        self.num_best = num_best


    def get_similarity(self,X,Y):
        """
        Return similarity of a vector `X` to a specific vector `Y` in the model.
        The vector is assumed to be either of unit length or empty.
        
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    
    def get_similarities(self,X):
        """
        
        Return similarity of a vector `X` to all vectors in the model.
        The vector is assumed to be either of unit length or empty.
        
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")
        

    def __getitem__(self,X):
        """
        Get similarities of a vector `X` to all items in the model
        """
        allSims = self.get_similarities(vec)
        
        #return either all similarities as a list, or only self.numBest most similar, depending on settings from the constructor
        
        if self.num_best is None:
            return allSims
        else:
            tops = [(label, sim) for label, sim in allSims]
            tops = sorted(tops, key = lambda item: -item[1]) # sort by -sim => highest sim first
            return tops[ : self.num_best] # return at most numBest top 2-tuples (label, sim)