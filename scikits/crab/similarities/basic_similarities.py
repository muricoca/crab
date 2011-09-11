#-*- coding:utf-8 -*-

"""
This module contains functions and classes for computing similarities across
a collection of vectors.
"""
#Authors: Marcel Caraciolo <marcel@muricoca.com>
#License: BSD Style


import numpy as np
from base import BaseSimilarity
from ..metrics.pairwise import loglikehood_coefficient


def find_common_elements(source_preferences, target_preferences):
    ''' Returns the preferences from both vectors '''
    src = dict(source_preferences)
    tgt = dict(target_preferences)

    inter = np.intersect1d(src.keys(), tgt.keys())

    common_preferences = zip(*[(src[item], tgt[item]) for item in inter \
            if not np.isnan(src[item]) and not np.isnan(tgt[item])])
    if common_preferences:
        return np.asarray([common_preferences[0]]), np.asarray([common_preferences[1]])
    else:
            return np.asarray([[]]), np.asarray([[]])

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

    Examples
    ---------
    >>> from scikits.crab.models.classes import MatrixPreferenceDataModel
    >>> from scikits.crab.metrics.pairwise import cosine_distances
    >>> from scikits.crab.similarities.basic_similarities import UserSimilarity
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
    >>> similarity = UserSimilarity(model, cosine_distances, 3)
    >>> similarity['Marcel Caraciolo']
    [('Marcel Caraciolo', 1.0), ('Sheldom', 0.99127582693458016),
      ('Lorena Abreu', 0.98658676452792504)]

   '''

    def __init__(self, model, distance, num_best=None):
        BaseSimilarity.__init__(self, model, distance, num_best)

    def get_similarity(self, source_id, target_id):
        source_preferences = self.model.preferences_from_user(source_id)
        target_preferences = self.model.preferences_from_user(target_id)

        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)

        if source_preferences.ndim == 1 and target_preferences.ndim == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])

        if self.distance == loglikehood_coefficient:
            return self.distance(self.model.items_count(), \
                source_preferences, target_preferences) \
                if not source_preferences.shape[1] == 0 and \
                not target_preferences.shape[1] == 0 else np.array([[np.nan]])

        #evaluate the similarity between the two users vectors.
        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 \
                and not target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return[(other_id, self.get_similarity(source_id, other_id))  for other_id, v in self.model]

    def __iter__(self):
        """
        For each object in model, compute the similarity function against all other objects and yield the result.
        """
        for source_id, preferences in self.model:
            yield source_id, self[source_id]

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

    Examples
    ---------
    >>> from scikits.crab.models.classes import MatrixPreferenceDataModel
    >>> from scikits.crab.metrics.pairwise import cosine_distances
    >>> from scikits.crab.similarities.basic_similarities import ItemSimilarity
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
    >>> similarity = ItemSimilarity(model, cosine_distances, 3)
    >>> similarity['The Night Listener']
    [('The Night Listener', 1.0), ('Lady in the Water', 0.98188311415053031),
        ('Just My Luck', 0.97489347126452108)]

    '''

    def __init__(self, model, distance, num_best=None):
        BaseSimilarity.__init__(self, model, distance, num_best)

    def get_similarity(self, source_id, target_id):
        source_preferences = self.model.preferences_for_item(source_id)
        target_preferences = self.model.preferences_for_item(target_id)

        if self.model.has_preference_values():
            source_preferences, target_preferences = \
                find_common_elements(source_preferences, target_preferences)

        if source_preferences.ndim == 1 and target_preferences.ndim == 1:
            source_preferences = np.asarray([source_preferences])
            target_preferences = np.asarray([target_preferences])

        if self.distance == loglikehood_coefficient:
            return self.distance(self.model.items_count(), \
                source_preferences, target_preferences) \
                if not source_preferences.shape[1] == 0 and \
                    not target_preferences.shape[1] == 0 else np.array([[np.nan]])

        #Evaluate the similarity between the two users vectors.
        return self.distance(source_preferences, target_preferences) \
            if not source_preferences.shape[1] == 0 and \
                not target_preferences.shape[1] == 0 else np.array([[np.nan]])

    def get_similarities(self, source_id):
        return [(other_id, self.get_similarity(source_id, other_id)) for other_id in self.model.item_ids()]

    def __iter__(self):
        """
        For each object in model, compute the similarity function against all other objects and yield the result.
        """
        for item_id in self.model.item_ids():
            yield item_id, self[item_id]
