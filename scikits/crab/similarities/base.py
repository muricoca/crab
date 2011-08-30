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

    In all instances, there is a data model against which we want to perform
    the similarity search.

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

          If `num_best` is set, queries return `num_best` most similar items,
          as a sorted list.

    """
    def __init__(self, model, distance, num_best=None):
        self.model = model
        self.distance = distance
        self._set_num_best(num_best)

    def _set_num_best(self, num_best):
        self.num_best = num_best

    def get_similarity(self, source_id, target_id):
        """
        Return similarity of the `source_id` to a specific `target_id` in the
        model.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def get_similarities(self, source_id):
        """

        Return similarity of the `source_id` to all sources in the model.

        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def __getitem__(self, source_id):
        """
        Get similarities of the `source_id` to all sources in the model.
        """
        all_sims = self.get_similarities(source_id)

        #return either all similarities as a list,
        #or only self.num_best most similar,
        #depending on settings from the constructor

        tops = sorted(all_sims, key=lambda x: -x[1])

        if all_sims:
            item_ids, preferences = zip(*all_sims)
            preferences = np.array(preferences).flatten()
            item_ids = np.array(item_ids).flatten()
            sorted_prefs = np.argsort(-preferences)
            tops = zip(item_ids[sorted_prefs], preferences[sorted_prefs])

        # return at most numBest top 2-tuples (label, sim)
        return tops[:self.num_best] if self.num_best is not None else tops
