#-*- coding:utf-8 -*-

""" 
Base Recommender Models.
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>
#          Bruno Melo <bruno@muricoca.com>
# License: BSD Style.

import unittest

from ..base import BaseRecommender

#test classes

class MyRecommender(BaseRecommender):
    def __init__(self,model):
        BaseRecommender.__init__(self,model)
    
################################################################################
# The tests


class testBaseRecommender(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()

