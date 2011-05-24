#-*- coding:utf-8 -*-

""" 
Base Similarity Models.
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>
#          Bruno Melo <bruno@muricoca.com>
# License: BSD Style.

import unittest

from ..base import BaseSimilarity

#test classes

class MySimilarity(BaseSimilarity):
    def __init__(self,model,distance):
        BaseSimilarity.__init__(self,model,distance)
    
################################################################################
# The tests


class testBaseSimilarity(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()

