import numpy as np
from numpy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal, run_module_suite, TestCase

from nose.tools import assert_raises

from ..pairwise import euclidean_distances, pearson_correlation, jaccard_coefficient, manhattan_distances,  \
               sorensen_coefficient, tanimoto_coefficient, cosine_distances, spearman_coefficient, loglikehood_coefficient

np.random.seed(0)

#class testPairwise(TestCase):

def test_euclidean_distances():
    """Check that the pairwise euclidian distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X,X)
    assert_array_almost_equal(D,[[1.]])   

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X,X,inverse=False)
    assert_array_almost_equal(D,[[0.]])
 
    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError,euclidean_distances,X,Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = euclidean_distances(X,Y)
    assert_array_almost_equal(D,[[0.29429806]])

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = euclidean_distances(X,Y)
    assert_array_almost_equal(D,[[0.29429806], [0.29429806]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = euclidean_distances(X,Y)
    assert_array_almost_equal(D, [[0.29429806,  1.],[ 0.29429806,  1. ]])
  
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = euclidean_distances(X,X)
    assert_array_almost_equal(D,[[1., 0.29429806], [0.29429806,1.]])    

    X = [[1.0, 0.0],[1.0,1.0]]
    Y = [[0.0, 0.0]]
    D = euclidean_distances(X,Y)
    assert_array_almost_equal(D, [[ 0.5 ],[0.41421356]] )

def test_pearson_correlation():
    """ Check that the pairwise Pearson distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = pearson_correlation(X,X)
    assert_array_almost_equal(D,[[1.]])   
 
    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError,pearson_correlation,X,Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = pearson_correlation(X,Y)
    assert_array_almost_equal(D,[[0.3960590]])

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = pearson_correlation(X,Y)
    assert_array_almost_equal(D,[[0.3960590], [0.3960590]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = pearson_correlation(X,Y)
    assert_array_almost_equal(D,[[0.3960590, 1.], [0.3960590, 1.]])

    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = pearson_correlation(X,X)
    assert_array_almost_equal(D,[[1.,0.39605902], [0.39605902,1. ]])    

    X = [[1.0, 0.0],[1.0,1.0]]
    Y = [[0.0, 0.0]]
    D = pearson_correlation(X,Y)
    assert_array_almost_equal(D,[[np.nan], [np.nan]])    


def test_spearman_distances():
    """ Check that the pairwise Spearman distances computation"""
    #Idepontent Test
    X = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5), ('e', 2.5),('f', 3.0)]]
    D = spearman_coefficient(X,X)
    assert_array_almost_equal(D,[[1.]])   
    
    #Vector x Non Vector
    X = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5), ('e', 2.5),('f', 3.0)]]
    Y = [[]]
    assert_raises(ValueError,spearman_coefficient,X,Y)

    #Vector A x Vector B
    X = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5), ('e', 2.5),('f', 3.0)]]
    Y = [[('a',3.0),('b', 3.5), ('c',1.5), ('d',5.0), ('e', 3.5),('f', 3.0)]]
    D = spearman_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.5428571428]])

    #Vector N x 1
    X = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5)],[ ('e', 2.5),('f', 3.0), ('g', 2.5), ('h', 4.0)] ]
    Y = [[('a',2.5),('b', 3.5), ('c',3.0), ('k',3.5)]]
    D = spearman_coefficient(X,Y)
    assert_array_almost_equal(D,[[1.], [0.]])
    
    #N-Dimmensional Vectors
    X = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5)],[ ('e', 2.5),('f', 3.0), ('g', 2.5), ('h', 4.0)] ]
    Y = [[('a',2.5),('b', 3.5), ('c',3.0), ('d',3.5)],[ ('e', 2.5),('f', 3.0), ('g', 2.5), ('h', 4.0)] ]
    D = spearman_coefficient(X,Y)
    assert_array_almost_equal(D,[[1.,0.], [0.,1.]])

def test_tanimoto_distances():
    """ Check that the pairwise Tanimoto distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    D = tanimoto_coefficient(X,X)
    assert_array_almost_equal(D,[[1.]])   

    #Vector x Non Vector
    X = [['a', 'b', 'c']]    
    Y = [[]]
    D = tanimoto_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.]])   

    #Vector A x Vector B
    X = [[1,2,3,4]]
    Y = [[2,3]]
    D = tanimoto_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.5]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    Y = [['a', 'b', 'c', 'k']]
    D = tanimoto_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.6], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    Y = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    D = tanimoto_coefficient(X,Y)
    assert_array_almost_equal(D,[[1., 0.], [0.,1.]])

    X = [[0,1],[1,1]]
    D = tanimoto_coefficient(X,X)
    assert_array_almost_equal(D,[[1. , 0.33333333], [0.33333333,0.33333333]])    

    X = [[0,1],[1,1]]
    Y = [[0,0]]
    D = tanimoto_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.3333333], [0.]])




def test_cosine_distances():
    """ Check that the pairwise Cosine distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = cosine_distances(X,X)
    assert_array_almost_equal(D,[[1.]])        
    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError,pearson_correlation,X,Y)
    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = cosine_distances(X,Y)
    assert_array_almost_equal(D,[[0.960646301]])
    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = cosine_distances(X,Y)
    assert_array_almost_equal(D,[[0.960646301], [0.960646301]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0], [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = cosine_distances(X,Y)
    assert_array_almost_equal(D,[[0.960646301, 1.], [ 0.960646301, 1.]])

    X = [[0,1],[1,1]]
    D = cosine_distances(X,X)
    assert_array_almost_equal(D,[[1., 0.70710678], [0.70710678, 1.]])    

    X = [[0,1],[1,1]]
    Y = [[0,0]]
    D = cosine_distances(X,Y)
    assert_array_almost_equal(D,[[np.nan], [np.nan]])    




def test_loglikehood_distances():
    """ Check that the pairwise LogLikehood distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    n_items = 3
    D = loglikehood_coefficient(n_items,X,X)
    assert_array_almost_equal(D,[[1.]])   

    #Vector x Non Vector
    X = [['a', 'b', 'c']]    
    Y = [[]]
    n_items = 3
    D = loglikehood_coefficient(n_items,X,Y)
    assert_array_almost_equal(D,[[0.]])   

    #Vector A x Vector B
    X = [[1,2,3,4]]
    Y = [[2,3]]
    n_items = 4
    D = loglikehood_coefficient(n_items,X,Y)
    assert_array_almost_equal(D,[[0.]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'],  ['e', 'f','g', 'h']]
    Y = [['a', 'b', 'c', 'k']]
    n_items = 8
    D = loglikehood_coefficient(n_items,X,Y)
    assert_array_almost_equal(D,[[0.67668852], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'],['e', 'f','g', 'h']]
    Y = [['a', 'b', 'c', 'd'],['e', 'f','g', 'h']]
    n_items = 7
    D = loglikehood_coefficient(n_items,X,Y)
    assert_array_almost_equal(D,[[1., 0.], [0., 1.]])

def test_sorensen_distances():
    """ Check that the pairwise Sorensen distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    D = sorensen_coefficient(X,X)
    assert_array_almost_equal(D,[[1.]])   

    #Vector x Non Vector
    X = [['a', 'b', 'c']]    
    Y = [[]]
    D = sorensen_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.]])   

    #Vector A x Vector B
    X = [[1,2,3,4]]
    Y = [[2,3]]
    D = sorensen_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.666666]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    Y = [['a', 'b', 'c', 'k']]
    D = sorensen_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.75], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    Y = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    D = sorensen_coefficient(X,Y)
    assert_array_almost_equal(D,[[1., 0.], [0., 1.]])

    X = [[0,1],[1,2]]
    D = sorensen_coefficient(X,X)
    assert_array_almost_equal(D,[[1., 0.5], [0.5,1.]])    

    X = [[0,1],[1,2]]
    Y = [[0,0]]
    D = sorensen_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.5], [0.]])

def test_manthattan_distances():
    """ Check that the pairwise Manhattan distances computation"""
    #Idepontent Test
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = manhattan_distances(X,X)
    assert_array_almost_equal(D,[[1.]])   

    #Vector x Non Vector
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[]]
    assert_raises(ValueError,manhattan_distances,X,Y)

    #Vector A x Vector B
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = manhattan_distances(X,Y)
    assert_array_almost_equal(D,[[0.25]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]]
    D = manhattan_distances(X,Y)
    assert_array_almost_equal(D,[[0.25], [0.25]])

    #N-Dimmensional Vectors
    X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    Y = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    D = manhattan_distances(X,Y)
    assert_array_almost_equal(D,[[1. , 1.], [1. , 1.]])

    X = [[0,1],[1,1]]
    D = manhattan_distances(X,X)
    assert_array_almost_equal(D,[[1., 1.], [0.5, 0.5]])    

    X = [[0,1],[1,1]]
    Y = [[0,0]]
    D = manhattan_distances(X,Y)
    assert_array_almost_equal(D,[[0.5], [0.]])


def test_jaccard_distances():
    """ Check that the pairwise Jaccard distances computation"""
    #Idepontent Test
    X = [['a', 'b', 'c']]
    D = jaccard_coefficient(X,X)
    assert_array_almost_equal(D,[[1.]])   

    #Vector x Non Vector
    X = [['a', 'b', 'c']]    
    Y = [[]]
    D = jaccard_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.]])   

    #Vector A x Vector B
    X = [[1,2,3,4]]
    Y = [[2,3]]
    D = jaccard_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.5]])

    #BUG FIX: How to fix for multi-dimm arrays

    #Vector N x 1
    X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    Y = [['a', 'b', 'c', 'k']]
    D = jaccard_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.6], [0.]])

    #N-Dimmensional Vectors
    X = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    Y = [['a', 'b', 'c', 'd'],['e', 'f','g']]
    D = jaccard_coefficient(X,Y)
    assert_array_almost_equal(D,[[1., 0.], [0. , 1.]])

    X = [[0,1],[1,2]]
    D = jaccard_coefficient(X,X)
    assert_array_almost_equal(D,[[1., 0.33333333], [0.33333333 ,1.]])    

    X = [[0,1],[1,2]]
    Y = [[0,3]]
    D = jaccard_coefficient(X,Y)
    assert_array_almost_equal(D,[[0.33333333], [0.]])    

    
#if __name__ == '__main__':
#    run_module_suite()
