from nose.tools import assert_equals, assert_almost_equals, assert_raises, assert_true
from ...similarities.basic_similarities import UserSimilarity
from ...metrics.pairwise import  euclidean_distances, jaccard_coefficient
from ...models.classes import  MatrixPreferenceDataModel, \
     MatrixBooleanPrefDataModel
from ...recommenders.knn import  UserBasedRecommender
from ..classes import CfEvaluator
from ...recommenders.knn.neighborhood_strategies import  NearestNeighborsStrategy


movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
     'The Night Listener': 3.0},
    'Luciana Nunes': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
     'You, Me and Dupree': 3.5},
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
     'Superman Returns': 3.5, 'The Night Listener': 4.0},
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
     'The Night Listener': 4.5, 'Superman Returns': 4.0,
     'You, Me and Dupree': 2.5},
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
     'You, Me and Dupree': 2.0},
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
     'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
    'Penny Frewman': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
      'Superman Returns': 4.0},
    'Maria Gabriela': {}}

model = MatrixPreferenceDataModel(movies)
boolean_model = MatrixBooleanPrefDataModel(movies)
similarity = UserSimilarity(model, euclidean_distances)
boolean_similarity = UserSimilarity(boolean_model, jaccard_coefficient)
neighborhood = NearestNeighborsStrategy()
recsys = UserBasedRecommender(model, similarity, neighborhood)
boolean_recsys = UserBasedRecommender(boolean_model, boolean_similarity, neighborhood)


def test_root_CfEvaluator_evaluate():
    """Check evaluate method in CfEvaluator """
    evaluator = CfEvaluator()

    #Test with invalid metric
    assert_raises(ValueError, evaluator.evaluate, recsys, 'rank')

    #Test with specified metric
    rmse = evaluator.evaluate(recsys, 'rmse', permutation=False)
    assert_true(rmse['rmse'] >= 0.0 and rmse['rmse'] <= 1.0)

    mae = evaluator.evaluate(recsys, 'mae', permutation=False)
    assert_true(mae['mae'] >= 0.0 and mae['mae'] <= 1.0)

    nmae = evaluator.evaluate(recsys, 'nmae', permutation=False)
    assert_true(nmae['nmae'] >= 0.0 and nmae['nmae'] <= 1.0)
