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

    precision = evaluator.evaluate(recsys, 'precision',
                                permutation=False)
    assert_true(precision['precision'] >= 0.0 and precision['precision'] <= 1.0)

    recall = evaluator.evaluate(recsys, 'recall', permutation=False)
    assert_true(recall['recall'] >= 0.0 and recall['recall'] <= 1.0)

    f1score = evaluator.evaluate(recsys, 'f1score', permutation=False)
    assert_true(f1score['f1score'] >= 0.0 and f1score['f1score'] <= 1.0)

    all_scores = evaluator.evaluate(recsys, permutation=False)
    assert_true(all_scores['f1score'] >= 0.0 and all_scores['f1score'] <= 1.0)
    assert_true(all_scores['recall'] >= 0.0 and all_scores['recall'] <= 1.0)
    assert_true(all_scores['precision'] >= 0.0 and all_scores['precision'] <= 1.0)
    assert_true(all_scores['nmae'] >= 0.0 and all_scores['nmae'] <= 1.0)
    assert_true(all_scores['mae'] >= 0.0 and all_scores['mae'] <= 1.0)
    assert_true(all_scores['rmse'] >= 0.0 and all_scores['rmse'] <= 1.0)

    #With values at sampling.
    nmae = evaluator.evaluate(recsys, 'nmae', permutation=False,
                    sampling_users=0.6, sampling_ratings=0.6)
    assert_true(nmae['nmae'] >= 0.0 and nmae['nmae'] <= 1.0)

    #Test with boolean recsys
    assert_raises(ValueError, evaluator.evaluate, boolean_recsys, 'rank')

    #Test with specified metric
    rmse = evaluator.evaluate(boolean_recsys, 'rmse', permutation=False)
    assert_true(rmse['rmse'] >= 0.0 and rmse['rmse'] <= 1.0)

    mae = evaluator.evaluate(boolean_recsys, 'mae', permutation=False)
    assert_true(mae['mae'] >= 0.0 and mae['mae'] <= 1.0)

    nmae = evaluator.evaluate(boolean_recsys, 'nmae', permutation=False)
    assert_true(nmae['nmae'] >= 0.0 and nmae['nmae'] <= 1.0)

    precision = evaluator.evaluate(boolean_recsys, 'precision',
                                permutation=False)
    assert_true(precision['precision'] >= 0.0 and precision['precision'] <= 1.0)

    recall = evaluator.evaluate(boolean_recsys, 'recall', permutation=False)
    assert_true(recall['recall'] >= 0.0 and recall['recall'] <= 1.0)

    f1score = evaluator.evaluate(boolean_recsys, 'f1score', permutation=False)
    assert_true(f1score['f1score'] >= 0.0 and f1score['f1score'] <= 1.0)

    all_scores = evaluator.evaluate(recsys, permutation=False)
    assert_true(all_scores['f1score'] >= 0.0 and all_scores['f1score'] <= 1.0)
    assert_true(all_scores['recall'] >= 0.0 and all_scores['recall'] <= 1.0)
    assert_true(all_scores['precision'] >= 0.0 and all_scores['precision'] <= 1.0)
    assert_true(all_scores['nmae'] >= 0.0 and all_scores['nmae'] <= 1.0)
    assert_true(all_scores['mae'] >= 0.0 and all_scores['mae'] <= 1.0)
    assert_true(all_scores['rmse'] >= 0.0 and all_scores['rmse'] <= 1.0)

    #With values at sampling.
    nmae = evaluator.evaluate(boolean_recsys, 'nmae', permutation=False,
                    sampling_users=0.6, sampling_ratings=0.6)
    assert_true(nmae['nmae'] >= 0.0 and nmae['nmae'] <= 1.0)


def test_root_CfEvaluator_evaluate_on_split():
    """Check evaluate_on_split method in CfEvaluator """
    evaluator = CfEvaluator()

    #Test with invalid metric
    assert_raises(ValueError, evaluator.evaluate_on_split, recsys, 'rank')

    #Test with specified metric
    rmse = evaluator.evaluate_on_split(recsys, 'rmse', permutation=False)
    for p in rmse[0]['error']:
        assert_true(p['rmse'] >= 0.0 and p['rmse'] <= 1.0)
    assert_true(rmse[1]['final_error']['avg']['rmse'] >= 0.0 and
                rmse[1]['final_error']['stdev']['rmse'] <= 1.0)

    mae = evaluator.evaluate_on_split(recsys, 'mae', permutation=False)
    for p in mae[0]['error']:
        assert_true(p['mae'] >= 0.0 and p['mae'] <= 1.0)
    assert_true(mae[1]['final_error']['avg']['mae'] >= 0.0 and
                mae[1]['final_error']['stdev']['mae'] <= 1.0)

    nmae = evaluator.evaluate_on_split(recsys, 'nmae', permutation=False)
    for p in nmae[0]['error']:
        assert_true(p['nmae'] >= 0.0 and p['nmae'] <= 1.0)
    assert_true(nmae[1]['final_error']['avg']['nmae'] >= 0.0 and
                nmae[1]['final_error']['stdev']['nmae'] <= 1.0)

    #Test with IR statistics
    precision = evaluator.evaluate_on_split(recsys, 'precision', permutation=False)
    for p in precision[0]['ir']:
        assert_true(p['precision'] >= 0.0 and p['precision'] <= 1.0)
    assert_true(precision[1]['final_error']['avg']['precision'] >= 0.0 and
                precision[1]['final_error']['stdev']['precision'] <= 1.0)

    recall = evaluator.evaluate_on_split(recsys, 'recall', permutation=False)
    for p in recall[0]['ir']:
        assert_true(p['recall'] >= 0.0 and p['recall'] <= 1.0)
    assert_true(recall[1]['final_error']['avg']['recall'] >= 0.0 and
                recall[1]['final_error']['stdev']['recall'] <= 1.0)

    f1score = evaluator.evaluate_on_split(recsys, 'f1score', permutation=False)
    for p in f1score[0]['ir']:
        assert_true(p['f1score'] >= 0.0 and p['f1score'] <= 1.0)
    assert_true(f1score[1]['final_error']['avg']['f1score'] >= 0.0 and
                f1score[1]['final_error']['stdev']['f1score'] <= 1.0)

    all_scores = evaluator.evaluate_on_split(recsys, permutation=False)
    for p in all_scores[0]['ir']:
        assert_true(p['f1score'] >= 0.0 and p['f1score'] <= 1.0)
        assert_true(p['recall'] >= 0.0 and p['recall'] <= 1.0)
        assert_true(p['precision'] >= 0.0 and p['precision'] <= 1.0)
    for p in all_scores[0]['error']:
        assert_true(p['mae'] >= 0.0 and p['mae'] <= 1.0)
        assert_true(p['rmse'] >= 0.0 and p['rmse'] <= 1.0)
        assert_true(p['nmae'] >= 0.0 and p['nmae'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['f1score'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['f1score'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['recall'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['recall'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['precision'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['precision'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['rmse'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['rmse'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['mae'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['mae'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['nmae'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['nmae'] <= 1.0)

    #Test with boolean model
    #Test with invalid metric
    assert_raises(ValueError, evaluator.evaluate_on_split, boolean_recsys, 'rank')

    #Test with specified metric
    rmse = evaluator.evaluate_on_split(boolean_recsys, 'rmse', permutation=False)
    for p in rmse[0]['error']:
        assert_true(p['rmse'] >= 0.0 and p['rmse'] <= 1.0)
    assert_true(rmse[1]['final_error']['avg']['rmse'] >= 0.0 and
                rmse[1]['final_error']['stdev']['rmse'] <= 1.0)

    mae = evaluator.evaluate_on_split(boolean_recsys, 'mae', permutation=False)
    for p in mae[0]['error']:
        assert_true(p['mae'] >= 0.0 and p['mae'] <= 1.0)
    assert_true(mae[1]['final_error']['avg']['mae'] >= 0.0 and
                mae[1]['final_error']['stdev']['mae'] <= 1.0)

    nmae = evaluator.evaluate_on_split(boolean_recsys, 'nmae', permutation=False)
    for p in nmae[0]['error']:
        assert_true(p['nmae'] >= 0.0 and p['nmae'] <= 1.0)
    assert_true(nmae[1]['final_error']['avg']['nmae'] >= 0.0 and
                nmae[1]['final_error']['stdev']['nmae'] <= 1.0)

    #Test with IR statistics
    precision = evaluator.evaluate_on_split(boolean_recsys, 'precision', permutation=False)
    for p in precision[0]['ir']:
        assert_true(p['precision'] >= 0.0 and p['precision'] <= 1.0)
    assert_true(precision[1]['final_error']['avg']['precision'] >= 0.0 and
                precision[1]['final_error']['stdev']['precision'] <= 1.0)

    recall = evaluator.evaluate_on_split(boolean_recsys, 'recall', permutation=False)
    for p in recall[0]['ir']:
        assert_true(p['recall'] >= 0.0 and p['recall'] <= 1.0)
    assert_true(recall[1]['final_error']['avg']['recall'] >= 0.0 and
                recall[1]['final_error']['stdev']['recall'] <= 1.0)

    f1score = evaluator.evaluate_on_split(boolean_recsys, 'f1score', permutation=False)
    for p in f1score[0]['ir']:
        assert_true(p['f1score'] >= 0.0 and p['f1score'] <= 1.0)
    assert_true(f1score[1]['final_error']['avg']['f1score'] >= 0.0 and
                f1score[1]['final_error']['stdev']['f1score'] <= 1.0)

    all_scores = evaluator.evaluate_on_split(boolean_recsys, permutation=False)
    for p in all_scores[0]['ir']:
        assert_true(p['f1score'] >= 0.0 and p['f1score'] <= 1.0)
        assert_true(p['recall'] >= 0.0 and p['recall'] <= 1.0)
        assert_true(p['precision'] >= 0.0 and p['precision'] <= 1.0)

    for p in all_scores[0]['error']:
        assert_true(p['mae'] >= 0.0 and p['mae'] <= 1.0)
        assert_true(p['rmse'] >= 0.0 and p['rmse'] <= 1.0)
        assert_true(p['nmae'] >= 0.0 and p['nmae'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['f1score'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['f1score'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['recall'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['recall'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['precision'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['precision'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['rmse'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['rmse'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['mae'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['mae'] <= 1.0)
    assert_true(all_scores[1]['final_error']['avg']['nmae'] >= 0.0 and
                all_scores[1]['final_error']['stdev']['nmae'] <= 1.0)
