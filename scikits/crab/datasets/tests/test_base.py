from os.path import join
from os.path import dirname
from nose.tools import assert_equal, assert_raises
from ..base import load_movielens_r100k, load_sample_songs, load_sample_movies
from ..book_crossing import load_bookcrossings
from ...utils.testing import assert_in


def test_movielens_r100k():
    #with timestamp = False
    movies = load_movielens_r100k()
    assert_in(movies, in_=['data', 'item_ids', 'user_ids', 'DESCR'])
    assert_equal(len(movies.data), 943)
    assert_equal(len(movies.item_ids), 1682)
    assert_equal(sum([len(items) for key, items in
             movies.data.iteritems()]), 100000)

    #with timestamp = True
    movies = load_movielens_r100k(True)
    assert_in(movies, in_=['data', 'item_ids', 'user_ids', 'DESCR'])
    assert_equal(len(movies.data), 943)
    assert_equal(len(movies.item_ids), 1682)
    assert_equal(movies.data[1][1], (874965758, 5))
    assert_equal(sum([len(items) for key, items in
             movies.data.iteritems()]), 100000)


def test_sample_songs():
    songs = load_sample_songs()
    assert_in(songs, in_=['data', 'item_ids', 'user_ids', 'DESCR'])
    assert_equal(len(songs.data), 8)
    assert_equal(len(songs.item_ids), 8)
    assert_equal(sum([len(items) for key, items in
             songs.data.iteritems()]), 49)


def test_sample_movies():
    movies = load_sample_movies()
    assert_in(movies, in_=['data', 'item_ids', 'user_ids', 'DESCR'])
    assert_equal(len(movies.data), 7)
    assert_equal(len(movies.item_ids), 6)
    assert_equal(sum([len(items) for key, items in
             movies.data.iteritems()]), 35)


def test_load_bookcrossings():
    data_home = join(dirname(__file__), 'data/')
    #explicit with sample data from tests/data
    books = load_bookcrossings(data_home)
    assert_equal(len(books.data), 26)
    assert_equal(len(books.item_ids), 100)
    assert_equal(sum([len(items) for key, items in
             books.data.iteritems()]), 99)

    #implicit with sample data from tests/data
    books = load_bookcrossings(data_home, implicit=True)
    assert_equal(len(books.data), 15)
    assert_equal(len(books.item_ids), 100)
    assert_equal(sum([len(items) for key, items in
             books.data.iteritems()]), 60)

    #explicit with download denied.
    data_home = dirname(__file__)
    assert_raises(IOError, load_bookcrossings, data_home, implicit=False,
                download_if_missing=False)
