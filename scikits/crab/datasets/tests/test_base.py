
from ..base import load_movielens_r100k, load_sample_songs, load_sample_movies
from ..book_crossing import load_bookcrossings

def test_movielens_r100k():
    movies = load_movielens_r100k()


def test_sample_songs():
    songs = load_sample_songs()

def test_sample_movies():
    songs = load_sample_movies()

def test_load_bookcrossings():
    books = load_bookcrossings()