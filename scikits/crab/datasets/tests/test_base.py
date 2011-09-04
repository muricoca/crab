
from ..base import load_movielens_r100k, load_sample_songs


def test_movielens_r100k():
    movies = load_movielens_r100k()


def test_sample_songs():
    songs = load_sample_songs()
