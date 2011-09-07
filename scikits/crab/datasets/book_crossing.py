"""Caching loader for the Book-Crossing Dataset

The description of the dataset is available on the official website at:

    http://www.informatik.uni-freiburg.de/~cziegler/BX/

Quoting the introduction:

    Collected by Cai-Nicolas Ziegler in a 4-week crawl
    (August / September 2004) from the Book-Crossing community
    with kind permission from Ron Hornbaker, CTO of Humankind
    Systems. Contains 278,858 users (anonymized but with
    demographic information) providing 1,149,780 ratings
    (explicit / implicit) about 271,379 books.


This dataset loader will download the dataset,
which its size is around 22 Mb compressed. Once
uncompressed the train set is around 130 MB.

The data is downloaded, extracted and cached in the '~/scikit_crab_data'
folder.

References
----------
Improving Recommendation Lists Through Topic Diversification,
Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen;
Proceedings of the 14th International World Wide Web Conference (WWW '05),
May 10-14, 2005, Chiba, Japan.


"""
# Copyright (c) 2011 Marcel Caraciolo <marcel@muricoca.com>
# License: Simplified BSD

import os
import urllib
import logging
import zipfile
from os.path import dirname
from os.path import join
import numpy as np
from base import Bunch
import csv

logger = logging.getLogger(__name__)

URL = "http://www.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
ARCHIVE_NAME = "BX-CSV-Dump.zip"


def download_book_crossings(target_dir):
    """ Download the book-crossing data and unzip it """
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        logger.warn("Downloading dataset from %s (77 MB)", URL)
        opener = urllib.urlopen(URL)
        open(archive_path, 'wb').write(opener.read())

    logger.info("Decompressing %s", archive_path)

    source_zip = zipfile.ZipFile(archive_path, 'r')
    archives = []
    for name in source_zip.namelist():
        if name.find('.csv') != -1:
            source_zip.extract(name, target_dir)
            archives.append(name)
    source_zip.close()
    os.remove(archive_path)

    return archives


def load_bookcrossings(data_home=None, download_if_missing=True,
                     implicit=False):
    """
    Load the filenames of the Book Crossings dataset

    data_home: optional, default: None
        Specify the storage folder for the datasets. If None,
        all files is stored in '~/data subfolders.

    download_if_missing: optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    implicit: optional, False by default
        If True, it will load the implicit ratings expressed by rating 0,
        otherwise it will load the explicit ratings expressed by rating 1-10.

    Examples
    --------
    >>> from os.path import join
    >>> from os.path import dirname
    >>> from scikits.crab.datasets.book_crossing import load_bookcrossings
    >>> data_home = join(dirname(__file__), 'scikits/crab/datasets/tests/data/')
    >>> books = load_bookcrossings(data_home)
    >>> len(books.data)
    26
    >>> len(books.item_ids)
    100

    """

    if data_home:
        if not os.path.exists(data_home):
            os.makedirs(data_home)
    else:
        data_home = join(dirname(__file__), 'data/')

    try:
        if not os.path.exists(os.path.join(data_home, 'BX-Book-Ratings.csv')) \
            and not open(os.path.join(data_home, 'BX-Books.csv')):
            raise IOError
    except Exception, e:
        print 80 * '_'
        print 'Loading files failed'
        print 80 * '_'
        print e

        if download_if_missing:
            print 'downloading the dataset...'
            try:
                download_book_crossings(data_home)
            except:
                raise Exception('FAIL: Problems during the download.')
            print 'dataset downloaded.'
        else:
            raise IOError('Book-Crossing dataset not found')

    #TO FIX: it is not working for np.loadtxt
    #ratings_m = np.loadtxt(os.path.join(data_home, 'BX-Book-Ratings.csv'),
    #            delimiter=';', skiprows=1)

    ratings_m = csv.reader(open(os.path.join(data_home,
                'BX-Book-Ratings.csv')), delimiter=';')
    ratings_m.next()
    data_books = {}
    if implicit:
        for user_id, item_id, rating in ratings_m:
            if rating == "0":
                data_books.setdefault(user_id, {})
                data_books[user_id][item_id] = True
    else:
        for user_id, item_id, rating in ratings_m:
            rating = int(rating)
            if rating != "0":
                data_books.setdefault(user_id, {})
                data_books[user_id][item_id] = int(rating)

    #Read the titles
    data_titles = np.loadtxt(os.path.join(data_home, 'BX-Books.csv'),
             delimiter=';', usecols=(0, 1), dtype=str)

    data_t = []
    for item_id, label in data_titles:
        data_t.append((item_id, label))
    data_titles = dict(data_t)

    fdescr = open(dirname(__file__) + '/descr/book-crossing.rst')

    return Bunch(data=data_books, item_ids=data_titles,
                 user_ids=None, DESCR=fdescr.read())
