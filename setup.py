#!/usr/bin/env python

import os

descr = """Crab is a flexible, fast recommender engine for Python. The engine
  aims to provide a rich set of components from which you can construct a
  customized recommender system from a set of algorithms."""

DISTNAME = 'scikits.crab'
DESCRIPTION = 'A recommender engine for Python.'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Marcel Caraciolo',
MAINTAINER_EMAIL = 'marcel@muricoca.com',
URL = 'http://muricoca.github.com/crab/'
LICENSE = 'new BSD'
DOWNLOAD_URL = "http://pypi.python.org/pypi/crab"
VERSION = '0.1'

from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
             namespace_package=['scikits'])

    config.set_options(
                ignore_setup_xxx_py=True,
                assume_default_configuration=True,
                delegate_options_to_subpackages=True,
                quiet=True,
    )

    config.add_subpackage('scikits')
    config.add_subpackage(DISTNAME)
    config.add_data_files('scikits/__init__.py')

    return config

if __name__ == "__main__":
    setup(configuration=configuration,
        name=DISTNAME,
        version=VERSION,
        include_package_data=True,
        install_requires='numpy',
        namespace_packages=['scikits'],
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ])
