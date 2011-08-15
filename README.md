.. -*- mode: rst -*-

About
=====

``Crab`` is a ﬂexible, fast recommender engine for Python that 
integrates classic information ﬁltering recom- mendation 
algorithms in the world of scientiﬁc Python packages 
(numpy, scipy, matplotlib).

The target audience is experienced Python developers familiar with
numpy and scipy.


The engine aims to provide a rich set of components from which you can 
construct a customized recommender system from a set of algorithms.


Building the HTML
=====================

You can build the HTML versions of this
webpage by installing sphinx (1.0.0+)::

  $ sudo pip install -U sphinx

Then for the html variant::

  $ cd tutorial
  $ make html

The results is available in the ``_build/html/`` subdolder. Point your browser
to the ``index.html`` file for table of content.


Testing
=======

The example snippets in the rST source files can be tested with `nose`_::

  $ nosetests -s --with-doctest --doctest-tests --doctest-extension=rst

.. _`nose`: http://somethingaboutorange.com/mrl/projects/nose/


Publishing a new version of the HTML homepage
=============================================

If your are part of the the github repo admin team, you can further
update the online HTML version using (in the ``tutorial/`` folder)::

  $ make clean html github


License
=======

This webpage is distributed under the Creative Commons Attribution
3.0 license.
