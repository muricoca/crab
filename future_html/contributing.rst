============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute.

The project is hosted on http://github.com/muricoca/crab

Submitting a bug report
=======================

In case you experience issues using the package, do not hesitate
to submit a ticket to the
`Bug Tracker <http://github.com/muricoca/crab/issues>`_.

You are also welcome to post there feature requests or links to pull-requests.

.. _git_repo:

Retrieving the latest code
==========================

You can check the latest sources with the command::

    git clone git://github.com/muricoca/crab.git

or if you have write privileges::

    git clone git@github.com:muricoca/crab.git


Contributing code
=================

.. note:

  To avoid duplicated work it is highly advised to contact the developers
  mailing list before starting work on a non-trivial feature.

  http://groups.google.com/group/scikit-crab


How to contribute
-----------------

The prefered way to contribute to `Crab` is to fork the main
repository on
`github <http://github.com/muricoca/crab/>`__:

 1. `Create an account <https://github.com/signup/free>`_ on
    github if you don't have one already.

 2. Fork the `crab repo
    <http://github.com/muricoca/crab/>`__: click on the 'Fork'
    button, at the top, center of the page. This creates a copy of
    the code on the github server where you can work.

 3. Clone this copy to your local disk (you need the `git` program to do
    this)::

        $ git clone git@github.com:YourLogin/crab.git

 4. Work on this copy, on your computer, using git to do the version
    control::

        $ git add modified_files
        $ git commit
        $ git push origin master

    and so on.

If your changes are not just trivial fixes, it is better to directly
work in a branch with the name of the feature your are working on. In
this case, replace step 4 by step 5:

  5. Create a branch to host your changes and publish it on your public
     repo::

        $ git checkout -b my-feature
        $ gid add modified_files
        $ git commit
        $ git push origin my-feature

When you are ready, and you have pushed your changes on your github repo, go
the web page of the repo, and click on 'Pull request' to send us a pull
request. This will send an email to the commiters, but might also send an
email to the mailing list in order to get more visibility.

It is recommented to check that your contribution complies with the following
rules before submitting a pull request:

    * Follow the `coding-guidelines`_ (see below).

    * All public methods should have informative docstrings with sample
      usage presented as doctests when appropriate.

You can also check for common programming errors with the following tools:

    * Code with a good unittest coverage (at least 80%), check with::

        $ pip install nose coverage
        $ nosetests --with-coverage path/to/tests_for_package

    * No pyflakes warnings, check with::

        $ pip install pyflakes
        $ pyflakes path/to/module.py

    * No PEP8 warnings, check with::

        $ pip install pep8
        $ pep8 path/to/module.py


Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the github wiki).


.. note::

  The current state of the crab code base is not compliant with
  all of those guidelines but we expect that enforcing those constraints
  on all new contributions will get the overall code base quality in the
  right direction.


EasyFix Issues
--------------

The best way to get your feet wet is
to pick up an issue from the `issue tracker
<https://github.com/muricoca/crab/issues?labels=EasyFix>`_
that are labeled as EasyFix. This means that the knowledge needed to solve
the issue is low, but still you are helping the project and letting more
experienced developers concentrate on other issues.

.. _contribute_documentation:

Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
rst docs (like this one), tutorials, etc. Rst docs live in the source
code repository, under directory doc/.

Further information about how to contribute will be explained here.


.. warning:: **Sphinx version**

   While we do our best to have the documentation build under as many
   version of Sphinx as possible, the different versions tend to behave
   slightly differently. To get the best results, you should use version
   1.0.

Developers web site
-------------------

More information can be found at the `developer's wiki
<https://github.com/muricoca/crab/wiki>`_.


Other ways to contribute
========================

Code is not the only way to contribute to this project. For instance,
documentation is also a very important part of the project and ofter
doesn't get as much attention as it deserves. If you find a typo in
the documentation, or have made improvements, don't hesitate to send
an email to the mailing list or a github pull request. Full
documentation can be found under directory doc/.

It also helps us if you spread the word: reference it from your blog,
articles, link to us from your website, or simply by saying "I use
it":

.. raw:: html
   <script type="text/javascript" src="http://www.ohloh.net/p/480792/widgets/project_users.js?style=rainbow"></script>


.. _coding-guidelines:

Coding guidelines
=================

The following are some guidelines on how new code should be written. Of
course, there are special cases and there will be exceptions to these
rules. However, following these rules when submitting new code makes
the review easier so new code can be integrated in less time.

Uniformly formatted code makes it easier to share code ownership. The
Crab tries to follow closely the official Python guidelines
detailed in `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ that
details how code should be formatted, and indented. Please read it and
follow it.

In addition, we add the following guidelines:

    * Use underscores to separate words in non class names: ``n_samples``
      rather than ``nsamples``.

    * Avoid multiple statements on one line. Prefer a line return after
      a control flow statement (``if``/``for``).

    * Use relative imports for references inside crab.

    * **Please don't use `import *` in any case**. It is considered harmful
      by the `official Python recommendations
      <http://docs.python.org/howto/doanddont.html#from-module-import>`_.
      It makes the code harder to read as the origin of symbols is no
      longer explicitly referenced, but most important, it prevents
      using a static analysis tool like `pyflakes
      <http://www.divmod.org/trac/wiki/DivmodPyflakes>`_ to automatically
      find bugs in Crab.

    * Use the `numpy docstring standard
      <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
      in all your docstrings.

A good example of code that we like can be found `here
<https://svn.enthought.com/enthought/browser/sandbox/docs/coding_standard.py>`_.


APIs of Crab objects
=============================

To have a uniform API, we try to have a common basic API for all the
objects. In addition, to avoid the proliferation of framework code, we
try to adopt simple conventions and limit to a minimum the number of
methods an object has to implement.

We are still defining the standards of implementing for Crab.

Unresolved API issues
----------------------

Some things are must still be decided:

    * Standard coding guidelines.
    * Refactoring of several methods and classes in early releases.

Working notes
---------------

For unresolved issues, TODOs, remarks on ongoing work, developers are
adviced to maintain notes on the github wiki:
https://github.com/muricoca/crab/wiki