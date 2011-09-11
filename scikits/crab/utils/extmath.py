"""
Extended math utilities.
"""
# Authors: Marcel Caraciolo <marcel@muricoca.com>
# License: BSD
import math

try:
    import itertools
    combinations = itertools.combinations
except AttributeError:
    def combinations(seq, r=None):
        """Generator returning combinations of items from sequence <seq>
        taken <r> at a time. Order is not significant. If <r> is not given,
        the entire sequence is returned.
        """
        if r == None:
            r = len(seq)
        if r <= 0:
            yield []
        else:
            for i in xrange(len(seq)):
                for cc in combinations(seq[i + 1:], r - 1):
                    yield [seq[i]] + cc

try:
    factorial = math.factorial
except AttributeError:
    # math.factorial is only available in Python >= 2.6
    def factorial(x):
        # don't use reduce operator or 2to3 will fail.
        # ripped from http://www.joelbdalley.com/page.pl?38
        # Ensure that n is a Natural number
        n = abs(int(x))
        if n < 1:
            n = 1

        # Store n! in variable x
        x = 1

        # Compute n!
        for i in range(1, n + 1):
            x = i * x

        # Return n!
        return x
