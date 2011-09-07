"""Testing utilities."""

# Copyright (c) 2011 Marcel Caraciolo <marcel@muricoca.com>
# License: Simplified BSD


def assert_in(obj, in_=None, out_=None):
    """Checks that all names in `in_` as in `obj`, but no name
    in `out_` is."""
    if in_ is not None:
        for name in in_:
            assert name in obj
    if out_ is not None:
        for name in out_:
            assert name not in obj
