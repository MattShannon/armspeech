"""Memoization of functions."""

# Copyright 2011, 2012, 2013, 2014 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps, ForwardRef

@codeDeps(ForwardRef(lambda: MemoizedFn))
def memoize(fn):
    return MemoizedFn(fn)

@codeDeps()
class MemoizedFn(object):
    """Dictionary-based function memoization.

    N.B. arguments to fn must all be hashable.
    """
    def __init__(self, fn):
        self.fn = fn

        self.mem = dict()

    def __call__(self, *args):
        if args not in self.mem:
            self.mem[args] = self.fn(*args)
        return self.mem[args]
