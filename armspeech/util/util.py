"""General purpose utility functions."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

@codeDeps()
def identityFn(x):
    return x

@codeDeps()
class ConstantFn(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value

@codeDeps()
def getElem(xs, index, length, default = None):
    try:
        lengthActual = len(xs)
    except TypeError:
        # not a sequence type
        return default
    if lengthActual != length:
        return default
    return xs[index]

@codeDeps()
def orderedDictRepr(orderedKeys, dictIn):
    assert len(orderedKeys) == len(dictIn)
    return '{'+', '.join([ repr(key)+': '+repr(dictIn[key]) for key in orderedKeys ])+'}'
