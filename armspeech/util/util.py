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
class ElemGetter(object):
    """Picklable item getter with length check."""
    def __init__(self, index, length):
        self.index = index
        self.length = length

    def __repr__(self):
        return 'ElemGetter(%s, %s)' % (repr(self.index), repr(self.length))

    def __call__(self, xs):
        lengthActual = len(xs)
        if lengthActual != self.length:
            raise RuntimeError('sequence of length %s expected (%s actual)' %
                               (self.length, lengthActual))
        return xs[self.index]

@codeDeps()
class AttrGetter(object):
    """Picklable attribute getter."""
    def __init__(self, attr):
        self.attr = attr

    def __repr__(self):
        return 'AttrGetter(%s)' % repr(self.attr)

    def __call__(self, obj):
        return getattr(obj, self.attr)

@codeDeps()
def orderedDictRepr(orderedKeys, dictIn):
    assert len(orderedKeys) == len(dictIn)
    return '{'+', '.join([ repr(key)+': '+repr(dictIn[key]) for key in orderedKeys ])+'}'
