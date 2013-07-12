"""General purpose utility functions."""

# Copyright 2011, 2012, 2013 Matt Shannon

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

        assert 0 <= self.index < self.length

    def __repr__(self):
        return 'ElemGetter(%r, %r)' % (self.index, self.length)

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
        return 'AttrGetter(%r)' % self.attr

    def __call__(self, obj):
        return getattr(obj, self.attr)

@codeDeps()
class AndThen(object):
    """Picklable function composition."""
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return 'AndThen(%r, %r)' % (self.first, self.second)

    def __call__(self, x):
        y = self.first(x)
        return self.second(y)

@codeDeps()
class MapElem(object):
    """Maps one element of a tuple."""
    def __init__(self, index, length, fn):
        self.index = index
        self.length = length
        self.fn = fn

        assert 0 <= self.index < self.length

    def __repr__(self):
        return 'MapElem(%r, %r, %r)' % (self.index, self.length, self.fn)

    def __call__(self, xs):
        lengthActual = len(xs)
        if lengthActual != self.length:
            raise RuntimeError('sequence of length %s expected (%s actual)' %
                               (self.length, lengthActual))
        xs = list(xs)
        xs[self.index] = self.fn(xs[self.index])
        xs = tuple(xs)
        return xs

@codeDeps()
def orderedDictRepr(orderedKeys, dictIn):
    assert len(orderedKeys) == len(dictIn)
    return '{'+', '.join([ repr(key)+': '+repr(dictIn[key]) for key in orderedKeys ])+'}'
