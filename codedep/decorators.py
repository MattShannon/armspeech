"""Decorators for adding dependency information to functions and objects."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep.hash import hashString

import inspect

def _updateInfo(fnOrClassOrObj):
    """Sets any unset codedep values to default values.

    fnOrClassOrObj should be a function or a class or an instance of a class.
    In the case that it is an instance of a class, _codedepCodeHash should
    typically already be set since it is usually not possible to look up the
    source lines automatically.

    Called whenever one codedep decorator is used, so that if one codedep value
    is set then they all are.
    This means that any one codedep decorator line suffices in the case where
    the other codedep values have their default values, which saves typing.

    Note that adding codedep values to an instance of a class means that the
    codedep values will be pickled (this may or may not be a problem,
    depending on the use case).
    """
    if '_codedepCodeHash' not in fnOrClassOrObj.__dict__:
        fnOrClassOrObj._codedepCodeHash = hashString(
            inspect.getsource(fnOrClassOrObj)
        )
    if '_codedepCodeDeps' not in fnOrClassOrObj.__dict__:
        fnOrClassOrObj._codedepCodeDeps = []

def codeHash(hashValue):
    """Returns a decorator which can be used to manually set _codedepCodeHash.

    For example:
        a = 2

        @codeHash('a98sd7f9')
        @codeDeps(a)
        def foo(x):
            return a * x
    """
    def decorate(fnOrClass):
        fnOrClass._codedepCodeHash = hashValue
        _updateInfo(fnOrClass)
        return fnOrClass
    return decorate

def codeDeps(*deps):
    """Returns a decorator which can be used to set _codedepCodeDeps.

    For example:
        @codeDeps()
        def bar(x):
            return x

        @codeDeps(bar)
        def foo(x):
            return bar(x)
    """
    def decorate(fnOrClass):
        fnOrClass._codedepCodeDeps = deps
        _updateInfo(fnOrClass)
        return fnOrClass
    return decorate

class ForwardRef(object):
    """Stores an arbitrary callable for use with getDeps.

    This is useful when codeDeps would otherwise need to use a forward
    reference to something that hasn't been defined yet.

    For example:
        @codeDeps(ForwardRef(lambda: bar))
        def foo(x):
            return bar(x)

        @codeDeps()
        def bar(x):
            return x
    """
    def __init__(self, thunk):
        self.thunk = thunk

def codedepEvalThunk(fnOrClassOrObjThunk):
    """Evaluates a thunk, adding codedep values to the result.

    Thunk should evaluate to a newly created function or class or instance of a
    class. It should be newly created since this function adds codedep values
    by attaching mutable state to the result of evaluating the thunk.

    For example:
        @codeDeps():
        class Two(object):
            pass

        @codeDeps(Two)
        def getTwo():
            return Two()

        two = codedepEvalThunk(getTwo)

    Note that adding codedep values to an instance of a class means that the
    codedep values will be pickled (this may or may not be a problem,
    depending on the use case).
    """
    fnOrClassOrObj = fnOrClassOrObjThunk()
    fnOrClassOrObj._codedepCodeHash = ''
    fnOrClassOrObj._codedepCodeDeps = (fnOrClassOrObjThunk,)
    _updateInfo(fnOrClassOrObj)
    return fnOrClassOrObj
