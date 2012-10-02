"""Implements code-level dependency tracking."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import hashlib
import inspect

def hashString(strr):
    """Computes git-style hash of a string."""
    m = hashlib.sha1()
    m.update('blob '+str(len(strr))+'\0')
    m.update(strr)
    return m.hexdigest()

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
    """
    if '_codedepCodeHash' not in fnOrClassOrObj.__dict__:
        fnOrClassOrObj._codedepCodeHash = hashString(inspect.getsource(fnOrClassOrObj))
    if '_codedepCodeDeps' not in fnOrClassOrObj.__dict__:
        fnOrClassOrObj._codedepCodeDeps = []

def codeHash(codeHash):
    """Returns a decorator which can be used to manually set _codedepCodeHash.

    For example:
        a = 2

        @codeHash('a98sd7f9')
        @codeDeps(a)
        def foo(x):
            return a * x
    """
    def decorate(fnOrClass):
        fnOrClass._codedepCodeHash = codeHash
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

    This is useful when codeDeps would otherwise need to use a forward reference
    to something that hasn't been defined yet.

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
    class. It should be newly created since this function adds codedep values by
    attaching mutable state to the result of evaluating the thunk.

    For example:
        @codeDeps():
        class Two(object):
            pass

        @codeDeps(Two)
        def getTwo():
            return Two()

        two = codedepEvalThunk(getTwo)
    """
    fnOrClassOrObj = fnOrClassOrObjThunk()
    fnOrClassOrObj._codedepCodeHash = ''
    fnOrClassOrObj._codedepCodeDeps = (fnOrClassOrObjThunk,)
    _updateInfo(fnOrClassOrObj)
    return fnOrClassOrObj

def _resolveAnyForwardRefs(deps):
    return [ dep.thunk() if isinstance(dep, ForwardRef) else dep for dep in deps ]

def getDeps(fnOrClassOrObj):
    fnOrClassOrObj._codedepCodeDeps = _resolveAnyForwardRefs(fnOrClassOrObj.__dict__['_codedepCodeDeps'])
    return fnOrClassOrObj.__dict__['_codedepCodeDeps']

def getAllDeps(fnOrClassOrObj):
    ret = []
    agenda = [fnOrClassOrObj]
    seen = set()
    while agenda:
        curr = agenda.pop()
        ident = id(curr)
        if not ident in seen:
            seen.add(ident)
            ret.append(curr)
            agenda.extend(reversed(getDeps(curr)))
    return ret

def computeHash(fnOrClassOrObj):
    return hashString(str([ dep.__dict__['_codedepCodeHash'] for dep in getAllDeps(fnOrClassOrObj) ]))

# (FIXME : does this guarantee the hash will change if any changes are made to a
#   set of functions? Can come up with a counter-example?)
def getHash(fnOrClassOrObj):
    if '_codedepHash' not in fnOrClassOrObj.__dict__:
        fnOrClassOrObj._codedepHash = computeHash(fnOrClassOrObj)
    return fnOrClassOrObj.__dict__['_codedepHash']
