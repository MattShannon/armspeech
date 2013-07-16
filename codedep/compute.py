"""Computes hash values for functions and objects."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep.hash import hashString
from codedep.decorators import codeDeps, ForwardRef

@codeDeps()
def _resolveAnyForwardRefs(deps):
    return [ dep.thunk() if isinstance(dep, ForwardRef) else dep for dep in deps ]

@codeDeps(_resolveAnyForwardRefs)
def getDeps(fnOrClassOrObj):
    if '_codedepCodeDeps' not in fnOrClassOrObj.__dict__:
        raise RuntimeError('codedep values not found for %s' % fnOrClassOrObj)
    fnOrClassOrObj._codedepCodeDeps = _resolveAnyForwardRefs(fnOrClassOrObj.__dict__['_codedepCodeDeps'])
    return fnOrClassOrObj.__dict__['_codedepCodeDeps']

@codeDeps(getDeps)
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

# (FIXME : does this guarantee the hash will change if any changes are made to a
#   set of functions? Can come up with a counter-example?)
@codeDeps(getAllDeps)
def computeHash(fnOrClassOrObj):
    return hashString(str([ dep.__dict__['_codedepCodeHash'] for dep in getAllDeps(fnOrClassOrObj) ]))

@codeDeps(computeHash)
def getHash(fnOrClassOrObj):
    if '_codedepHash' not in fnOrClassOrObj.__dict__:
        fnOrClassOrObj._codedepHash = computeHash(fnOrClassOrObj)
    return fnOrClassOrObj.__dict__['_codedepHash']
