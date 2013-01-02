"""Semiring definitions."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.util.mathhelp import logAdd, logSum
from codedep import codeDeps

# N.B. ldivide should be such that ring.times(a, ring.ldivide(a, b)) == b
#   (so ldivide(a, b) == a^{-1} b if a is invertible)

@codeDeps()
class RealsField(object):
    @property
    def zero(self):
        return 0.0
    @property
    def one(self):
        return 1.0
    def plus(self, a, b):
        return a + b
    def sum(self, it):
        return sum(it)
    def minus(self, a, b):
        return a - b
    def times(self, a, b):
        return a * b
    def inv(self, a):
        return 1.0 / a
    def divide(self, a, b):
        return a / b
    def ldivide(self, a, b):
        return b / a
    def isClose(self, a, b):
        return abs(a - b) < 1e-8
    def lt(self, a, b):
        return a < b
    def max(self, it):
        return max(it)

@codeDeps(logAdd, logSum)
class LogRealsField(object):
    @property
    def zero(self):
        return float('-inf')
    @property
    def one(self):
        return 0.0
    def plus(self, a, b):
        return logAdd(a, b)
    def sum(self, it):
        return logSum(it)
    def times(self, a, b):
        return a + b
    def inv(self, a):
        return -a
    def divide(self, a, b):
        return a - b
    def ldivide(self, a, b):
        return b - a
    def isClose(self, a, b):
        return abs(a - b) < 1e-8
    def lt(self, a, b):
        return a < b
    def max(self, it):
        return max(it)
