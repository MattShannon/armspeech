"""Math helper functions."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import numpy as np
import armspeech.numpy_settings
import math
import random

@codeDeps()
def getNegInf():
    return float('-inf')

_negInf = float('-inf')

@codeDeps()
def assert_allclose(actual, desired, rtol = 1e-7, atol = 1e-14,
                    msg = 'items not almost equal'):
    if np.shape(actual) != np.shape(desired):
        raise AssertionError('%s (wrong shape)\n ACTUAL:  %r\n DESIRED: %r' %
                             (msg, actual, desired))
    if not np.allclose(actual, desired, rtol, atol):
        absErr = np.abs(actual - desired)
        relErr = np.abs((actual - desired) / desired)
        raise AssertionError('%s\n ACTUAL:  %r\n DESIRED: %r\n'
                             ' ABS ERR: %r (max %s)\n REL ERR: %r (max %s)' %
                             (msg, actual, desired,
                              absErr, np.max(absErr), relErr, np.max(relErr)))

@codeDeps()
def logAdd(a, b):
    """Computes log(exp(a) + exp(b)) in a way that avoids underflow."""
    if a == _negInf and b == _negInf:
        # np.logaddexp(_negInf, _negInf) incorrectly returns nan
        return _negInf
    else:
        return np.logaddexp(a, b)

@codeDeps(logAdd)
def logSum(l):
    """Computes log(sum(exp(l))) in a way that avoids underflow.

    N.B. l should be a sequence type (an iterable), not an iterator.
    """
    if len(l) == 0:
        return _negInf
    elif len(l) == 1:
        return l[0]
    elif len(l) == 2:
        return logAdd(l[0], l[1])
    elif len(l) < 10:
        ret = reduce(np.logaddexp, l)
        if math.isnan(ret):
            # np.logaddexp(_negInf, _negInf) incorrectly returns nan
            pass
        else:
            return ret

    k = max(l)
    if k == _negInf:
        return _negInf
    else:
        return np.log(np.sum(np.exp(np.array(l) - k))) + k

@codeDeps()
class ThreshMax(object):
    """Computes the thresholded maximum of a list.

    The thresholded maximum of a list *xs* of floats is a list of all the items
    of *xs* within a specified threshold *thresh* of the maximum of *xs*.

    If *xs* is not a list of floats, a *key* function mapping items to floats
    may be specified.

    If *thresh* is *0.0* (and given that *key* returns floats), this is
    equivalent to the built-in *max* function, except that it returns a list of
    all maximal values rather than picking the first one, and if *xs* is empty
    it returns an empty list rather than raising an exception.
    """
    def __init__(self, thresh, key = None):
        self.thresh = thresh
        self.key = key

        # thresh < 0.0 can cause problems when there are -inf values in xs, and
        #   thresh == +inf can cause problems when there are +inf values in xs
        assert 0.0 <= self.thresh < float('inf')

    def __call__(self, xs):
        thresh = self.thresh
        key = self.key

        if key is None:
            maxValue = max(xs) if xs else 0.0
            return [
                x for x in xs
                if x >= maxValue - thresh
            ]
        else:
            maxValue = max(map(key, xs)) if xs else 0.0
            return [
                x for x in xs
                if key(x) >= maxValue - thresh
            ]

    def flatten(self, xss):
        return self([ x for xs in xss for x in xs ])

@codeDeps()
def sigmoid(a):
    if a > 40.0:
        return 1.0
    elif a < -500.0:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-a))

@codeDeps()
def logDet(mat):
    if np.shape(mat) == (0, 0):
        return 0.0
    else:
        # FIXME : replace with slogdet once we're using numpy 2.0?
        return np.sum(np.log(np.linalg.svd(mat, compute_uv = False)))

@codeDeps()
def logDetPosDef(mat):
    if np.shape(mat) == (0, 0):
        return 0.0
    else:
        # FIXME : replace with slogdet once we're using numpy 2.0?
        return np.sum(np.log(np.diag(np.linalg.cholesky(mat)))) * 2.0

@codeDeps()
def sampleDiscrete(valueProbList, absTol = 1e-6):
    """Sample a value from the given discrete distribution.

    valueProbList is a list of (value, prob) pairs, where prob is the
    probability of that value. An exception is raised if probabilities
    do not sum to 1.0 within given tolerance.

    N.B. valueProbList should be a sequence type (an iterable), not an iterator.
    """
    total = sum([ prob for value, prob in valueProbList ])
    if abs(total - 1.0) > absTol:
        raise RuntimeError('probabilities should sum to 1.0 not %s' % total)
    rand = random.random()
    cumul = 0.0
    for value, prob in valueProbList:
        cumul += prob
        if cumul > rand:
            break
    return value

@codeDeps()
class AsArray(object):
    def __init__(self):
        pass
    def __repr__(self):
        return 'AsArray()'
    def __call__(self, x):
        return np.asarray(x)

@codeDeps()
def reprArray(arr):
    """Returns a repr of a numpy ndarray suitable for reading with eval.

    Default repr function provided by numpy sometimes includes an erroneous
    shape keyword that messes things up.
    """
    if np.size(arr) == 0:
        return 'zeros(%r, dtype = %r)' % (np.shape(arr), arr.dtype)
    else:
        return repr(arr)
