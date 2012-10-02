"""Math helper functions."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import numpy as np
import armspeech.numpy_settings
import math
import random

def getNegInf():
    return float('-inf')

_negInf = float('-inf')

def assert_allclose(actual, desired, rtol = 1e-7, atol = 1e-14, msg = 'items not almost equal'):
    if np.shape(actual) != np.shape(desired):
        raise AssertionError(
            msg+' (wrong shape)'+
            '\n ACTUAL:  '+repr(actual)+
            '\n DESIRED: '+repr(desired)
        )
    if not np.allclose(actual, desired, rtol, atol):
        absErr = np.abs(actual - desired)
        relErr = np.abs((actual - desired) / desired)
        raise AssertionError(
            msg+
            '\n ACTUAL:  '+repr(actual)+
            '\n DESIRED: '+repr(desired)+
            '\n ABS ERR: '+repr(absErr)+' (max '+str(np.max(absErr))+')'+
            '\n REL ERR: '+repr(relErr)+' (max '+str(np.max(relErr))+')'
        )

def logAdd(a, b):
    """Computes log(exp(a) + exp(b)) in a way that avoids underflow."""
    if a == _negInf and b == _negInf:
        # np.logaddexp(_negInf, _negInf) incorrectly returns nan
        return _negInf
    else:
        return np.logaddexp(a, b)

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

def sigmoid(a):
    if a > 40.0:
        return 1.0
    elif a < -500.0:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-a))

def logDet(mat):
    if np.shape(mat) == (0, 0):
        return 0.0
    else:
        # FIXME : replace with slogdet once we're using numpy 2.0?
        return np.sum(np.log(np.linalg.svd(mat, compute_uv = False)))

def sampleDiscrete(valueProbList, absTol = 1e-6):
    """Sample a value from the given discrete distribution.

    valueProbList is a list of (value, prob) pairs, where prob is the
    probability of that value. An exception is raised if probabilities
    do not sum to 1.0 within given tolerance.

    N.B. valueProbList should be a sequence type (an iterable), not an iterator.
    """
    total = sum([ prob for value, prob in valueProbList ])
    if abs(total - 1.0) > absTol:
        raise RuntimeError('probabilities should sum to 1.0 not '+str(total))
    rand = random.random()
    cumul = 0.0
    for value, prob in valueProbList:
        cumul += prob
        if cumul > rand:
            break
    return value

class AsArray(object):
    def __init__(self):
        pass
    def __repr__(self):
        return 'AsArray()'
    def __call__(self, x):
        return np.asarray(x)
