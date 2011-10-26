"""Math helper functions."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import numpy as np
import math
import random

def logAdd(a, b):
    k = max(a, b)
    if k == float('-inf'):
        return float('-inf')
    else:
        return np.log(math.exp(a - k) + math.exp(b - k)) + k

def logSum(l):
    """Computes log(sum(exp(l))) but in a better way.

    N.B. l should be a sequence type (an iterable), not an iterator.
    """
    if len(l) == 0:
        return float('-inf')
    else:
        k = max(l)
        if k == float('-inf'):
            return float('-inf')
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
        return np.sum(np.log(np.abs(np.linalg.svd(mat, compute_uv = False))))

def sampleDiscrete(valueProbList, absTol = 1e-6):
    """Sample a value from the given discrete distribution.

    valueProbList is a list of (value, prob) pairs, where prob is the
    probability of that value.  An exception is raised if probabilities
    do not sum to 1.0 within given tolerance.

    N.B. valueProbList should be a sequence type (an iterable), not an iterator.
    """
    total = sum(prob for value, prob in valueProbList)
    if abs(total - 1.0) > absTol:
        raise RuntimeError('probabilities should sum to 1.0 not '+str(total))
    rand = random.random()
    cumul = 0.0
    for value, prob in valueProbList:
        cumul += prob
        if cumul > rand:
            break
    return value
