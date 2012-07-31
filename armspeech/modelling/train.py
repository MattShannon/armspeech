"""Helper functions for training models."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import nodetree
import dist as d
from minimize import minimize
from armspeech.util.timing import timed

def expectationMaximization(distPrev, accumulate, createAcc = d.defaultCreateAcc, estimateTotAux = d.defaultEstimateTotAux, afterAcc = None, monotoneAux = True, verbosity = 0):
    """Performs one step of expectation-maximization."""
    acc = createAcc(distPrev)
    accumulate(acc)
    if afterAcc is not None:
        afterAcc(acc)
    logLikePrev = acc.logLike()
    count = acc.count()
    dist, (aux, auxRat) = estimateTotAux(acc)
    if monotoneAux and aux < logLikePrev:
        raise RuntimeError('re-estimated auxiliary value (%s) less than previous log likelihood (%s) during expectation-maximization' % (aux / count, logLikePrev / count))
    if verbosity >= 2:
        print 'trainEM:    logLikePrev = %s -> aux = %s (%s) (%s count)' % (logLikePrev / count, aux / count, d.ratToString(auxRat), count)
    return dist, logLikePrev, (aux, auxRat), count

def trainEM(distInit, accumulate, createAcc = d.defaultCreateAcc, estimateTotAux = d.defaultEstimateTotAux, logLikePrevInit = float('-inf'), deltaThresh = 1e-8, minIterations = 1, maxIterations = None, beforeAcc = None, afterAcc = None, afterEst = None, monotone = False, monotoneAux = True, verbosity = 0):
    assert minIterations >= 1
    assert maxIterations is None or maxIterations >= minIterations

    dist = distInit
    logLikePrev = logLikePrevInit
    it = 0
    converged = False
    while it < minIterations or (not converged) and (maxIterations is None or it < maxIterations):
        if beforeAcc is not None:
            beforeAcc(dist)
        logLikePrevPrev = logLikePrev
        if verbosity >= 2:
            print 'trainEM: it %s:' % (it + 1)
        dist, logLikePrev, (aux, auxRat), count = expectationMaximization(dist, accumulate, createAcc = createAcc, estimateTotAux = estimateTotAux, afterAcc = afterAcc, monotoneAux = monotoneAux, verbosity = verbosity)
        deltaLogLikePrev = logLikePrev - logLikePrevPrev
        if monotone and deltaLogLikePrev < 0.0:
            raise RuntimeError('log likelihood decreased during expectation-maximization')
        if verbosity >= 2:
            print 'trainEM:    deltaLogLikePrev = %s' % (deltaLogLikePrev / count)
        if afterEst is not None:
            afterEst(dist = dist, it = it)
        converged = (abs(deltaLogLikePrev) <= deltaThresh * count)
        it += 1

    if verbosity >= 1:
        if converged:
            print 'trainEM: converged at thresh', deltaThresh, 'in', it, 'iterations'
        else:
            print 'trainEM: did NOT converge at thresh', deltaThresh, 'in', it, 'iterations'

    return dist

def trainCG(distInit, accumulate, ps = d.defaultParamSpec, length = -50, verbosity = 0):
    def negLogLike_derivParams(params):
        dist = ps.parseAll(distInit, params)
        acc = ps.createAccG(dist)
        accumulate(acc)
        count = acc.count()
        # FIXME : is it better to return logLike or logLike-per-count (e.g. logLike-per-frame)? (i.e. for which of these is minimize typically faster?)
        assert count > 0.0
        return -acc.logLike() / count, -ps.derivParams(acc) / count

    params = ps.params(distInit)
    if verbosity >= 2:
        print 'trainCG: initial params =', params
        print 'trainCG: initial derivParams =', -negLogLike_derivParams(params)[1]
    params, negLogLikes, lengthUsed = minimize(negLogLike_derivParams, params, length = length, verbosity = verbosity)
    if verbosity >= 3:
        print 'trainCG: logLikes =', map(lambda x: -x, negLogLikes)
    if verbosity >= 2:
        print 'trainCG: final params =', params
        print 'trainCG: final derivParams =', -negLogLike_derivParams(params)[1]
    if verbosity >= 1:
        print 'trainCG: logLike %s -> %s (delta = %s)' % (-negLogLikes[0], -negLogLikes[-1], negLogLikes[0] - negLogLikes[-1])
        print 'trainCG: (used', lengthUsed, 'function evaluations)'
    dist = ps.parseAll(distInit, params)

    return dist

def trainCGandEM(distInit, accumulate, ps = d.defaultParamSpec, createAccEM = d.defaultCreateAcc, estimate = d.defaultEstimate, iterations = 5, length = -50, afterEst = None, verbosity = 0):
    assert iterations >= 1

    dist = distInit
    for it in range(1, iterations + 1):
        if verbosity >= 1:
            print 'trainCGandEM: starting it =', it, 'of CG and EM'

        dist = (timed(trainCG) if verbosity >= 2 else trainCG)(dist, accumulate, ps = ps, length = length, verbosity = verbosity)

        acc = createAccEM(dist)
        (timed(accumulate) if verbosity >= 2 else accumulate)(acc)
        dist = estimate(acc)

        if afterEst is not None:
            afterEst(dist = dist, it = it)

        if verbosity >= 1:
            print 'trainCGandEM: finished it =', it, 'of CG and EM'
            print 'trainCGandEM:'

    return dist

def mixupLinearGaussianEstimatePartial(acc, estimateChild):
    if isinstance(acc, d.LinearGaussianAcc):
        return acc.estimateInitialMixtureOfTwoExperts()
mixupLinearGaussianEstimate = nodetree.getDagMap([mixupLinearGaussianEstimatePartial, d.defaultEstimatePartial])
