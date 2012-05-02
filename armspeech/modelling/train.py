"""Helper functions for training models."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import dist as d
from minimize import minimize
from armspeech.util.timing import timed

def trainEM(distInit, accumulate, createAcc = d.defaultCreateAcc, estimate = d.defaultEstimate, deltaThresh = 1e-8, minIterations = 1, maxIterations = None, beforeAcc = None, afterAcc = None, afterEst = None, monotone = False, verbosity = 0):
    assert minIterations >= 1
    assert maxIterations is None or maxIterations >= minIterations

    def estimateSubCore(distPrev, logLikePrevPrev):
        if beforeAcc is not None:
            beforeAcc(distPrev)
        acc = createAcc(distPrev)
        accumulate(acc)
        if afterAcc is not None:
            afterAcc(acc)
        dist, logLikePrev, occ = estimate(acc)
        deltaPrev = logLikePrev - logLikePrevPrev
        if monotone and deltaPrev < 0.0:
            raise RuntimeError('log likelihood decreased during expectation-maximization')
        return dist, logLikePrev, deltaPrev, occ

    estimateSub = timed(estimateSubCore) if verbosity >= 2 else estimateSubCore

    it = 0
    dist, logLikePrev, deltaPrev, occ = estimateSub(distInit, float('-inf'))
    if afterEst is not None:
        afterEst(dist = dist, it = it)
    it += 1
    if verbosity >= 2:
        print '(trainEM: (after it', it, ', logLikePrev =', logLikePrev / occ, ', deltaPrev =', deltaPrev / occ, '))'
    while it < minIterations or abs(deltaPrev / occ) > deltaThresh and (maxIterations is None or it < maxIterations):
        dist, logLikePrev, deltaPrev, occ = estimateSub(dist, logLikePrev)
        if afterEst is not None:
            afterEst(dist = dist, it = it)
        it += 1
        if verbosity >= 2:
            print '(trainEM: (after it', it, ', logLikePrev =', logLikePrev / occ, ', deltaPrev =', deltaPrev / occ, '))'
    if verbosity >= 1:
        if abs(deltaPrev / occ) <= deltaThresh:
            print 'trainEM: converged at thresh', deltaThresh, 'in', it, 'iterations'
        else:
            print 'trainEM: did NOT converge at thresh', deltaThresh, 'in', it, 'iterations'
    return dist, logLikePrev, occ

def trainCG(distInit, accumulate, ps = d.defaultParamSpec, length = -50, verbosity = 0):
    def negLogLike_derivParams(params):
        dist = ps.parseAll(distInit, params)
        acc = ps.createAccG(dist)
        accumulate(acc)
        # FIXME : is it better to return logLike or logLike-per-frame? (i.e. for which of these is minimize typically faster?)
        return -acc.logLike(), -ps.derivParams(acc)

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
        print 'trainCG: logLike', -negLogLikes[0], '->', -negLogLikes[-1], '( delta =', negLogLikes[0] - negLogLikes[-1], ')'
        print 'trainCG: (used', lengthUsed, 'function evaluations)'
    dist = ps.parseAll(distInit, params)

    # FIXME : temporary (until all estimate routines return just the dist)
    acc = ps.createAccG(dist)
    accumulate(acc)
    logLike = acc.logLike()
    occ = acc.occ

    return dist, logLike, occ

def trainCGandEM(distInit, accumulate, ps = d.defaultParamSpec, createAccEM = d.defaultCreateAcc, estimate = d.defaultEstimate, iterations = 5, length = -50, afterEst = None, verbosity = 0):
    assert iterations >= 1

    dist = distInit
    for it in range(1, iterations + 1):
        if verbosity >= 1:
            print 'trainCGandEM: starting it =', it, 'of CG and EM'

        dist, trainLogLike, trainOcc = (timed(trainCG) if verbosity >= 2 else trainCG)(dist, accumulate, ps = ps, length = length, verbosity = verbosity)

        acc = createAccEM(dist)
        (timed(accumulate) if verbosity >= 2 else accumulate)(acc)
        dist, trainLogLike, trainOcc = estimate(acc)

        if afterEst is not None:
            afterEst(dist = dist, logLike = trainLogLike, occ = trainOcc, it = it)

        if verbosity >= 1:
            print 'trainCGandEM: finished it =', it, 'of CG and EM'
            print 'trainCGandEM:'

    return dist, trainLogLike, trainOcc

def mixupLinearGaussianEstimatePartial(acc, estimateChild):
    if isinstance(acc, d.LinearGaussianAcc):
        return acc.estimateInitialMixtureOfTwoExperts()
mixupLinearGaussianEstimate = d.getEstimate([mixupLinearGaussianEstimatePartial, d.defaultEstimatePartial])
