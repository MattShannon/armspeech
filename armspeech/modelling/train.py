"""Helper functions for training models.

Several of the functions below implement approximate maximum likelihood
estimation by optimizing the scaled log likelihood. The scale factor used in
these cases is the reciprocal of the "count" for the given dist, unless this
count is less than 1.0, in which case the scale factor used is 1.0. The scale
factor is included to make the logging output produced during optimization more
intuitively comprehensible. The inclusion of the scale factor typically has
little or no impact on the minimization itself.

The special-casing for small counts is necessary since the count may be zero
even when the training set is non-empty (e.g. for autoregressive sequence
distributions where the count is the number of frames rather than the
occupancy). Special-cased values are used only internally and for logging output
in the functions below, and are not part of their return values.
"""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import nodetree
import dist as d
from minimize import minimize
from armspeech.util.timing import timed
from codedep import codeDeps

@codeDeps(d.Rat, d.getDefaultCreateAcc, d.getDefaultEstimateTotAux)
def expectationMaximization(distPrev, accumulate, createAcc = d.getDefaultCreateAcc(), estimateTotAux = d.getDefaultEstimateTotAux(), afterAcc = None, monotoneAux = True, verbosity = 0):
    """Performs one step of expectation maximization.

    See the note in the docstring for this module for information on how the
    log likelihood is scaled. This scaling has no effect on the dist returned
    by this function.
    """
    acc = createAcc(distPrev)
    accumulate(acc)
    if afterAcc is not None:
        afterAcc(acc)
    logLikePrev = acc.logLike()
    count = acc.count()
    count = max(count, 1.0)
    dist, (aux, auxRat) = estimateTotAux(acc)
    if monotoneAux and aux < logLikePrev:
        raise RuntimeError('re-estimated auxiliary value (%s) less than previous log likelihood (%s) during expectation-maximization (count = %s)' % (aux / count, logLikePrev / count, count))
    if verbosity >= 2:
        print 'trainEM:    logLikePrev = %s -> aux = %s (%s) (%s count)' % (logLikePrev / count, aux / count, d.Rat.toString(auxRat), count)
    return dist, logLikePrev, (aux, auxRat), count

@codeDeps(d.getDefaultCreateAcc, d.getDefaultEstimateTotAux,
    expectationMaximization
)
def trainEM(distInit, accumulate, createAcc = d.getDefaultCreateAcc(), estimateTotAux = d.getDefaultEstimateTotAux(), logLikePrevInit = float('-inf'), deltaThresh = 1e-8, minIterations = 1, maxIterations = None, beforeAcc = None, afterAcc = None, afterEst = None, monotone = False, monotoneAux = True, verbosity = 0):
    """Re-estimates a distribution using expectation maximization.

    See the note in the docstring for this module for information on how the
    log likelihood is scaled. This scaling only affects the dist returned by
    this function to the extent that it effectively scales the deltaThresh
    threshold used to assess convergence, and so may sometimes affect the number
    of iterations of expectation maximization performed.
    """
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

# FIXME : try alternative minimizers (e.g. LBFGS, minFunc)
@codeDeps(d.getDefaultParamSpec, minimize)
def trainCG(distInit, accumulate, ps = d.getDefaultParamSpec(), length = -50, verbosity = 0):
    """Re-estimates a distribution using a conjugate gradient optimizer.

    See the note in the docstring for this module for information on how the
    log likelihood is scaled. This scaling is presumed to have only a small
    impact on the dist returned by this function.
    """
    # (FIXME : investigate how large the effect of the scale factor is for
    #   a few example dists?)
    def negLogLike_derivParams(params):
        dist = ps.parseAll(distInit, params)
        acc = ps.createAccG(dist)
        accumulate(acc)
        count = acc.count()
        count = max(count, 1.0)
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

@codeDeps(d.getDefaultCreateAcc, d.getDefaultEstimateTotAux,
    d.getDefaultParamSpec, expectationMaximization, timed, trainCG
)
def trainCGandEM(distInit, accumulate, ps = d.getDefaultParamSpec(), createAccEM = d.getDefaultCreateAcc(), estimateTotAux = d.getDefaultEstimateTotAux(), iterations = 5, length = -50, afterEst = None, verbosity = 0):
    """Re-estimates a distribution using conjugate gradients and EM.

    See the note in the docstring for this module for information on how the
    log likelihood is scaled. This scaling is presumed to have only a small
    impact on the dist returned by this function (via its impact on trainCG).
    """
    assert iterations >= 1

    dist = distInit
    for it in range(1, iterations + 1):
        if verbosity >= 1:
            print 'trainCGandEM: starting it =', it, 'of CG and EM'

        dist = (timed(trainCG) if verbosity >= 2 else trainCG)(dist, accumulate, ps = ps, length = length, verbosity = verbosity)

        dist, _, _, _ = expectationMaximization(dist, accumulate, createAcc = createAccEM, estimateTotAux = estimateTotAux, verbosity = verbosity)

        if afterEst is not None:
            afterEst(dist = dist, it = it)

        if verbosity >= 1:
            print 'trainCGandEM: finished it =', it, 'of CG and EM'
            print 'trainCGandEM:'

    return dist

@codeDeps(d.LinearGaussianAcc, d.estimateInitialMixtureOfTwoExperts)
def mixupLinearGaussianEstimatePartial(acc, estimateChild):
    if isinstance(acc, d.LinearGaussianAcc):
        return d.estimateInitialMixtureOfTwoExperts(acc)
