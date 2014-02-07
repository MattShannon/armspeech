"""Helper functions for distributed model training."""

# Copyright 2011, 2012, 2013, 2014 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.modelling.dist as d
from bisque.distribute import liftLocal, lit, lift
from codedep import codeDeps

@codeDeps()
def accumulate(distPrev, corpus, uttIds, createAcc):
    acc = createAcc(distPrev)
    for uttId in uttIds:
        input, output = corpus.data(uttId)
        acc.add(input, output)
    return acc

@codeDeps(accumulate, d.getDefaultCreateAcc, lift, liftLocal)
def accumulateJobSet(
    distPrevArt,
    corpusArt,
    uttIdChunkArts,
    createAccArt = liftLocal(d.getDefaultCreateAcc)(),
):
    accArts = [ lift(accumulate)(distPrevArt, corpusArt, uttIdChunkArt,
                                 createAccArt)
                for uttIdChunkArt in uttIdChunkArts ]
    return accArts

@codeDeps(d.addAcc)
def estimate(distPrev, createAcc, estimateDist, afterAcc, verbosity, *accs):
    accTot = createAcc(distPrev)
    for acc in accs:
        d.addAcc(accTot, acc)
    if afterAcc is not None:
        afterAcc(accTot)
    logLikePrev = accTot.logLike()
    count = accTot.count()
    count = max(count, 1.0)
    dist = estimateDist(accTot)
    if verbosity >= 2:
        print ('trainEM: logLikePrev = %s (%s count)' %
               (logLikePrev / count, count))
    return dist

@codeDeps(d.getDefaultCreateAcc, d.getDefaultEstimate, estimate, lift,
    liftLocal, lit
)
def estimateJobSet(
    distPrevArt,
    accArts,
    createAccArt = liftLocal(d.getDefaultCreateAcc)(),
    estimateArt = liftLocal(d.getDefaultEstimate)(),
    afterAccArt = lit(None),
    verbosityArt = lit(0),
):
    return lift(estimate)(
        distPrevArt, createAccArt, estimateArt, afterAccArt, verbosityArt,
        *accArts
    )

@codeDeps(d.Rat, d.addAcc)
def estimateWithTotAux(distPrev, createAcc, estimateTotAux, afterAcc,
                       monotoneAux, verbosity, *accs):
    accTot = createAcc(distPrev)
    for acc in accs:
        d.addAcc(accTot, acc)
    if afterAcc is not None:
        afterAcc(accTot)
    logLikePrev = accTot.logLike()
    count = accTot.count()
    count = max(count, 1.0)
    dist, (aux, auxRat) = estimateTotAux(accTot)
    if monotoneAux and aux < logLikePrev:
        raise RuntimeError('re-estimated auxiliary value (%s) less than'
                           ' previous log likelihood (%s) during'
                           ' expectation-maximization (count = %s)' %
                           (aux / count, logLikePrev / count, count))
    if verbosity >= 2:
        auxRatString = d.Rat.toString(auxRat)
        print ('trainEM: logLikePrev = %s -> aux = %s (%s) (%s count)' %
               (logLikePrev / count, aux / count, auxRatString, count))
    return dist

@codeDeps(d.getDefaultCreateAcc, d.getDefaultEstimateTotAux, estimateWithTotAux,
    lift, liftLocal, lit
)
def estimateWithTotAuxJobSet(
    distPrevArt,
    accArts,
    createAccArt = liftLocal(d.getDefaultCreateAcc)(),
    estimateTotAuxArt = liftLocal(d.getDefaultEstimateTotAux)(),
    afterAccArt = lit(None),
    monotoneAuxArt = lit(True),
    verbosityArt = lit(0),
):
    return lift(estimateWithTotAux)(
        distPrevArt, createAccArt, estimateTotAuxArt, afterAccArt,
        monotoneAuxArt, verbosityArt, *accArts
    )

@codeDeps(accumulateJobSet, d.getDefaultCreateAcc, d.getDefaultEstimateTotAux,
    estimateWithTotAuxJobSet, liftLocal, lit
)
def expectationMaximizationJobSet(
    distPrevArt,
    corpusArt,
    uttIdChunkArts,
    createAccArt = liftLocal(d.getDefaultCreateAcc)(),
    estimateTotAuxArt = liftLocal(d.getDefaultEstimateTotAux)(),
    afterAccArt = lit(None),
    monotoneAuxArt = lit(True),
    verbosityArt = lit(0),
):
    """Returns job set to perform one step of expectation maximization."""
    accArts = accumulateJobSet(distPrevArt, corpusArt, uttIdChunkArts,
                               createAccArt)
    distArt = estimateWithTotAuxJobSet(distPrevArt, accArts, createAccArt,
                                       estimateTotAuxArt, afterAccArt,
                                       monotoneAuxArt, verbosityArt)
    return distArt

@codeDeps(d.getDefaultCreateAcc, d.getDefaultEstimateTotAux,
    expectationMaximizationJobSet, liftLocal, lit
)
def trainEMJobSet(
    distInitArt,
    corpusArt,
    uttIdChunkArts,
    numIterationsLit = lit(1),
    createAccArt = liftLocal(d.getDefaultCreateAcc)(),
    estimateTotAuxArt = liftLocal(d.getDefaultEstimateTotAux)(),
    afterAccArt = lit(None),
    monotoneAuxArt = lit(True),
    verbosityArt = lit(0),
):
    numIterations = numIterationsLit.litValue
    distArt = distInitArt
    for it in range(numIterations):
        distArt = expectationMaximizationJobSet(distArt, corpusArt,
                                                uttIdChunkArts, createAccArt,
                                                estimateTotAuxArt, afterAccArt,
                                                monotoneAuxArt, verbosityArt)
    return distArt
