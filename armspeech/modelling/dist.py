"""Probability distributions and their accumulators."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.util.mathhelp import logSum, sigmoid, sampleDiscrete, reprArray
import nodetree
import semiring
import wnet
from armspeech.util.memoize import memoize
from armspeech.util.mathhelp import assert_allclose
from armspeech.util.util import orderedDictRepr
from codedep import codeDeps, ForwardRef

import logging
import math
import numpy as np
import armspeech.numpy_settings
import numpy.linalg as la
import armspeech.util.mylinalg as mla
from scipy import special
import random
from itertools import izip
from armspeech.util.iterhelp import contextualizeIter
from collections import deque

# (FIXME : add more checks to validate Dists and Accs on creation (including
#   checking for NaNs))

def eval_local(reprString):
    # (FIXME : the contents of test_dist affects what must be included here)
    from questions import IdLabelValuer, SubsetQuestion
    from summarizer import VectorSeqSummarizer
    from transform import AddBias, ConstantTransform, IdentityTransform
    from transform import LinearTransform, ShiftOutputTransform
    from transform import VectorizeTransform, DotProductTransform
    from transform import PolynomialTransform1D
    from wnet import ConcreteNet
    from armspeech.util.mathhelp import AsArray

    from numpy import array, zeros, dtype, eye, float64, Inf, inf

    return eval(reprString)

@codeDeps()
class SynthMethod(object):
    Meanish = 0
    Sample = 1

@codeDeps()
class Rat(object):
    Exact = 0
    Approx = 1
    LowerBound = 2

    ratToStringDict = {
        Exact: 'Exact',
        Approx: 'Approx',
        LowerBound: 'LowerBound',
    }

    @staticmethod
    def toString(rat):
        return Rat.ratToStringDict[rat]

@codeDeps(Rat)
def sumRats(rats):
    if any([ rat == Rat.Approx for rat in rats ]):
        return Rat.Approx
    elif any([ rat == Rat.LowerBound for rat in rats ]):
        return Rat.LowerBound
    else:
        assert all([ rat == Rat.Exact for rat in rats ])
        return Rat.Exact

@codeDeps(sumRats)
def sumValuedRats(valuedRats):
    values, rats = zip(*valuedRats)
    return sum(values), sumRats(rats)

@codeDeps(ForwardRef(lambda: AccCommon), nodetree.nodeList)
def accNodeList(parentNode):
    return nodetree.nodeList(
        parentNode,
        includeNode = lambda node: isinstance(node, AccCommon)
    )
@codeDeps(ForwardRef(lambda: Dist), nodetree.nodeList)
def distNodeList(parentNode):
    return nodetree.nodeList(
        parentNode,
        includeNode = lambda node: isinstance(node, Dist)
    )

@codeDeps(distNodeList, nodetree.chainPartialFns, nodetree.getDagMap,
    sumValuedRats
)
def getEstimateTotAux(estimateAuxPartials, idValue = id):
    estimateAuxPartial = nodetree.chainPartialFns(estimateAuxPartials)
    def estimateTotAux(acc):
        auxValuedRats = dict()
        def estimatePartial(acc, estimateChild):
            ret = estimateAuxPartial(acc, estimateChild)
            if ret is None:
                raise RuntimeError('none of the given partial functions was'
                                   ' defined at acc %r' % acc)
            dist, auxValuedRat = ret
            auxValuedRats[idValue(dist)] = auxValuedRat
            return dist
        dist = nodetree.getDagMap([estimatePartial])(acc)
        totAux, totAuxRat = sumValuedRats([
            auxValuedRats[idValue(distNode)]
            for distNode in distNodeList(dist)
        ])

        totAuxAgain, totAuxAgainRat = sumValuedRats(auxValuedRats.values())
        assert np.allclose(totAuxAgain, totAux) and totAuxAgainRat == totAuxRat

        return dist, (totAux, totAuxRat)
    return estimateTotAux

@codeDeps()
def defaultEstimateAuxPartial(acc, estimateChild):
    return acc.estimateAux(estimateChild)
@codeDeps(defaultEstimateAuxPartial, getEstimateTotAux)
def getDefaultEstimateTotAux():
    return getEstimateTotAux([defaultEstimateAuxPartial])

@codeDeps(ForwardRef(lambda: TermAcc))
def defaultEstimateAuxNoRevertPartial(acc, estimateChild):
    if isinstance(acc, TermAcc):
        return acc.estimateSingleAux()
    else:
        return acc.estimateAux(estimateChild)
@codeDeps(defaultEstimateAuxNoRevertPartial, getEstimateTotAux)
def getDefaultEstimateTotAuxNoRevert():
    return getEstimateTotAux([defaultEstimateAuxNoRevertPartial])

@codeDeps()
def defaultEstimatePartial(acc, estimateChild):
    dist, _ = acc.estimateAux(estimateChild)
    return dist
@codeDeps(defaultEstimatePartial, nodetree.getDagMap)
def getDefaultEstimate():
    return nodetree.getDagMap([defaultEstimatePartial])

@codeDeps()
def defaultCreateAccPartial(dist, createAccChild):
    return dist.createAcc(createAccChild)
@codeDeps(defaultCreateAccPartial, nodetree.getDagMap)
def getDefaultCreateAcc():
    return nodetree.getDagMap([defaultCreateAccPartial])

@codeDeps(ForwardRef(lambda: AutoregressiveNetDist))
def verboseNetCreateAccPartial(dist, createAccChild):
    if isinstance(dist, AutoregressiveNetDist):
        return dist.createAcc(createAccChild, verbosity = 3)
@codeDeps(defaultCreateAccPartial, nodetree.getDagMap,
    verboseNetCreateAccPartial
)
def getVerboseNetCreateAcc():
    return nodetree.getDagMap([verboseNetCreateAccPartial,
                               defaultCreateAccPartial])

@codeDeps(nodetree.getDagMap)
def getParams(partialMaps):
    return nodetree.getDagMap(
        partialMaps,
        storeValue = lambda params, args: True,
        restoreValue = lambda b, args: []
    )
@codeDeps(nodetree.getDagMap)
def getDerivParams(partialMaps):
    return nodetree.getDagMap(
        partialMaps,
        storeValue = lambda derivParams, args: True,
        restoreValue = lambda b, args: []
    )
@codeDeps(nodetree.getDagMap)
def getParse(partialMaps):
    return nodetree.getDagMap(
        partialMaps,
        storeValue = lambda (node, paramsLeft), args: node,
        restoreValue = lambda node, args: (node, args[1])
    )
@codeDeps(nodetree.getDagMap)
def getCreateAccG(partialMaps):
    return nodetree.getDagMap(partialMaps)

@codeDeps(getCreateAccG, getDerivParams, getParams, getParse)
class ParamSpec(object):
    def __init__(self, paramsPartials, derivParamsPartials, parsePartials,
                 createAccGPartials):
        self.params = getParams(paramsPartials)
        self.derivParams = getDerivParams(derivParamsPartials)
        self.parse = getParse(parsePartials)
        self.createAccG = getCreateAccG(createAccGPartials)
    def parseAll(self, dist, params):
        distNew, paramsLeft = self.parse(dist, params)
        if len(paramsLeft) != 0:
            raise RuntimeError('extra parameters left after parsing complete')
        return distNew

@codeDeps()
def defaultParamsPartial(node, paramsChild):
    return np.concatenate([node.paramsSingle(),
                           node.paramsChildren(paramsChild)])
@codeDeps()
def defaultDerivParamsPartial(node, derivParamsChild):
    return np.concatenate([node.derivParamsSingle(),
                           node.derivParamsChildren(derivParamsChild)])
@codeDeps()
def defaultParsePartial(node, params, parseChild):
    newNode, paramsLeft = node.parseSingle(params)
    return newNode.parseChildren(paramsLeft, parseChild)
@codeDeps()
def defaultCreateAccGPartial(dist, createAccChild):
    return dist.createAccG(createAccChild)
@codeDeps(ParamSpec, defaultCreateAccGPartial, defaultDerivParamsPartial,
    defaultParamsPartial, defaultParsePartial
)
def getDefaultParamSpec():
    return ParamSpec(
        [defaultParamsPartial],
        [defaultDerivParamsPartial],
        [defaultParsePartial],
        [defaultCreateAccGPartial]
    )

@codeDeps()
def nopParamsPartial(node, paramsChild):
    pass
@codeDeps()
def nopDerivParamsPartial(node, derivParamsChild):
    pass
@codeDeps()
def nopParsePartial(node, params, parseChild):
    pass
@codeDeps()
def nopCreateAccGPartial(dist, createAccChild):
    pass

@codeDeps()
def noLocalParamsPartial(node, paramsChild):
    return node.paramsChildren(paramsChild)
@codeDeps()
def noLocalDerivParamsPartial(node, derivParamsChild):
    return node.derivParamsChildren(derivParamsChild)
@codeDeps()
def noLocalParsePartial(node, params, parseChild):
    return node.parseChildren(params, parseChild)

@codeDeps(nodetree.getDefaultMap)
def isolateDist(dist):
    """Returns an isolated copy of a distribution.

    Creates a new DAG with the same content as the sub-DAG with head dist but
    with fresh objects at each node. Therefore no nodes in the new DAG are
    shared outside the new DAG.
    """
    return nodetree.getDefaultMap()(dist)

@codeDeps(ParamSpec, defaultCreateAccGPartial, defaultDerivParamsPartial,
    defaultParamsPartial, defaultParsePartial, noLocalDerivParamsPartial,
    noLocalParamsPartial, noLocalParsePartial, nopCreateAccGPartial
)
def getByTagParamSpec(f):
    def byTagParamsPartial(node, paramsChild):
        if f(node.tag):
            return defaultParamsPartial(node, paramsChild)
    def byTagDerivParamsPartial(node, derivParamsChild):
        if f(node.tag):
            return defaultDerivParamsPartial(node, derivParamsChild)
    def byTagParsePartial(node, params, parseChild):
        if f(node.tag):
            return defaultParsePartial(node, params, parseChild)
    return ParamSpec(
        [byTagParamsPartial, noLocalParamsPartial],
        [byTagDerivParamsPartial, noLocalDerivParamsPartial],
        [byTagParsePartial, noLocalParsePartial],
        [nopCreateAccGPartial, defaultCreateAccGPartial]
    )

@codeDeps()
def addAcc(accTo, accFrom):
    """Adds accumulator sub-DAG accFrom to accumulator sub-DAG accTo.

    Copes properly with sharing, and raises an exception in the case of invalid
    sharing.
    However assumes accTo is an isolated sub-DAG, i.e. that none of the child
    nodes of accTo are shared with parent nodes outside accTo's sub-DAG (and
    similarly for accFrom), and this method has undefined behaviour if this is
    not true.
    """
    lookup = dict()
    agenda = [(accTo, accFrom)]
    while agenda:
        nodeTo, nodeFrom = agenda.pop()
        identTo = id(nodeTo)
        identFrom = id(nodeFrom)
        if identFrom in lookup:
            assert lookup[identFrom] == identTo
        else:
            lookup[identFrom] = identTo
            nodeTo.addAccSingle(nodeFrom)
            agenda.extend(reversed(nodeTo.addAccChildPairs(nodeFrom)))

@codeDeps()
def parseConcat(dists, params, parseChild):
    distNews = []
    paramsLeft = params
    for dist in dists:
        distNew, paramsLeft = parseChild(dist, paramsLeft)
        distNews.append(distNew)
    return distNews, paramsLeft

@codeDeps()
class PruneSpec(object):
    pass
@codeDeps(PruneSpec)
class SimplePruneSpec(PruneSpec):
    def __init__(self, betaThresh, logOccThresh):
        self.betaThresh = betaThresh
        self.logOccThresh = logOccThresh
    def __repr__(self):
        return 'SimplePruneSpec(%r, %r)' % (self.betaThresh, self.logOccThresh)

@codeDeps(assert_allclose)
class Memo(object):
    def __init__(self, maxOcc):
        self.maxOcc = maxOcc

        self.occ = 0.0
        self.fakeOcc = 0.0
        self.inputs = []
        self.outputs = []

    def add(self, input, output, occ = 1.0):
        if occ != 1.0:
            raise RuntimeError('Memo occupancies must be 1.0')
        self.occ += occ
        if self.maxOcc is None or len(self.inputs) < self.maxOcc:
            self.fakeOcc += occ
            self.inputs.append(input)
            self.outputs.append(output)
        elif random.random() * self.occ < self.fakeOcc:
            # (FIXME : behind the scenes, only do subset selection every
            #   certain number of inputs (for efficiency)?)
            assert len(self.inputs) == self.maxOcc
            delIndex = random.randrange(self.maxOcc)
            self.inputs[delIndex] = input
            self.outputs[delIndex] = output
        assert_allclose(self.fakeOcc, len(self.inputs))

    # FIXME : do random subset selection for addAcc too
    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.fakeOcc += acc.fakeOcc
        self.inputs += acc.inputs
        self.outputs += acc.outputs
        if self.maxOcc is not None and self.fakeOcc > self.maxOcc:
            self.fakeOcc = self.maxOcc
            self.inputs = self.inputs[:self.maxOcc]
            self.outputs = self.outputs[:self.maxOcc]

@codeDeps()
class EstimationError(Exception):
    pass

@codeDeps()
class InvalidParamsError(Exception):
    pass

@codeDeps()
class SynthSeqTooLongError(Exception):
    pass

@codeDeps(accNodeList)
class AccCommon(object):
    """A common baseclass for an accumulator.

    Note that in subclasses addAccSingle (and addAccChildPairs) should be such
    that the overall accumulator addition implemented by addAcc is associative
    and commutative (this should happen naturally, but we mention this here
    to be explicit).
    """
    def children(self):
        abstract
    def mapChildren(self, mapChild):
        raise RuntimeError('mapChildren not defined for accumulator nodes')
    #@property
    #def occ(self):
    #    abstract
    def add(self, input, output, occ = 1.0):
        abstract
    # (FIXME : for all of the Accs defined below, add more checks that acc is
    #   of the right type during addAccSingle?)
    def addAccSingle(self, acc):
        abstract
    def addAccChildPairs(self, acc):
        selfChildren = self.children()
        accChildren = acc.children()
        assert len(selfChildren) == len(accChildren)
        return zip(selfChildren, accChildren)
    def count(self):
        return self.occ
    def logLikeSingle(self):
        abstract
    def logLike(self):
        return sum([ accNode.logLikeSingle()
                     for accNode in accNodeList(self) ])
    def withTag(self, tag):
        """Set tag and return self.

        This is intended to be used immediately after object creation, such as:

            acc = SomeAcc([2.0, 3.0, 4.0]).withTag('hi')
        """
        self.tag = tag
        return self

@codeDeps(AccCommon)
class AccEM(AccCommon):
    def estimateAux(self, estimateChild):
        abstract

@codeDeps(AccCommon)
class AccG(AccCommon):
    def derivParamsSingle(self):
        abstract
    def derivParamsChildren(self, derivParamsChild):
        children = self.children()
        if children:
            return np.concatenate([ derivParamsChild(child)
                                    for child in children ])
        else:
            return []

@codeDeps(AccEM, AccG)
class Acc(AccEM, AccG):
    pass

@codeDeps(Acc, EstimationError, Rat)
class TermAcc(Acc):
    """Acc with no children."""
    def children(self):
        return []

    def estimateSingleAux(self):
        abstract

    def estimateSingleAuxSafe(self):
        """A version of estimateSingleAux which tries to recover from errors.

        Tries to revert to the previous dist self.distPrev when
        estimateSingleAux raises an EstimationError.
        """
        try:
            return self.estimateSingleAux()
        except EstimationError, detail:
            if not hasattr(self, 'distPrev') or self.distPrev is None:
                raise
            else:
                logging.warning('reverting to previous dist due to error'
                                ' during %s estimation: %s' %
                                (self.distPrev.__class__.__name__, detail))
                distNew = self.distPrev.mapChildren(None)
                return distNew, (self.logLikeSingle(), Rat.Exact)

    def estimateAux(self, estimateChild):
        return self.estimateSingleAuxSafe()

@codeDeps(AccG)
class DerivTermAccG(AccG):
    def __init__(self, distPrev, tag = None):
        assert len(distPrev.children()) == 0
        self.distPrev = distPrev
        self.tag = tag

        self.occ = 0.0
        self.logLikePrev = 0.0
        self.derivParams = np.zeros([len(distPrev.paramsSingle())])

    def children(self):
        return []

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.logLikePrev += self.distPrev.logProb(input, output) * occ
        self.derivParams += self.distPrev.logProbDerivParams(input,
                                                             output) * occ

    # N.B. assumes distPrev is the same for self and acc (not checked).
    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.logLikePrev += acc.logLikePrev
        self.derivParams += acc.derivParams

    def logLikeSingle(self):
        return self.logLikePrev

    def derivParamsSingle(self):
        return self.derivParams

@codeDeps(ForwardRef(lambda: FixedValueDist), Rat, TermAcc)
class FixedValueAcc(TermAcc):
    def __init__(self, value, tag = None):
        self.value = value
        self.tag = tag

        self.occ = 0.0

    def add(self, input, output, occ = 1.0):
        if output != self.value:
            raise RuntimeError('output %r != fixed value %r for'
                               ' FixedValueAcc' % (output, self.value))
        self.occ += occ

    def addAccSingle(self, acc):
        assert self.value == acc.value
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateSingleAux(self):
        return FixedValueDist(self.value, tag = self.tag), (0.0, Rat.Exact)

@codeDeps(ForwardRef(lambda: OracleDist), Rat, TermAcc)
class OracleAcc(TermAcc):
    def __init__(self, tag = None):
        self.tag = tag

        self.occ = 0.0

    def add(self, input, output, occ = 1.0):
        self.occ += occ

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateSingleAux(self):
        return OracleDist(tag = self.tag), (0.0, Rat.Exact)

@codeDeps(EstimationError, ForwardRef(lambda: LinearGaussian), Rat, TermAcc,
    mla.pinv
)
class LinearGaussianAcc(TermAcc):
    def __init__(self, distPrev = None, inputLength = None,
                 varianceFloor = None, tag = None):
        self.distPrev = distPrev
        if distPrev is not None:
            inputLength = len(distPrev.coeff)
        assert inputLength is not None and inputLength >= 0
        if varianceFloor is not None:
            self.varianceFloor = varianceFloor
        else:
            if distPrev is not None:
                self.varianceFloor = distPrev.varianceFloor
            else:
                self.varianceFloor = 0.0
        self.tag = tag

        self.occ = 0.0
        self.sumSqr = 0.0
        self.sumTarget = np.zeros([inputLength])
        self.sumOuter = np.zeros([inputLength, inputLength])

        assert self.varianceFloor is not None
        assert self.varianceFloor >= 0.0

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.sumSqr += (output ** 2) * occ
        self.sumTarget += input * output * occ
        self.sumOuter += np.outer(input, input) * occ

    # N.B. assumes distPrev (if present) is the same for self and acc (not
    #   checked).
    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.sumSqr += acc.sumSqr
        self.sumTarget += acc.sumTarget
        self.sumOuter += acc.sumOuter

    def auxFn(self, coeff, variance):
        term = (self.sumSqr - 2.0 * np.dot(self.sumTarget, coeff) +
                np.dot(np.dot(self.sumOuter, coeff), coeff))
        aux = (-0.5 * math.log(2.0 * math.pi) * self.occ +
               -0.5 * math.log(variance) * self.occ - 0.5 * term / variance)
        return aux, Rat.Exact

    def logLikeSingle(self):
        return self.auxFn(self.distPrev.coeff, self.distPrev.variance)[0]

    def auxDerivParams(self, coeff, variance):
        term = (self.sumSqr - 2.0 * np.dot(self.sumTarget, coeff) +
                np.dot(np.dot(self.sumOuter, coeff), coeff))
        derivCoeff = (self.sumTarget - np.dot(self.sumOuter, coeff)) / variance
        derivLogPrecision = 0.5 * self.occ - 0.5 * term / variance
        return np.append(derivCoeff, derivLogPrecision), Rat.Exact

    def derivParamsSingle(self):
        return self.auxDerivParams(self.distPrev.coeff,
                                   self.distPrev.variance)[0]

    def estimateSingleAux(self):
        if self.occ == 0.0:
            raise EstimationError('require occ > 0')
        try:
            sumOuterInv = mla.pinv(self.sumOuter)
        except la.LinAlgError, detail:
            raise EstimationError('could not compute pseudo-inverse: %s' %
                                  detail)
        coeff = np.dot(sumOuterInv, self.sumTarget)
        variance = (self.sumSqr - np.dot(coeff, self.sumTarget)) / self.occ

        if variance < self.varianceFloor:
            variance = self.varianceFloor

        if variance <= 0.0:
            raise EstimationError('computed variance is zero or negative: %r' %
                                  variance)
        elif variance < 1e-10:
            raise EstimationError('computed variance too miniscule (variances'
                                  ' this small can lead to substantial loss of'
                                  ' precision during accumulation): %r' %
                                  variance)
        distNew = LinearGaussian(coeff, variance, self.varianceFloor,
                                 tag = self.tag)
        return distNew, self.auxFn(coeff, variance)

@codeDeps(EstimationError, ForwardRef(lambda: LinearGaussianVec), Rat, TermAcc,
    mla.pinv
)
class LinearGaussianVecAcc(TermAcc):
    def __init__(self, distPrev, tag = None):
        self.distPrev = distPrev
        self.tag = tag

        self.varianceFloorVec = self.distPrev.varianceFloorVec

        order, inputLength = np.shape(self.distPrev.coeffVec)
        self.occ = 0.0
        self.sumOuter = np.zeros((order, inputLength + 1, inputLength + 1))

        assert self.varianceFloorVec is not None
        assert np.all(self.varianceFloorVec >= 0.0)

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        inputOutput = np.concatenate((input, np.reshape(output, (1, -1))),
                                     axis = 0)
        self.sumOuter += np.einsum(
            inputOutput, [1, 0],
            inputOutput, [2, 0],
            [0, 1, 2]
        ) * occ

    # N.B. assumes distPrev (if present) is the same for self and acc (not
    #   checked).
    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.sumOuter += acc.sumOuter

    def auxFn(self, coeffVec, varianceVec):
        # (FIXME : could make this a bit nicer?)

        order, inputLength = np.shape(self.distPrev.coeffVec)
        sumSqrVec = self.sumOuter[:, inputLength, inputLength]
        sumTargetVec = self.sumOuter[:, inputLength, :inputLength]
        sumOuterInputVec = self.sumOuter[:, :inputLength, :inputLength]

        auxes = []
        for vecIndex in range(order):
            sumSqr = sumSqrVec[vecIndex]
            sumTarget = sumTargetVec[vecIndex]
            sumOuterInput = sumOuterInputVec[vecIndex]
            coeff = coeffVec[vecIndex]
            variance = varianceVec[vecIndex]

            term = (sumSqr - 2.0 * np.dot(sumTarget, coeff) +
                    np.dot(np.dot(sumOuterInput, coeff), coeff))
            aux = (-0.5 * math.log(2.0 * math.pi) * self.occ +
                   -0.5 * math.log(variance) * self.occ +
                   -0.5 * term / variance)
            auxes.append(aux)

        return np.sum(auxes), Rat.Exact

    def logLikeSingle(self):
        return self.auxFn(self.distPrev.coeffVec, self.distPrev.varianceVec)[0]

    def auxDerivParams(self, coeffVec, varianceVec):
        # (FIXME : could make this a bit nicer?)

        order, inputLength = np.shape(self.distPrev.coeffVec)
        sumSqrVec = self.sumOuter[:, inputLength, inputLength]
        sumTargetVec = self.sumOuter[:, inputLength, :inputLength]
        sumOuterInputVec = self.sumOuter[:, :inputLength, :inputLength]

        derivCoeffVec = []
        derivLogPrecisionVec = []
        for vecIndex in range(order):
            sumSqr = sumSqrVec[vecIndex]
            sumTarget = sumTargetVec[vecIndex]
            sumOuterInput = sumOuterInputVec[vecIndex]
            coeff = coeffVec[vecIndex]
            variance = varianceVec[vecIndex]

            term = (sumSqr - 2.0 * np.dot(sumTarget, coeff) +
                    np.dot(np.dot(sumOuterInput, coeff), coeff))
            derivCoeff = (sumTarget - np.dot(sumOuterInput, coeff)) / variance
            derivLogPrecision = 0.5 * self.occ - 0.5 * term / variance
            derivCoeffVec.append(derivCoeff)
            derivLogPrecisionVec.append(derivLogPrecision)

        return np.append(derivCoeffVec, derivLogPrecisionVec), Rat.Exact

    def derivParamsSingle(self):
        return self.auxDerivParams(self.distPrev.coeffVec,
                                   self.distPrev.varianceVec)[0]

    def estimateSingleAux(self):
        if self.occ == 0.0:
            raise EstimationError('require occ > 0')

        # (FIXME : this is not quite equivalent to an array of LinearGaussian
        #   objects in the case that EstimationError errors are thrown and
        #   caught. Not sure this is a problem, though.)

        # (FIXME : could make this a bit nicer?)

        order, inputLength = np.shape(self.distPrev.coeffVec)
        sumSqrVec = self.sumOuter[:, inputLength, inputLength]
        sumTargetVec = self.sumOuter[:, inputLength, :inputLength]
        sumOuterInputVec = self.sumOuter[:, :inputLength, :inputLength]

        coeffVec = []
        varianceVec = []
        for vecIndex in range(order):
            sumSqr = sumSqrVec[vecIndex]
            sumTarget = sumTargetVec[vecIndex]
            sumOuterInput = sumOuterInputVec[vecIndex]
            varianceFloor = self.varianceFloorVec[vecIndex]

            try:
                sumOuterInv = mla.pinv(sumOuterInput)
            except la.LinAlgError, detail:
                raise EstimationError('could not compute pseudo-inverse: %s' %
                                      detail)
            coeff = np.dot(sumOuterInv, sumTarget)
            variance = (sumSqr - np.dot(coeff, sumTarget)) / self.occ

            if variance < varianceFloor:
                variance = varianceFloor

            if variance <= 0.0:
                raise EstimationError('computed variance is zero or negative: %r' %
                                      variance)
            elif variance < 1e-10:
                raise EstimationError('computed variance too miniscule (variances'
                                      ' this small can lead to substantial loss of'
                                      ' precision during accumulation): %r' %
                                      variance)

            coeffVec.append(coeff)
            varianceVec.append(variance)
        coeffVec = np.asarray(coeffVec)
        varianceVec = np.asarray(varianceVec)
        if coeffVec.size == 0:
            coeffVec = np.zeros((order, inputLength))
        if varianceVec.size == 0:
            varianceVec = np.zeros((order,))

        distNew = LinearGaussianVec(coeffVec, varianceVec,
                                    self.varianceFloorVec, tag = self.tag)
        return distNew, self.auxFn(coeffVec, varianceVec)

@codeDeps(ForwardRef(lambda: ConstantClassifier), EstimationError, Rat, TermAcc,
    assert_allclose
)
class ConstantClassifierAcc(TermAcc):
    def __init__(self, distPrev = None, numClasses = None, probFloors = None,
                 tag = None):
        self.distPrev = distPrev
        if distPrev is not None:
            numClasses = len(distPrev.probs)
        assert numClasses >= 1
        if probFloors is not None:
            self.probFloors = probFloors
        else:
            if distPrev is not None:
                self.probFloors = distPrev.probFloors
            else:
                self.probFloors = np.zeros((numClasses,))
        self.tag = tag

        self.occ = 0.0
        self.occs = np.zeros([numClasses])

        assert self.probFloors is not None
        assert len(self.probFloors) == len(self.occs)
        assert all(self.probFloors >= 0.0)
        assert sum(self.probFloors) <= 1.0

    def add(self, input, classIndex, occ = 1.0):
        self.occ += occ
        self.occs[classIndex] += occ

    # N.B. assumes class 0 in self corresponds to class 0 in acc, etc.
    #   Also assumes distPrev (if present) is the same for self and acc (not
    #   checked).
    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.occs += acc.occs

    def auxFn(self, probs):
        aux = sum([ occ * logProb
                    for occ, logProb in zip(self.occs, np.log(probs))
                    if occ > 0.0 ])
        return aux, Rat.Exact

    def logLikeSingle(self):
        return self.auxFn(self.distPrev.probs)[0]

    def auxDerivParams(self, probs):
        auxDeriv = ((self.occs[:-1] - self.occs[-1]) -
                    (probs[:-1] - probs[-1]) * self.occ)
        return auxDeriv, Rat.Exact

    def derivParamsSingle(self):
        return self.auxDerivParams(self.distPrev.probs)[0]

    def estimateSingleAux(self):
        if self.occ == 0.0:
            raise EstimationError('require occ > 0')
        probs = self.occs / self.occ

        # find the probs which maximize the auxiliary function, subject to
        #   the given flooring constraints
        # FIXME : think more about maths of flooring procedure below. It
        #   is guaranteed to terminate, but think there are cases (for more
        #   than 2 classes) where it doesn't find the constrained ML
        #   optimum.
        floored = (probs < self.probFloors)
        done = False
        while not done:
            probsBelow = self.probFloors * floored
            probsAbove = probs * (-floored)
            probsAbove = probsAbove / sum(probsAbove) * (1.0 - sum(probsBelow))
            flooredOld = floored
            floored = floored + (probsAbove < self.probFloors)
            done = all(flooredOld == floored)
        probs = probsBelow + probsAbove
        assert_allclose(sum(probs), 1.0)
        assert all(probs >= self.probFloors)

        distNew = ConstantClassifier(probs, self.probFloors, tag = self.tag)
        return distNew, self.auxFn(probs)

@codeDeps(ForwardRef(lambda: BinaryLogisticClassifier), EstimationError, Rat,
    TermAcc, mla.pinv
)
class BinaryLogisticClassifierAcc(TermAcc):
    def __init__(self, distPrev, tag = None):
        self.distPrev = distPrev
        self.tag = tag

        dim = len(self.distPrev.coeff)
        self.occ = 0.0
        self.sumTarget = np.zeros([dim])
        self.sumOuter = np.zeros([dim, dim])
        self.logLikePrev = 0.0

    def add(self, input, classIndex, occ = 1.0):
        if occ > 0.0:
            probPrev1 = self.distPrev.prob(input, 1)
            probPrevProduct = probPrev1 * (1.0 - probPrev1)
            self.occ += occ
            self.sumTarget += input * (probPrev1 - classIndex) * occ
            self.sumOuter += np.outer(input, input) * probPrevProduct * occ
            self.logLikePrev += self.distPrev.logProb(input, classIndex) * occ

    # N.B. assumes class 0 in self corresponds to class 0 in acc, etc.
    # (FIXME : accumulated values encode a local quadratic approx of likelihood
    #   function at current params. However should the origin in parameter
    #   space be treated as absolute zero rather than the current params?
    #   Would allow decision tree clustering with BinaryLogisticClassifier
    #   (although quadratic approx may not be very good in this situation).)
    def addAccSingle(self, acc):
        assert np.all(self.distPrev.coeff == acc.distPrev.coeff)
        self.occ += acc.occ
        self.sumTarget += acc.sumTarget
        self.sumOuter += acc.sumOuter
        self.logLikePrev += acc.logLikePrev

    def auxFn(self, coeff):
        coeffDelta = coeff - self.distPrev.coeff
        if np.all(coeffDelta == 0.0):
            return self.logLikePrev, Rat.Exact
        else:
            targetTerm = -np.dot(self.sumTarget, coeffDelta)
            outerTerm = -0.5 * np.dot(np.dot(self.sumOuter, coeffDelta),
                                      coeffDelta)
            aux = self.logLikePrev + targetTerm + outerTerm
            return aux, Rat.Approx

    def logLikeSingle(self):
        return self.logLikePrev

    def derivParamsSingle(self):
        return -self.sumTarget

    # (FIXME : estimation doesn't always converge, even in the case where
    #   classes are not linearly separable and we have a clearly defined
    #   maximum. Come up with a better procedure? For example, could
    #   say that if current update decreases log likelihood, then take a
    #   half-step and try again (tho N.B. requires tracking previous log like
    #   somehow). Does this always converge? Could also try Firth adjustment,
    #   or other forms of regularization (though conceptually this is solving
    #   a different problem -- shouldn't have to use any regularization to get
    #   the nice non-linearly-separable case to work!).)
    def estimateSingleAux(self):
        if self.occ == 0.0:
            raise EstimationError('require occ > 0')
        try:
            sumOuterInv = mla.pinv(self.sumOuter)
        except la.LinAlgError, detail:
            raise EstimationError('could not compute pseudo-inverse: %s' %
                                  detail)
        coeffDelta = -np.dot(sumOuterInv, self.sumTarget)

        # approximate constrained maximum likelihood
        step = 0.7
        while any(np.abs(self.distPrev.coeff + coeffDelta * step) >
                  self.distPrev.coeffFloor):
            step *= 0.5
        coeff = self.distPrev.coeff + coeffDelta * step
        assert all(np.abs(coeff) <= self.distPrev.coeffFloor)

        distNew = BinaryLogisticClassifier(coeff, self.distPrev.coeffFloor,
                                           tag = self.tag)
        return distNew, self.auxFn(coeff)

@codeDeps(Acc, ForwardRef(lambda: MixtureDist), Rat, assert_allclose, logSum)
class MixtureAcc(Acc):
    def __init__(self, distPrev, classAcc, regAccs, tag = None):
        self.numComps = distPrev.numComps
        self.distPrev = distPrev
        self.classAcc = classAcc
        self.regAccs = regAccs
        self.tag = tag

        self.occ = 0.0
        self.entropy = 0.0

    def children(self):
        return [self.classAcc] + self.regAccs

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        logProbs = [ self.distPrev.logProbComp(input, comp, output)
                     for comp in range(self.numComps) ]
        logTot = logSum(logProbs)
        relOccs = np.exp(logProbs - logTot)
        assert_allclose(sum(relOccs), 1.0)
        for comp in range(self.numComps):
            relOcc = relOccs[comp]
            if relOcc > 0.0:
                self.classAcc.add(input, comp, occ * relOcc)
                self.regAccs[comp].add(input, output, occ * relOcc)
                self.entropy -= occ * relOcc * math.log(relOcc)

    # N.B. assumes component 0 in self corresponds to component 0 in acc, etc.
    #   Also assumes distPrev is the same for self and acc (not checked).
    def addAccSingle(self, acc):
        assert self.numComps == acc.numComps
        self.occ += acc.occ
        self.entropy += acc.entropy

    def logLikeSingle(self):
        return self.entropy

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        classDist = estimateChild(self.classAcc)
        regDists = [ estimateChild(regAcc) for regAcc in self.regAccs ]
        distNew = MixtureDist(classDist, regDists, tag = self.tag)
        return distNew, (self.entropy, Rat.LowerBound)

@codeDeps(Acc, ForwardRef(lambda: IdentifiableMixtureDist), Rat)
class IdentifiableMixtureAcc(Acc):
    def __init__(self, classAcc, regAccs, tag = None):
        self.classAcc = classAcc
        self.regAccs = regAccs
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.classAcc] + self.regAccs

    def add(self, input, output, occ = 1.0):
        comp, acOutput = output
        self.occ += occ
        self.classAcc.add(input, comp, occ)
        self.regAccs[comp].add(input, acOutput, occ)

    def addAccSingle(self, acc):
        assert len(self.regAccs) == len(acc.regAccs)
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        classDist = estimateChild(self.classAcc)
        regDists = [ estimateChild(regAcc) for regAcc in self.regAccs ]
        distNew = IdentifiableMixtureDist(classDist, regDists, tag = self.tag)
        return distNew, (0.0, Rat.Exact)

@codeDeps(ForwardRef(lambda: VectorAcc))
def createVectorAcc(order, outIndices, vectorSummarizer, createAccForIndex):
    accComps = dict()
    for outIndex in outIndices:
        accComps[outIndex] = createAccForIndex(outIndex)
    return VectorAcc(order, vectorSummarizer, outIndices, accComps)

@codeDeps(Acc, Rat, ForwardRef(lambda: VectorDist))
class VectorAcc(Acc):
    def __init__(self, order, vectorSummarizer, keys, accComps, tag = None):
        assert len(keys) == len(accComps)
        for key in keys:
            assert key in accComps
        self.order = order
        self.vectorSummarizer = vectorSummarizer
        self.keys = keys
        self.accComps = accComps
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [ self.accComps[key] for key in self.keys ]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        for outIndex in self.accComps:
            summary = self.vectorSummarizer(input, output[:outIndex], outIndex)
            self.accComps[outIndex].add(summary, output[outIndex], occ)

    def addAccSingle(self, acc):
        assert self.order == acc.order
        assert self.keys == acc.keys
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        distComps = dict()
        for outIndex in self.accComps:
            distComps[outIndex] = estimateChild(self.accComps[outIndex])
        distNew = VectorDist(self.order, self.vectorSummarizer, self.keys,
                             distComps, tag = self.tag)
        return distNew, (0.0, Rat.Exact)

@codeDeps(ForwardRef(lambda: DiscreteAcc))
def createDiscreteAcc(keys, createAccFor):
    accDict = dict()
    for key in keys:
        accDict[key] = createAccFor(key)
    return DiscreteAcc(keys, accDict)

@codeDeps(Acc, ForwardRef(lambda: DiscreteDist), Rat)
class DiscreteAcc(Acc):
    def __init__(self, keys, accDict, tag = None):
        assert len(keys) == len(accDict)
        for key in keys:
            assert key in accDict
        self.keys = keys
        self.accDict = accDict
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [ self.accDict[key] for key in self.keys ]

    def add(self, input, output, occ = 1.0):
        label, acInput = input
        self.occ += occ
        self.accDict[label].add(acInput, output, occ)

    def addAccSingle(self, acc):
        assert self.keys == acc.keys
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        distDict = dict()
        for label in self.accDict:
            distDict[label] = estimateChild(self.accDict[label])
        distNew = DiscreteDist(self.keys, distDict, tag = self.tag)
        return distNew, (0.0, Rat.Exact)

@codeDeps(Acc)
class AutoGrowingDiscreteAcc(Acc):
    """A discrete accumulator that creates sub-accumulators as necessary.

    Sub-accumulators are created whenever a new phonetic context is seen.

    (N.B. the accumulator sub-DAGs created by createAcc should probably not
    have any nodes which are shared outside that sub-DAG. (Could think about
    more carefully if we ever have a use case).)
    """
    def __init__(self, createAcc, tag = None):
        self.accDict = dict()
        self.createAcc = createAcc
        self.tag = tag

        self.occ = 0.0

    def children(self):
        # (FIXME : the order of the result here depends on hash map details, so
        #   could get different secHashes for resulting pickled files. Probably
        #   not an issue, but if it was, could solve by sorting based on key.)
        return self.accDict.values()

    def add(self, input, output, occ = 1.0):
        label, acInput = input
        self.occ += occ
        if not label in self.accDict:
            self.accDict[label] = self.createAcc()
        self.accDict[label].add(acInput, output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def addAccChildPairs(self, acc):
        ret = []
        for label in acc.accDict:
            if not label in self.accDict:
                self.accDict[label] = self.createAcc()
            ret.append((self.accDict[label], acc.accDict[label]))
        return ret

@codeDeps(Acc)
class DecisionTreeAcc(Acc):
    pass

@codeDeps(DecisionTreeAcc, ForwardRef(lambda: DecisionTreeNode), Rat)
class DecisionTreeAccNode(DecisionTreeAcc):
    def __init__(self, fullQuestion, accYes, accNo, tag = None):
        self.fullQuestion = fullQuestion
        self.accYes = accYes
        self.accNo = accNo
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.accYes, self.accNo]

    def add(self, input, output, occ = 1.0):
        label, acInput = input
        self.occ += occ
        labelValuer, question = self.fullQuestion
        if question(labelValuer(label)):
            self.accYes.add(input, output, occ)
        else:
            self.accNo.add(input, output, occ)

    def addAccSingle(self, acc):
        assert self.fullQuestion == acc.fullQuestion
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        distYes = estimateChild(self.accYes)
        distNo = estimateChild(self.accNo)
        distNew = DecisionTreeNode(self.fullQuestion, distYes, distNo,
                                   tag = self.tag)
        return distNew, (0.0, Rat.Exact)

@codeDeps(DecisionTreeAcc, ForwardRef(lambda: DecisionTreeLeaf), Rat)
class DecisionTreeAccLeaf(DecisionTreeAcc):
    def __init__(self, acc, tag = None):
        self.acc = acc
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, output, occ = 1.0):
        label, acInput = input
        self.occ += occ
        self.acc.add(acInput, output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        return DecisionTreeLeaf(dist, tag = self.tag), (0.0, Rat.Exact)

@codeDeps(Acc, ForwardRef(lambda: MappedInputDist), Rat)
class MappedInputAcc(Acc):
    """Acc where input is mapped using a fixed transform."""
    def __init__(self, inputTransform, acc, tag = None):
        self.inputTransform = inputTransform
        self.acc = acc
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.acc.add(self.inputTransform(input), output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        overallDistNew = MappedInputDist(self.inputTransform, dist,
                                         tag = self.tag)
        return overallDistNew, (0.0, Rat.Exact)

@codeDeps(Acc, ForwardRef(lambda: MappedOutputDist), Rat)
class MappedOutputAcc(Acc):
    """Acc where output is mapped using a fixed transform."""
    def __init__(self, outputTransform, acc, tag = None):
        self.outputTransform = outputTransform
        self.acc = acc
        self.tag = tag

        self.occ = 0.0
        self.logJac = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.acc.add(input, self.outputTransform(input, output), occ)
        self.logJac += self.outputTransform.logJac(input, output) * occ

    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.logJac += acc.logJac

    def logLikeSingle(self):
        return self.logJac

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        overallDistNew = MappedOutputDist(self.outputTransform, dist,
                                          tag = self.tag)
        return overallDistNew, (self.logJac, Rat.Exact)

@codeDeps(AccEM, Rat, ForwardRef(lambda: TransformedInputDist))
class TransformedInputLearnDistAccEM(AccEM):
    """Acc for transformed input, where we learn the sub-dist using EM."""
    def __init__(self, inputTransform, acc, tag = None):
        self.inputTransform = inputTransform
        self.acc = acc
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.acc.add(self.inputTransform(input), output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        overallDistNew = TransformedInputDist(self.inputTransform, dist,
                                              tag = self.tag)
        return overallDistNew, (0.0, Rat.Exact)

@codeDeps(AccEM, Rat, ForwardRef(lambda: TransformedInputDist))
class TransformedInputLearnTransformAccEM(AccEM):
    """Acc for transformed input, where we learn the transform using EM."""
    def __init__(self, inputTransformAcc, dist, tag = None):
        self.inputTransformAcc = inputTransformAcc
        self.dist = dist
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.inputTransformAcc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.inputTransformAcc.add((self.dist, input), output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def estimateAux(self, estimateChild):
        inputTransform = estimateChild(self.inputTransformAcc)
        overallDistNew = TransformedInputDist(inputTransform, self.dist,
                                              tag = self.tag)
        return overallDistNew, (0.0, Rat.Exact)

@codeDeps(AccG)
class TransformedInputAccG(AccG):
    """Acc for transformed input, where we compute the gradient.

    The gradient is computed with respect to the parameters of both the
    transform and the sub-dist.
    """
    def __init__(self, (inputTransformAcc, inputTransform), (acc, dist),
                 tag = None):
        self.inputTransformAcc = inputTransformAcc
        self.inputTransform = inputTransform
        self.acc = acc
        self.dist = dist
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.inputTransformAcc, self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.inputTransformAcc.add((self.dist, input), output, occ)
        self.acc.add(self.inputTransform(input), output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

@codeDeps(AccEM, Rat, ForwardRef(lambda: TransformedOutputDist))
class TransformedOutputLearnDistAccEM(AccEM):
    """Acc for transformed output, where we learn the sub-dist using EM."""
    def __init__(self, outputTransform, acc, tag = None):
        self.outputTransform = outputTransform
        self.acc = acc
        self.tag = tag

        self.occ = 0.0
        self.logJac = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.acc.add(input, self.outputTransform(input, output), occ)
        self.logJac += self.outputTransform.logJac(input, output) * occ

    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.logJac += acc.logJac

    def logLikeSingle(self):
        return self.logJac

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        overallDistNew = TransformedOutputDist(self.outputTransform, dist,
                                               tag = self.tag)
        return overallDistNew, (self.logJac, Rat.Exact)

@codeDeps(AccEM, Rat, ForwardRef(lambda: TransformedOutputDist))
class TransformedOutputLearnTransformAccEM(AccEM):
    """Acc for transformed output, where we learn the transform using EM."""
    def __init__(self, outputTransformAcc, dist, tag = None):
        self.outputTransformAcc = outputTransformAcc
        self.dist = dist
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.outputTransformAcc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.outputTransformAcc.add((self.dist, input), output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def estimateAux(self, estimateChild):
        outputTransform = estimateChild(self.outputTransformAcc)
        overallDistNew = TransformedOutputDist(outputTransform, self.dist,
                                               tag = self.tag)
        return overallDistNew, (0.0, Rat.Exact)

@codeDeps(AccG)
class TransformedOutputAccG(AccG):
    """Acc for transformed output, where we compute the gradient.

    The gradient is computed with respect to the parameters of both the
    transform and the sub-dist.
    """
    def __init__(self, (outputTransformAcc, outputTransform), (acc, dist),
                 tag = None):
        self.outputTransformAcc = outputTransformAcc
        self.outputTransform = outputTransform
        self.acc = acc
        self.dist = dist
        self.tag = tag

        self.occ = 0.0
        # (FIXME : should logJac tracking go in outputTransformAcc instead?)
        self.logJac = 0.0

    def children(self):
        return [self.outputTransformAcc, self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.outputTransformAcc.add((self.dist, input), output, occ)
        self.acc.add(input, self.outputTransform(input, output), occ)
        self.logJac += self.outputTransform.logJac(input, output) * occ

    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.logJac += acc.logJac

    def logLikeSingle(self):
        return self.logJac

    def derivParamsSingle(self):
        return []

@codeDeps(Acc, ForwardRef(lambda: PassThruDist), Rat)
class PassThruAcc(Acc):
    def __init__(self, acc, tag = None):
        self.acc = acc
        self.tag = tag

        self.occ = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, output, occ = 1.0):
        self.occ += occ
        self.acc.add(input, output, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        return PassThruDist(dist, tag = self.tag), (0.0, Rat.Exact)

@codeDeps(Acc, ForwardRef(lambda: CountFramesDist), Rat)
class CountFramesAcc(Acc):
    def __init__(self, acc, tag = None):
        self.acc = acc
        self.tag = tag

        self.occ = 0.0
        self.frames = 0.0

    def children(self):
        return [self.acc]

    def add(self, input, outSeq, occ = 1.0):
        self.occ += occ
        self.frames += len(outSeq) * occ
        self.acc.add(input, outSeq, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.frames += acc.frames

    def count(self):
        return self.frames

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        return CountFramesDist(dist, tag = self.tag), (0.0, Rat.Exact)

@codeDeps(Acc, ForwardRef(lambda: DebugDist), Memo, Rat)
class DebugAcc(Acc):
    def __init__(self, maxOcc, acc, tag = None):
        self.acc = acc
        self.tag = tag

        self.memo = Memo(maxOcc = maxOcc)

    def children(self):
        return [self.acc]

    @property
    def occ(self):
        return self.memo.occ

    def add(self, input, output, occ = 1.0):
        self.memo.add(input, output, occ)
        self.acc.add(input, output, occ)

    def addAccSingle(self, acc):
        self.memo.addAccSingle(acc.memo)

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        overallDistNew = DebugDist(self.memo.maxOcc, dist, tag = self.tag)
        return overallDistNew, (0.0, Rat.Exact)

@codeDeps(Acc, ForwardRef(lambda: AutoregressiveSequenceDist), Rat,
    contextualizeIter
)
class AutoregressiveSequenceAcc(Acc):
    def __init__(self, depth, seqFor, fillFrames, acc, tag = None):
        self.depth = depth
        self.seqFor = seqFor
        self.fillFrames = fillFrames
        self.acc = acc
        self.tag = tag

        assert len(self.fillFrames) <= self.depth

        self.occ = 0.0
        self.frames = 0.0

    def children(self):
        return [self.acc]

    def add(self, (uttId, input), outSeq, occ = 1.0):
        inSeq = self.seqFor(input)
        assert len(inSeq) == len(outSeq)
        self.occ += occ
        contextedOutSeq = contextualizeIter(self.depth, outSeq,
                                            fillFrames = self.fillFrames)
        for inFrame, (outContext, outFrame) in izip(inSeq, contextedOutSeq):
            self.frames += occ
            self.acc.add((inFrame, outContext), outFrame, occ)

    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.frames += acc.frames

    def count(self):
        return self.frames

    def logLikeSingle(self):
        return 0.0

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        dist = estimateChild(self.acc)
        overallDistNew = AutoregressiveSequenceDist(self.depth, self.seqFor,
                                                    self.fillFrames, dist,
                                                    tag = self.tag)
        return overallDistNew, (0.0, Rat.Exact)

@codeDeps(Acc, ForwardRef(lambda: AutoregressiveNetDist), Rat,
    wnet.forwardBackwardAlt
)
class AutoregressiveNetAcc(Acc):
    def __init__(self, distPrev, durAcc, acAcc, verbosity, tag = None):
        self.distPrev = distPrev
        self.durAcc = durAcc
        self.acAcc = acAcc
        self.verbosity = verbosity
        self.tag = tag

        self.occ = 0.0
        self.frames = 0.0
        self.entropy = 0.0

    def children(self):
        return [self.durAcc, self.acAcc]

    def add(self, (uttId, input), outSeq, occ = 1.0):
        if self.verbosity >= 2:
            print 'fb: uttId %s' % uttId
        self.occ += occ
        self.frames += len(outSeq) * occ

        timedNet, labelToWeight = self.distPrev.getTimedNet(input, outSeq)
        totalLogProb, edgeGen = wnet.forwardBackwardAlt(
            timedNet,
            labelToWeight = labelToWeight,
            divisionRing = self.distPrev.ring,
            getAgenda = self.distPrev.getAgenda
        )

        pruneSpec = self.distPrev.pruneSpec
        numFilled = len(self.distPrev.fillFrames)
        outSeqFilled = self.distPrev.fillFrames + outSeq
        entropy = totalLogProb * occ
        accedEdges = 0
        for (label, labelStartTime, labelEndTime), logOcc in edgeGen:
            if label is not None and (pruneSpec is None or
                                      pruneSpec.logOccThresh is None or
                                      logOcc > -pruneSpec.logOccThresh):
                labelOcc = math.exp(logOcc) * occ
                entropy -= labelToWeight((label, labelStartTime,
                                          labelEndTime)) * labelOcc

                acInput = outSeqFilled[
                    max(labelStartTime - self.distPrev.depth + numFilled, 0):
                    (labelStartTime + numFilled)
                ]
                if not label[0]:
                    _, phInput, phOutput = label
                    self.durAcc.add((phInput, acInput), phOutput, labelOcc)
                else:
                    _, phInput = label
                    assert labelEndTime == labelStartTime + 1
                    acOutput = outSeq[labelStartTime]
                    self.acAcc.add((phInput, acInput), acOutput, labelOcc)
                accedEdges += 1
        self.entropy += entropy

        if self.verbosity >= 2:
            print 'fb:    log like = %s (net path entropy = %s)' % (
                (0.0, 0.0) if len(outSeq) == 0
                else (totalLogProb / len(outSeq), entropy / len(outSeq))
            )
        if self.verbosity >= 3:
            print 'fb:    (accumulated over %s edges)' % accedEdges
        if self.verbosity >= 2:
            print 'fb:'

    # N.B. assumes distPrev is the same for self and acc (not checked).
    def addAccSingle(self, acc):
        self.occ += acc.occ
        self.frames += acc.frames
        self.entropy += acc.entropy

    def count(self):
        return self.frames

    def logLikeSingle(self):
        return self.entropy

    def derivParamsSingle(self):
        return []

    def estimateAux(self, estimateChild):
        durDist = estimateChild(self.durAcc)
        acDist = estimateChild(self.acAcc)
        if self.verbosity >= 1:
            print 'fb:    overall net path entropy = %s (%s frames)' % (
                (0.0, 0) if self.frames == 0
                else (self.entropy / self.frames, self.frames)
            )
        distNew = AutoregressiveNetDist(self.distPrev.depth,
                                        self.distPrev.netFor,
                                        self.distPrev.fillFrames,
                                        durDist, acDist,
                                        self.distPrev.pruneSpec,
                                        tag = self.tag)
        return distNew, (self.entropy, Rat.LowerBound)


@codeDeps(SynthMethod)
class Dist(object):
    """Conditional probability distribution."""
    def children(self):
        abstract
    def mapChildren(self, mapChild):
        abstract
    def logProb(self, input, output):
        abstract
    def logProbDerivInput(self, input, output):
        abstract
    def logProbDerivOutput(self, input, output):
        abstract
    def createAcc(self, createAccChild):
        abstract
    def createAccG(self, createAccChild):
        return self.createAcc(createAccChild)
    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        abstract
    def paramsSingle(self):
        abstract
    def paramsChildren(self, paramsChild):
        children = self.children()
        if children:
            return np.concatenate([ paramsChild(child) for child in children ])
        else:
            return []
    def parseSingle(self, params):
        abstract
    def parseChildren(self, params, parseChild):
        abstract
    def flooredSingle(self):
        return 0, 0
    def withTag(self, tag):
        """Set tag and return self.

        This is intended to be used immediately after object creation, such as:

            dist = SomeDist([2.0, 3.0, 4.0]).withTag('hi')

        This is particularly important here since a design goal is that Dists
        are immutable.
        """
        self.tag = tag
        return self

@codeDeps(Dist)
class TermDist(Dist):
    """Dist with no children."""
    def children(self):
        return []
    def createAccSingle(self):
        abstract
    def createAcc(self, createAccChild):
        return self.createAccSingle()
    def parseChildren(self, params, parseChild):
        return self, params

@codeDeps(FixedValueAcc, SynthMethod, TermDist)
class FixedValueDist(TermDist):
    def __init__(self, value, tag = None):
        self.value = value
        self.tag = tag

    def __repr__(self):
        return ('FixedValueDist(%r, tag=%r)' %
                (self.value, self.tag))

    def mapChildren(self, mapChild):
        return FixedValueDist(self.value, tag = self.tag)

    def logProb(self, input, output):
        if output == self.value:
            return 0.0
        else:
            return float('-inf')

    def logProbDerivInput(self, input, output):
        return np.zeros(np.shape(input))

    def createAccSingle(self):
        return FixedValueAcc(self.value, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.value

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return FixedValueDist(self.value, tag = self.tag), params

@codeDeps(OracleAcc, SynthMethod, TermDist)
class OracleDist(TermDist):
    def __init__(self, tag = None):
        self.tag = tag

    def __repr__(self):
        return 'OracleDist(tag=%r)' % self.tag

    def mapChildren(self, mapChild):
        return OracleDist(tag = self.tag)

    def logProb(self, input, output):
        return 0.0

    def createAccSingle(self):
        return OracleAcc(tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return actualOutput

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return OracleDist(tag = self.tag), params

@codeDeps(InvalidParamsError, LinearGaussianAcc, SynthMethod, TermDist)
class LinearGaussian(TermDist):
    def __init__(self, coeff, variance, varianceFloor, tag = None):
        self.coeff = coeff
        self.variance = variance
        self.varianceFloor = varianceFloor
        self.tag = tag
        self.gConst = -0.5 * math.log(2.0 * math.pi)

        assert self.varianceFloor is not None
        assert self.varianceFloor >= 0.0
        assert self.variance >= self.varianceFloor
        assert self.variance > 0.0
        if self.variance < 1e-10:
            raise RuntimeError('LinearGaussian variance too miniscule'
                               ' (variances this small can lead to substantial'
                               ' loss of precision during accumulation): %r' %
                               self.variance)

    def __repr__(self):
        return ('LinearGaussian(%r, %r, %r, tag=%r)' %
                (self.coeff, self.variance, self.varianceFloor, self.tag))

    def mapChildren(self, mapChild):
        return LinearGaussian(self.coeff, self.variance, self.varianceFloor,
                              tag = self.tag)

    def logProb(self, input, output):
        mean = np.dot(self.coeff, input)
        return (self.gConst - 0.5 * math.log(self.variance) +
                -0.5 * (output - mean) ** 2 / self.variance)

    def logProbDerivInput(self, input, output):
        mean = np.dot(self.coeff, input)
        return self.coeff * (output - mean) / self.variance

    def logProbDerivOutput(self, input, output):
        mean = np.dot(self.coeff, input)
        return -(output - mean) / self.variance

    def residual(self, input, output):
        mean = np.dot(self.coeff, input)
        return (output - mean) / math.sqrt(self.variance)

    def createAccSingle(self):
        return LinearGaussianAcc(distPrev = self, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        mean = np.dot(self.coeff, input)
        if method == SynthMethod.Meanish:
            return mean
        elif method == SynthMethod.Sample:
            return random.gauss(mean, math.sqrt(self.variance))
        else:
            raise RuntimeError('unknown SynthMethod %r' % method)

    def paramsSingle(self):
        return np.append(self.coeff, -math.log(self.variance))

    def parseSingle(self, params):
        n = len(self.coeff)
        coeff = params[:n]
        variance = math.exp(-params[n])
        if variance < self.varianceFloor:
            raise InvalidParamsError('variance = %r < varianceFloor = %r'
                                     ' during LinearGaussian parsing' %
                                     (variance, self.varianceFloor))
        distNew = LinearGaussian(coeff, variance, self.varianceFloor,
                                 tag = self.tag)
        return distNew, params[n + 1:]

    def flooredSingle(self):
        return ((1, 1) if np.allclose(self.variance, self.varianceFloor)
                else (0, 1))

@codeDeps(InvalidParamsError, LinearGaussianVecAcc, SynthMethod, TermDist,
    reprArray
)
class LinearGaussianVec(TermDist):
    def __init__(self, coeffVec, varianceVec, varianceFloorVec, tag = None):
        self.coeffVec = coeffVec
        self.varianceVec = varianceVec
        self.varianceFloorVec = varianceFloorVec
        self.tag = tag

        order, inputLength = np.shape(self.coeffVec)
        assert np.shape(varianceVec) == (order,)
        if self.varianceFloorVec is not None:
            assert np.shape(varianceFloorVec) == (order,)

        assert self.varianceFloorVec is not None
        assert np.all(self.varianceFloorVec >= 0.0)
        assert np.all(self.varianceVec >= self.varianceFloorVec)
        assert np.all(self.varianceVec > 0.0)
        if np.any(self.varianceVec < 1e-10):
            raise RuntimeError('LinearGaussian variance too miniscule'
                               ' (variances this small can lead to substantial'
                               ' loss of precision during accumulation): %r' %
                               self.varianceVec)

    def __repr__(self):
        return ('LinearGaussianVec(%s, %s, %s, tag=%r)' %
                (reprArray(self.coeffVec), reprArray(self.varianceVec),
                 reprArray(self.varianceFloorVec), self.tag))

    def mapChildren(self, mapChild):
        return LinearGaussianVec(self.coeffVec, self.varianceVec,
                                 self.varianceFloorVec, tag = self.tag)

    def logProb(self, input, output):
        meanVec = np.sum(self.coeffVec * input.T, axis = 1)
        lps = (
            -0.5 * np.log(self.varianceVec) +
            -0.5 * (output - meanVec) ** 2 / self.varianceVec +
            -0.5 * math.log(2.0 * math.pi)
        )
        return np.sum(lps)

    def logProbDerivInput(self, input, output):
        meanVec = np.sum(self.coeffVec * input.T, axis = 1)
        return self.coeffVec.T * (output - meanVec) / self.varianceVec

    def logProbDerivOutput(self, input, output):
        meanVec = np.sum(self.coeffVec * input.T, axis = 1)
        return -(output - meanVec) / self.varianceVec

    def residualVec(self, input, output):
        meanVec = np.sum(self.coeffVec * input.T, axis = 1)
        return (output - meanVec) / math.sqrt(self.varianceVec)

    def createAccSingle(self):
        return LinearGaussianVecAcc(distPrev = self, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        meanVec = np.sum(self.coeffVec * input.T, axis = 1)
        if method == SynthMethod.Meanish:
            return meanVec
        elif method == SynthMethod.Sample:
            return np.array([
                random.gauss(mean, math.sqrt(variance))
                for mean, variance in zip(meanVec, self.varianceVec)
            ])
        else:
            raise RuntimeError('unknown SynthMethod %r' % method)

    def paramsSingle(self):
        return np.append(self.coeffVec, -np.log(self.varianceVec))

    def parseSingle(self, params):
        order, inputLength = np.shape(self.coeffVec)
        n = order * inputLength
        coeffVec = np.reshape(params[:n], np.shape(self.coeffVec))
        varianceVec = np.exp(-params[n:(n + order)])
        if np.any(varianceVec < self.varianceFloorVec):
            raise InvalidParamsError('variance = %r < varianceFloor = %r'
                                     ' during LinearGaussian parsing' %
                                     (varianceVec, self.varianceFloorVec))
        distNew = LinearGaussianVec(coeffVec, varianceVec,
                                    self.varianceFloorVec, tag = self.tag)
        return distNew, params[(n + order):]

    def flooredSingle(self):
        isClose = [ np.allclose(variance, varianceFloor)
                    for variance, varianceFloor in zip(self.varianceVec,
                                                       self.varianceFloorVec) ]
        return (np.sum(isClose), np.size(isClose))

@codeDeps(DerivTermAccG, SynthMethod, TermDist)
class StudentDist(TermDist):
    def __init__(self, df, precision, tag = None):
        if df <= 0.0:
            raise ValueError('df = %r but should be > 0.0' % df)
        if precision <= 0.0:
            raise ValueError('precision = %r but should be > 0.0' % precision)
        self.df = df
        self.precision = precision
        self.tag = tag

        self.gConst = (special.gammaln(0.5) +
                       -special.betaln(0.5, 0.5 * self.df) +
                       0.5 * math.log(self.precision) +
                       -0.5 * math.log(self.df) +
                       -0.5 * math.log(math.pi))

    def __repr__(self):
        return ('StudentDist(%r, %r, tag=%r)' %
                (self.df, self.precision, self.tag))

    def mapChildren(self, mapChild):
        return StudentDist(self.df, self.precision, tag = self.tag)

    def logProb(self, input, output):
        assert np.shape(output) == ()
        a = output * output * self.precision / self.df
        return self.gConst - 0.5 * (self.df + 1.0) * math.log(1.0 + a)

    def logProbDerivInput(self, input, output):
        assert np.shape(output) == ()
        return np.zeros(np.shape(input))

    def logProbDerivOutput(self, input, output):
        assert np.shape(output) == ()
        a = output * output * self.precision / self.df
        return -(self.df + 1.0) * output * self.precision / self.df / (1.0 + a)

    def logProbDerivParams(self, input, output):
        a = output * output * self.precision / self.df
        K = self.df - (1.0 + self.df) / (1.0 + a)
        return np.array([
            0.5 * K + 0.5 * self.df * (special.psi(0.5 * (self.df + 1.0)) +
                                       -special.psi(0.5 * self.df) +
                                       -math.log(1.0 + a)),
            -0.5 * K
        ])

    def createAcc(self, createAccChild):
        raise RuntimeError('cannot estimate Student distribution using EM')

    def createAccG(self, createAccChild):
        return DerivTermAccG(distPrev = self, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        if method == SynthMethod.Meanish:
            return 0.0
        elif method == SynthMethod.Sample:
            while True:
                out = np.random.standard_t(self.df) / math.sqrt(self.precision)
                if math.isinf(out):
                    print 'NOTE: redoing sample from t-dist since it was', out
                else:
                    return out
        else:
            raise RuntimeError('unknown SynthMethod %r' % method)

    def paramsSingle(self):
        return np.array([math.log(self.df), math.log(self.precision)])

    def parseSingle(self, params):
        df = math.exp(params[0])
        precision = math.exp(params[1])
        return StudentDist(df, precision, tag = self.tag), params[2:]

@codeDeps(ConstantClassifierAcc, InvalidParamsError, SynthMethod, TermDist,
    assert_allclose, sampleDiscrete
)
class ConstantClassifier(TermDist):
    def __init__(self, probs, probFloors, tag = None):
        self.probs = probs
        self.probFloors = probFloors
        self.tag = tag

        assert len(self.probs) >= 1
        assert_allclose(sum(self.probs), 1.0)
        assert self.probFloors is not None
        assert len(self.probFloors) == len(self.probs)
        assert all(self.probFloors >= 0.0)
        assert sum(self.probFloors) <= 1.0
        assert all(self.probs >= self.probFloors)

    def __repr__(self):
        return ('ConstantClassifier(%r, %r, tag=%r)' %
                (self.probs, self.probFloors, self.tag))

    def mapChildren(self, mapChild):
        return ConstantClassifier(self.probs, self.probFloors, tag = self.tag)

    def logProb(self, input, classIndex):
        prob = self.probs[classIndex]
        return math.log(prob) if prob != 0.0 else float('-inf')

    def logProbDerivInput(self, input, classIndex):
        return np.zeros(np.shape(input))

    def createAccSingle(self):
        return ConstantClassifierAcc(distPrev = self, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        if method == SynthMethod.Meanish:
            prob, classIndex = max([
                (prob, classIndex)
                for classIndex, prob in enumerate(self.probs)
            ])
            return classIndex
        elif method == SynthMethod.Sample:
            return sampleDiscrete(list(enumerate(self.probs)))
        else:
            raise RuntimeError('unknown SynthMethod %r' % method)

    def paramsSingle(self):
        logProbs = np.log(self.probs)
        if not np.all(np.isfinite(logProbs)):
            raise RuntimeError('this parameterization of ConstantClassifier'
                               ' cannot cope with zero (or NaN) probabilities'
                               ' (probs = %r)' % self.probs)
        sumZeroLogProbs = logProbs - np.mean(logProbs)
        return sumZeroLogProbs[:-1]

    def parseSingle(self, params):
        n = len(self.probs) - 1
        p = params[:n]
        if not np.all(np.isfinite(p)):
            raise InvalidParamsError('params %r not all finite during'
                                     ' ConstantClassifier parsing' % p)
        sumZeroLogProbs = np.append(p, -sum(p))
        assert_allclose(sum(sumZeroLogProbs), 0.0)
        probs = np.exp(sumZeroLogProbs)
        probs = probs / sum(probs)
        if not all(probs >= self.probFloors):
            raise InvalidParamsError('probs = %r not all >= probFloors = %r'
                                     ' during ConstantClassifier parsing' %
                                     (probs, self.probFloors))
        distNew = ConstantClassifier(probs, self.probFloors, tag = self.tag)
        return distNew, params[n:]

    def flooredSingle(self):
        numFloored = sum([
            (1 if np.allclose(prob, probFloor) else 0)
            for prob, probFloor in zip(self.probs, self.probFloors)
        ])
        if np.allclose(sum(self.probFloors), 1.0):
            assert numFloored == len(self.probs)
            return numFloored, len(self.probs)
        else:
            return numFloored, len(self.probs) - 1

@codeDeps(BinaryLogisticClassifierAcc, InvalidParamsError, SynthMethod,
    TermDist, sigmoid
)
class BinaryLogisticClassifier(TermDist):
    def __init__(self, coeff, coeffFloor, tag = None):
        self.coeff = coeff
        self.coeffFloor = coeffFloor
        self.tag = tag

        assert len(self.coeffFloor) == len(self.coeff)
        assert all(self.coeffFloor >= 0.0)
        assert all(np.abs(self.coeff) <= self.coeffFloor)

    def __repr__(self):
        return ('BinaryLogisticClassifier(%r, %r, tag=%r)' %
                (self.coeff, self.coeffFloor, self.tag))

    def mapChildren(self, mapChild):
        return BinaryLogisticClassifier(self.coeff, self.coeffFloor,
                                        tag = self.tag)

    def logProb(self, input, classIndex):
        prob = self.prob(input, classIndex)
        return math.log(prob) if prob != 0.0 else float('-inf')

    def prob(self, input, classIndex):
        prob1 = sigmoid(np.dot(self.coeff, input))
        if classIndex == 0:
            return 1.0 - prob1
        else:
            return prob1

    def logProbDerivInput(self, input, classIndex):
        return self.coeff * (classIndex - sigmoid(np.dot(self.coeff, input)))

    def createAccSingle(self):
        return BinaryLogisticClassifierAcc(self, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        prob1 = sigmoid(np.dot(self.coeff, input))
        if method == SynthMethod.Meanish:
            if prob1 > 0.5:
                return 1
            else:
                return 0
        elif method == SynthMethod.Sample:
            if random.random() < prob1:
                return 1
            else:
                return 0
        else:
            raise RuntimeError('unknown SynthMethod %r' % method)

    def paramsSingle(self):
        return self.coeff

    def parseSingle(self, params):
        n = len(self.coeff)
        coeff = params[:n]
        if not all(np.abs(coeff) <= self.coeffFloor):
            raise InvalidParamsError('abs(coeff = %r) not all <= coeffFloor'
                                     ' = %r during BinaryLogisticClassifier'
                                     ' parsing' % (coeff, self.coeffFloor))
        distNew = BinaryLogisticClassifier(coeff, self.coeffFloor,
                                           tag = self.tag)
        return distNew, params[n:]

    def flooredSingle(self):
        numFloored = sum([
            (1 if np.allclose(abs(coeffValue), coeffFloorValue) else 0)
            for coeffValue, coeffFloorValue in zip(self.coeff, self.coeffFloor)
        ])
        return numFloored, len(self.coeff)

@codeDeps(Dist, MixtureAcc, SynthMethod, logSum, parseConcat)
class MixtureDist(Dist):
    def __init__(self, classDist, regDists, tag = None):
        self.numComps = len(regDists)
        self.classDist = classDist
        self.regDists = regDists
        self.tag = tag

    def __repr__(self):
        return ('MixtureDist(%r, %r, tag=%r)' %
                (self.classDist, self.regDists, self.tag))

    def children(self):
        return [self.classDist] + self.regDists

    def mapChildren(self, mapChild):
        classDist = mapChild(self.classDist)
        regDists = [ mapChild(regDist) for regDist in self.regDists ]
        return MixtureDist(classDist, regDists, tag = self.tag)

    def logProb(self, input, output):
        return logSum([ self.logProbComp(input, comp, output)
                        for comp in range(self.numComps) ])

    def logProbComp(self, input, comp, output):
        return (self.classDist.logProb(input, comp) +
                self.regDists[comp].logProb(input, output))

    def logProbDerivInput(self, input, output):
        logTot = self.logProb(input, output)
        return np.sum([
            (regDist.logProbDerivInput(input, output) +
             self.classDist.logProbDerivInput(input, comp)) *
            math.exp(self.logProbComp(input, comp, output) - logTot)
            for comp, regDist in enumerate(self.regDists)
        ], axis = 0)

    def logProbDerivOutput(self, input, output):
        logTot = self.logProb(input, output)
        return np.sum([
            regDist.logProbDerivOutput(input, output) *
            math.exp(self.logProbComp(input, comp, output) - logTot)
            for comp, regDist in enumerate(self.regDists)
        ], axis = 0)

    def createAcc(self, createAccChild):
        classAcc = createAccChild(self.classDist)
        regAccs = [ createAccChild(regDist) for regDist in self.regDists ]
        return MixtureAcc(self, classAcc, regAccs, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        if method == SynthMethod.Meanish:
            return np.sum([
                regDist.synth(input, SynthMethod.Meanish, actualOutput) *
                math.exp(self.classDist.logProb(input, comp))
                for comp, regDist in enumerate(self.regDists)
            ], axis = 0)
        elif method == SynthMethod.Sample:
            comp = self.classDist.synth(input, method)
            output = self.regDists[comp].synth(input, method, actualOutput)
            return output
        else:
            raise RuntimeError('unknown SynthMethod %r' % method)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dists, paramsLeft = parseConcat(self.children(), params, parseChild)
        return MixtureDist(dists[0], dists[1:], tag = self.tag), paramsLeft

@codeDeps(Dist, IdentifiableMixtureAcc, SynthMethod, parseConcat)
class IdentifiableMixtureDist(Dist):
    def __init__(self, classDist, regDists, tag = None):
        self.classDist = classDist
        self.regDists = regDists
        self.tag = tag

    def __repr__(self):
        return ('IdentifiableMixtureDist(%r, %r, tag=%r)' %
                (self.classDist, self.regDists, self.tag))

    def children(self):
        return [self.classDist] + self.regDists

    def mapChildren(self, mapChild):
        classDist = mapChild(self.classDist)
        regDists = [ mapChild(regDist) for regDist in self.regDists ]
        return IdentifiableMixtureDist(classDist, regDists, tag = self.tag)

    def logProb(self, input, output):
        comp, acOutput = output
        return (self.classDist.logProb(input, comp) +
                self.regDists[comp].logProb(input, acOutput))

    def logProbDerivInput(self, input, output):
        comp, acOutput = output
        return (self.regDists[comp].logProbDerivInput(input, acOutput) +
                self.classDist.logProbDerivInput(input, comp))

    def logProbDerivOutput(self, input, output):
        comp, acOutput = output
        return self.regDists[comp].logProbDerivOutput(input, acOutput)

    def createAcc(self, createAccChild):
        classAcc = createAccChild(self.classDist)
        regAccs = [ createAccChild(regDist) for regDist in self.regDists ]
        return IdentifiableMixtureAcc(classAcc, regAccs, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        actualComp, actualAcOutput = (actualOutput if actualOutput is not None
                                      else (None, None))
        comp = self.classDist.synth(input, method, actualComp)
        acOutput = self.regDists[comp].synth(input, method, actualAcOutput)
        return comp, acOutput

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dists, paramsLeft = parseConcat(self.children(), params, parseChild)
        distNew = IdentifiableMixtureDist(dists[0], dists[1:], tag = self.tag)
        return distNew, paramsLeft

@codeDeps(ForwardRef(lambda: VectorDist))
def createVectorDist(order, outIndices, vectorSummarizer, createDistForIndex):
    distComps = dict()
    for outIndex in outIndices:
        distComps[outIndex] = createDistForIndex(outIndex)
    return VectorDist(order, vectorSummarizer, outIndices, distComps)

@codeDeps(Dist, SynthMethod, VectorAcc, orderedDictRepr, parseConcat)
class VectorDist(Dist):
    def __init__(self, order, vectorSummarizer, keys, distComps, tag = None):
        assert len(keys) == len(distComps)
        for key in keys:
            assert key in distComps
        self.order = order
        self.vectorSummarizer = vectorSummarizer
        self.keys = keys
        self.distComps = distComps
        self.tag = tag

    def __repr__(self):
        return ('VectorDist(%r, %r, %r, %s, tag=%r)' %
                (self.order, self.vectorSummarizer, self.keys,
                 orderedDictRepr(self.keys, self.distComps),
                 self.tag))

    def children(self):
        return [ self.distComps[key] for key in self.keys ]

    def mapChildren(self, mapChild):
        distComps = dict()
        for outIndex in self.distComps:
            distComps[outIndex] = mapChild(self.distComps[outIndex])
        return VectorDist(self.order, self.vectorSummarizer, self.keys,
                          distComps, tag = self.tag)

    def logProb(self, input, output):
        lp = 0.0
        for outIndex in self.distComps:
            summary = self.vectorSummarizer(input, output[:outIndex], outIndex)
            lp += self.distComps[outIndex].logProb(summary, output[outIndex])
        return lp

    def logProbDerivInput(self, input, output):
        # FIXME : complete
        notyetimplemented

    def logProbDerivOutput(self, input, output):
        # FIXME : complete
        notyetimplemented

    def createAcc(self, createAccChild):
        accComps = dict()
        for outIndex in self.distComps:
            accComps[outIndex] = createAccChild(self.distComps[outIndex])
        return VectorAcc(self.order, self.vectorSummarizer, self.keys,
                         accComps, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        partialOutput = []
        for outIndex in range(self.order):
            if not outIndex in self.distComps:
                out = actualOutput[outIndex]
            else:
                summary = self.vectorSummarizer(input, partialOutput, outIndex)
                out = self.distComps[outIndex].synth(
                    summary,
                    method,
                    None if actualOutput is None else actualOutput[outIndex]
                )
            partialOutput.append(out)
        return partialOutput

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dists, paramsLeft = parseConcat(self.children(), params, parseChild)
        distNew = VectorDist(self.order, self.vectorSummarizer, self.keys,
                             dict(zip(self.keys, dists)), tag = self.tag)
        return distNew, paramsLeft

@codeDeps(ForwardRef(lambda: DiscreteDist))
def createDiscreteDist(keys, createDistFor):
    distDict = dict()
    for key in keys:
        distDict[key] = createDistFor(key)
    return DiscreteDist(keys, distDict)

@codeDeps(DiscreteAcc, Dist, SynthMethod, orderedDictRepr, parseConcat)
class DiscreteDist(Dist):
    def __init__(self, keys, distDict, tag = None):
        assert len(keys) == len(distDict)
        for key in keys:
            assert key in distDict
        self.keys = keys
        self.distDict = distDict
        self.tag = tag

    def __repr__(self):
        return ('DiscreteDist(%r, %s, tag=%r)' %
                (self.keys, orderedDictRepr(self.keys, self.distDict),
                 self.tag))

    def children(self):
        return [ self.distDict[key] for key in self.keys ]

    def mapChildren(self, mapChild):
        distDict = dict()
        for label in self.distDict:
            distDict[label] = mapChild(self.distDict[label])
        return DiscreteDist(self.keys, distDict, tag = self.tag)

    def logProb(self, input, output):
        label, acInput = input
        return self.distDict[label].logProb(acInput, output)

    def logProbDerivInput(self, input, output):
        label, acInput = input
        return self.distDict[label].logProbDerivInput(acInput, output)

    def logProbDerivOutput(self, input, output):
        label, acInput = input
        return self.distDict[label].logProbDerivOutput(acInput, output)

    def createAcc(self, createAccChild):
        accDict = dict()
        for label in self.distDict:
            accDict[label] = createAccChild(self.distDict[label])
        return DiscreteAcc(self.keys, accDict, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        label, acInput = input
        return self.distDict[label].synth(acInput, method, actualOutput)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dists, paramsLeft = parseConcat(self.children(), params, parseChild)
        distNew = DiscreteDist(self.keys, dict(zip(self.keys, dists)),
                               tag = self.tag)
        return distNew, paramsLeft

@codeDeps(Dist)
class DecisionTree(Dist):
    pass

@codeDeps(DecisionTree, DecisionTreeAccNode, SynthMethod)
class DecisionTreeNode(DecisionTree):
    def __init__(self, fullQuestion, distYes, distNo, tag = None):
        self.fullQuestion = fullQuestion
        self.distYes = distYes
        self.distNo = distNo
        self.tag = tag

    def __repr__(self):
        return ('DecisionTreeNode(%r, %r, %r, tag=%r)' %
                (self.fullQuestion, self.distYes, self.distNo, self.tag))

    def children(self):
        return [self.distYes, self.distNo]

    def mapChildren(self, mapChild):
        return DecisionTreeNode(self.fullQuestion, mapChild(self.distYes),
                                mapChild(self.distNo), tag = self.tag)

    def logProb(self, input, output):
        label, acInput = input
        labelValuer, question = self.fullQuestion
        if question(labelValuer(label)):
            return self.distYes.logProb(input, output)
        else:
            return self.distNo.logProb(input, output)

    def logProbDerivInput(self, input, output):
        label, acInput = input
        labelValuer, question = self.fullQuestion
        if question(labelValuer(label)):
            return self.distYes.logProbDerivInput(input, output)
        else:
            return self.distNo.logProbDerivInput(input, output)

    def logProbDerivOutput(self, input, output):
        label, acInput = input
        labelValuer, question = self.fullQuestion
        if question(labelValuer(label)):
            return self.distYes.logProbDerivOutput(input, output)
        else:
            return self.distNo.logProbDerivOutput(input, output)

    def createAcc(self, createAccChild):
        return DecisionTreeAccNode(self.fullQuestion,
                                   createAccChild(self.distYes),
                                   createAccChild(self.distNo),
                                   tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        label, acInput = input
        labelValuer, question = self.fullQuestion
        if question(labelValuer(label)):
            return self.distYes.synth(input, method, actualOutput)
        else:
            return self.distNo.synth(input, method, actualOutput)

    def countLeaves(self):
        return self.distYes.countLeaves() + self.distNo.countLeaves()

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        paramsLeft = params
        distYes, paramsLeft = parseChild(self.distYes, paramsLeft)
        distNo, paramsLeft = parseChild(self.distNo, paramsLeft)
        distNew = DecisionTreeNode(self.fullQuestion, distYes, distNo,
                                   tag = self.tag)
        return distNew, paramsLeft

@codeDeps(DecisionTree, DecisionTreeAccLeaf, SynthMethod)
class DecisionTreeLeaf(DecisionTree):
    def __init__(self, dist, tag = None):
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return ('DecisionTreeLeaf(%r, tag=%r)' %
                (self.dist, self.tag))

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        return DecisionTreeLeaf(mapChild(self.dist), tag = self.tag)

    def logProb(self, input, output):
        label, acInput = input
        return self.dist.logProb(acInput, output)

    def logProbDerivInput(self, input, output):
        label, acInput = input
        return self.dist.logProbDerivInput(acInput, output)

    def logProbDerivOutput(self, input, output):
        label, acInput = input
        return self.dist.logProbDerivOutput(acInput, output)

    def createAcc(self, createAccChild):
        return DecisionTreeAccLeaf(createAccChild(self.dist), tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        label, acInput = input
        return self.dist.synth(acInput, method, actualOutput)

    def countLeaves(self):
        return 1

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        return DecisionTreeLeaf(dist, tag = self.tag), paramsLeft

# (FIXME : merge MappedInputDist with TransformedInputDist?
#   (Also merge some of the corresponding Accs?))
@codeDeps(Dist, MappedInputAcc, SynthMethod)
class MappedInputDist(Dist):
    """Dist where input is mapped using a fixed transform."""
    def __init__(self, inputTransform, dist, tag = None):
        self.inputTransform = inputTransform
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return ('MappedInputDist(%r, %r, tag=%r)' %
                (self.inputTransform, self.dist, self.tag))

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        dist = mapChild(self.dist)
        return MappedInputDist(self.inputTransform, dist, tag = self.tag)

    def logProb(self, input, output):
        return self.dist.logProb(self.inputTransform(input), output)

    def logProbDerivInput(self, input, output):
        return np.dot(
            self.inputTransform.deriv(input),
            self.dist.logProbDerivInput(self.inputTransform(input), output)
        )

    def logProbDerivOutput(self, input, output):
        return self.dist.logProbDerivOutput(self.inputTransform(input), output)

    def createAcc(self, createAccChild):
        acc = createAccChild(self.dist)
        return MappedInputAcc(self.inputTransform, acc, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.dist.synth(self.inputTransform(input), method,
                               actualOutput)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        overallDistNew = MappedInputDist(self.inputTransform, dist,
                                         tag = self.tag)
        return overallDistNew, paramsLeft

# (FIXME : merge MappedOutputDist with TransformedOutputDist?
#   (Also merge some of the corresponding Accs?))
@codeDeps(Dist, MappedOutputAcc, SynthMethod)
class MappedOutputDist(Dist):
    """Dist where output is mapped using a fixed transform."""
    def __init__(self, outputTransform, dist, tag = None):
        self.outputTransform = outputTransform
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return ('MappedOutputDist(%r, %r, tag=%r)' %
                (self.outputTransform, self.dist, self.tag))

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        dist = mapChild(self.dist)
        return MappedOutputDist(self.outputTransform, dist, tag = self.tag)

    def logProb(self, input, output):
        return (self.dist.logProb(input, self.outputTransform(input, output)) +
                self.outputTransform.logJac(input, output))

    def logProbDerivInput(self, input, output):
        outputT = self.outputTransform(input, output)
        return (np.dot(self.outputTransform.derivInput(input, output),
                       self.dist.logProbDerivOutput(input, outputT)) +
                self.dist.logProbDerivInput(input, outputT) +
                self.outputTransform.logJacDerivInput(input, output))

    def logProbDerivOutput(self, input, output):
        return np.dot(
            self.outputTransform.deriv(input, output),
            self.dist.logProbDerivOutput(input,
                                         self.outputTransform(input, output))
        ) + self.outputTransform.logJacDeriv(input, output)

    def createAcc(self, createAccChild):
        acc = createAccChild(self.dist)
        return MappedOutputAcc(self.outputTransform, acc, tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.outputTransform.inv(input,
            self.dist.synth(
                input,
                method,
                (None if actualOutput is None
                 else self.outputTransform(input, actualOutput))
            )
        )

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        overallDistNew = MappedOutputDist(self.outputTransform, dist,
                                          tag = self.tag)
        return overallDistNew, paramsLeft

@codeDeps(Dist, SynthMethod, TransformedInputAccG,
    TransformedInputLearnDistAccEM, TransformedInputLearnTransformAccEM
)
class TransformedInputDist(Dist):
    """Dist where input is transformed using a learnable transform."""
    def __init__(self, inputTransform, dist, tag = None):
        self.inputTransform = inputTransform
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return ('TransformedInputDist(%r, %r, tag=%r)' %
                (self.inputTransform, self.dist, self.tag))

    def children(self):
        return [self.inputTransform, self.dist]

    def mapChildren(self, mapChild):
        inputTransform = mapChild(self.inputTransform)
        dist = mapChild(self.dist)
        return TransformedInputDist(inputTransform, dist, tag = self.tag)

    def logProb(self, input, output):
        return self.dist.logProb(self.inputTransform(input), output)

    def logProbDerivInput(self, input, output):
        return np.dot(
            self.inputTransform.deriv(input),
            self.dist.logProbDerivInput(self.inputTransform(input), output)
        )

    def logProbDerivOutput(self, input, output):
        return self.dist.logProbDerivOutput(self.inputTransform(input), output)

    def createAcc(self, createAccChild, estTransform = False):
        if estTransform:
            inputTransformAcc = createAccChild(self.inputTransform)
            return TransformedInputLearnTransformAccEM(inputTransformAcc,
                                                       self.dist,
                                                       tag = self.tag)
        else:
            acc = createAccChild(self.dist)
            return TransformedInputLearnDistAccEM(self.inputTransform, acc,
                                                  tag = self.tag)

    def createAccG(self, createAccChild):
        inputTransformAcc = createAccChild(self.inputTransform)
        acc = createAccChild(self.dist)
        return TransformedInputAccG((inputTransformAcc, self.inputTransform),
                                    (acc, self.dist), tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.dist.synth(self.inputTransform(input), method,
                               actualOutput)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        inputTransform, paramsLeft = parseChild(self.inputTransform, params)
        dist, paramsLeft = parseChild(self.dist, paramsLeft)
        overallDistNew = TransformedInputDist(inputTransform, dist,
                                              tag = self.tag)
        return overallDistNew, paramsLeft

@codeDeps(Dist, SynthMethod, TransformedOutputAccG,
    TransformedOutputLearnDistAccEM, TransformedOutputLearnTransformAccEM
)
class TransformedOutputDist(Dist):
    """Dist where output is transformed using a learnable transform."""
    def __init__(self, outputTransform, dist, tag = None):
        self.outputTransform = outputTransform
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return ('TransformedOutputDist(%r, %r, tag=%r)' %
                (self.outputTransform, self.dist, self.tag))

    def children(self):
        return [self.outputTransform, self.dist]

    def mapChildren(self, mapChild):
        outputTransform = mapChild(self.outputTransform)
        dist = mapChild(self.dist)
        return TransformedOutputDist(outputTransform, dist, tag = self.tag)

    def logProb(self, input, output):
        return (
            self.dist.logProb(input, self.outputTransform(input, output)) +
            self.outputTransform.logJac(input, output)
        )

    def logProbDerivInput(self, input, output):
        outputT = self.outputTransform(input, output)
        return (np.dot(self.outputTransform.derivInput(input, output),
                       self.dist.logProbDerivOutput(input, outputT)) +
                self.dist.logProbDerivInput(input, outputT) +
                self.outputTransform.logJacDerivInput(input, output))

    def logProbDerivOutput(self, input, output):
        return np.dot(
            self.outputTransform.deriv(input, output),
            self.dist.logProbDerivOutput(input,
                                         self.outputTransform(input, output))
        ) + self.outputTransform.logJacDeriv(input, output)

    def createAcc(self, createAccChild, estTransform = False):
        if estTransform:
            outputTransformAcc = createAccChild(self.outputTransform)
            return TransformedOutputLearnTransformAccEM(outputTransformAcc,
                                                        self.dist,
                                                        tag = self.tag)
        else:
            acc = createAccChild(self.dist)
            return TransformedOutputLearnDistAccEM(self.outputTransform, acc,
                                                   tag = self.tag)

    def createAccG(self, createAccChild):
        outputTransformAcc = createAccChild(self.outputTransform)
        acc = createAccChild(self.dist)
        return TransformedOutputAccG(
            (outputTransformAcc, self.outputTransform),
            (acc, self.dist),
            tag = self.tag
        )

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.outputTransform.inv(input,
            self.dist.synth(
                input,
                method,
                (None if actualOutput is None
                 else self.outputTransform(input, actualOutput))
            )
        )

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        outputTransform, paramsLeft = parseChild(self.outputTransform, params)
        dist, paramsLeft = parseChild(self.dist, paramsLeft)
        overallDistNew = TransformedOutputDist(outputTransform, dist,
                                               tag = self.tag)
        return overallDistNew, paramsLeft

@codeDeps(Dist, PassThruAcc, SynthMethod)
class PassThruDist(Dist):
    def __init__(self, dist, tag = None):
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return 'PassThruDist(%r, tag=%r)' % (self.dist, self.tag)

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        return PassThruDist(mapChild(self.dist), tag = self.tag)

    def logProb(self, input, output):
        return self.dist.logProb(input, output)

    def logProbDerivInput(self, input, output):
        return self.dist.logProbDerivInput(input, output)

    def logProbDerivOutput(self, input, output):
        return self.dist.logProbDerivOutput(input, output)

    def createAcc(self, createAccChild):
        return PassThruAcc(createAccChild(self.dist), tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.dist.synth(input, method, actualOutput)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        return PassThruDist(dist, tag = self.tag), paramsLeft

@codeDeps(CountFramesAcc, Dist, SynthMethod)
class CountFramesDist(Dist):
    def __init__(self, dist, tag = None):
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return 'CountFramesDist(%r, tag=%r)' % (self.dist, self.tag)

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        return CountFramesDist(mapChild(self.dist), tag = self.tag)

    def logProb(self, input, outSeq):
        return self.dist.logProb(input, outSeq)

    def logProbDerivInput(self, input, outSeq):
        return self.dist.logProbDerivInput(input, outSeq)

    def logProbDerivOutput(self, input, outSeq):
        return self.dist.logProbDerivOutput(input, outSeq)

    def createAcc(self, createAccChild):
        return CountFramesAcc(createAccChild(self.dist), tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.dist.synth(input, method, actualOutput)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        return CountFramesDist(dist, tag = self.tag), paramsLeft

@codeDeps(DebugAcc, Dist, SynthMethod)
class DebugDist(Dist):
    def __init__(self, maxOcc, dist, tag = None):
        self.maxOcc = maxOcc
        self.dist = dist
        self.tag = tag

    def __repr__(self):
        return 'DebugDist(%r, %r, tag=%r)' % (self.maxOcc, self.dist, self.tag)

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        return DebugDist(self.maxOcc, mapChild(self.dist), tag = self.tag)

    def logProb(self, input, output):
        return self.dist.logProb(input, output)

    def logProbDerivInput(self, input, output):
        return self.dist.logProbDerivInput(input, output)

    def logProbDerivOutput(self, input, output):
        return self.dist.logProbDerivOutput(input, output)

    def createAcc(self, createAccChild):
        return DebugAcc(self.maxOcc, createAccChild(self.dist), tag = self.tag)

    def synth(self, input, method = SynthMethod.Sample, actualOutput = None):
        return self.dist.synth(input, method, actualOutput)

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        return DebugDist(self.maxOcc, dist, tag = self.tag), paramsLeft

@codeDeps(AutoregressiveSequenceAcc, Dist, SynthMethod, contextualizeIter)
class AutoregressiveSequenceDist(Dist):
    def __init__(self, depth, seqFor, fillFrames, dist, tag = None):
        self.depth = depth
        self.seqFor = seqFor
        self.fillFrames = fillFrames
        self.dist = dist
        self.tag = tag

        assert len(self.fillFrames) <= self.depth

    def __repr__(self):
        return ('AutoregressiveSequenceDist(%r, %r, %r, %r, tag=%r)' %
                (self.depth, self.seqFor, self.fillFrames, self.dist,
                 self.tag))

    def children(self):
        return [self.dist]

    def mapChildren(self, mapChild):
        return AutoregressiveSequenceDist(self.depth, self.seqFor,
                                          self.fillFrames, mapChild(self.dist),
                                          tag = self.tag)

    def logProb(self, (uttId, input), outSeq):
        inSeq = self.seqFor(input)
        lp = 0.0
        assert len(inSeq) == len(outSeq)
        contextedOutSeq = contextualizeIter(self.depth, outSeq,
                                            fillFrames = self.fillFrames)
        for inFrame, (outContext, outFrame) in izip(inSeq, contextedOutSeq):
            lp += self.dist.logProb((inFrame, outContext), outFrame)
        return lp

    def logProbDerivInput(self, (uttId, input), outSeq):
        # FIXME : complete
        notyetimplemented

    def logProbDerivOutput(self, (uttId, input), outSeq):
        # FIXME : complete
        notyetimplemented

    def sum(self, (uttId, input), outSeq, computeValue):
        inSeq = self.seqFor(input)
        assert len(inSeq) == len(outSeq)
        contextedOutSeq = contextualizeIter(self.depth, outSeq,
                                            fillFrames = self.fillFrames)
        return sum([
            computeValue(self.dist, (inFrame, outContext), outFrame)
            for inFrame, (outContext, outFrame) in izip(inSeq, contextedOutSeq)
        ])

    def createAcc(self, createAccChild):
        return AutoregressiveSequenceAcc(self.depth, self.seqFor,
                                         self.fillFrames,
                                         createAccChild(self.dist),
                                         tag = self.tag)

    def synth(self, (uttId, input), method = SynthMethod.Sample,
              actualOutput = None):
        return list(self.synthIterator((uttId, input), method = method,
                                       actualOutput = actualOutput))

    def synthIterator(self, (uttId, input), method = SynthMethod.Sample,
                      actualOutput = None):
        inSeq = self.seqFor(input)
        actualOutSeq = actualOutput
        outContext = deque(self.fillFrames)
        if actualOutSeq is not None:
            assert len(actualOutSeq) == len(inSeq)
        for frameIndex, inFrame in enumerate(inSeq):
            outFrame = self.dist.synth(
                (inFrame, list(outContext)),
                method,
                None if actualOutSeq is None else actualOutSeq[frameIndex]
            )

            yield outFrame

            outContext.append(outFrame)
            if len(outContext) > self.depth:
                outContext.popleft()

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        dist, paramsLeft = parseChild(self.dist, params)
        overallDistNew = AutoregressiveSequenceDist(self.depth, self.seqFor,
                                                    self.fillFrames, dist,
                                                    tag = self.tag)
        return overallDistNew, paramsLeft

@codeDeps(wnet.FlatMappedNet, wnet.SequenceNet, wnet.probLeftToRightNet)
class SimpleLeftToRightNetFor(object):
    def __init__(self, subLabels):
        self.subLabels = subLabels
    def __repr__(self):
        return 'SimpleLeftToRightNetFor(%r)' % self.subLabels
    def __call__(self, labelSeq):
        net = wnet.FlatMappedNet(
            lambda label: wnet.probLeftToRightNet(
                [ (label, subLabel) for subLabel in self.subLabels ],
                [ [ ((label, subLabel), adv)
                    for adv in [0, 1] ]
                  for subLabel in self.subLabels ]
            ),
            wnet.SequenceNet(labelSeq, None)
        )
        return net

@codeDeps(AutoregressiveNetAcc, Dist, SynthMethod, SynthSeqTooLongError,
    memoize, sampleDiscrete, semiring.LogRealsField, wnet.FlatMappedNet,
    wnet.MappedLabelNet, wnet.PriorityQueueSumAgenda, wnet.TrivialNet,
    wnet.UnrolledNet, wnet.concretizeNetTopSort, wnet.nodeSetCompute, wnet.sum
)
class AutoregressiveNetDist(Dist):
    """An autoregressive distribution over sequences.

    The generative model is that for each input we have a net, and we jump
    forwards through this net probabilistically, generating acoustic output at
    the emitting nodes. The conditional probability of a given transition in the
    net is specified by durDist and the conditional probability of a given
    emission is specified by acDist. Each transition and each emission are
    allowed to be conditioned on the previous emissions up to a given depth. We
    stop emitting when we reach the end node of the net. This whole process is
    therefore a generative model which takes some input and produces a finite
    acoustic output sequence outSeq, where outSeq[t] is the acoustic output at
    "time" t.

    netFor is a function which takes some input and returns a net. The form of
    input is arbitrary, and is only passed to netFor. The emitting nodes of this
    net should be labelled by phonetic context, and the non-None edges should be
    labelled by (phonetic context, phonetic output) pairs. Here phonetic context
    and phonetic output are arbitrary user-specified data. durDist should have a
    (phonetic context, acoustic context) pair as input and a phonetic output as
    output. acDist should have a (phonetic context, acoustic context) pair as
    input and an acoustic output as output. The acoustic context at time t is
    given by conceptually prepending fillFrames to outSeq, then taking as many
    of the previous depth frames from this sequence as possible. fillFrames thus
    affects only the initial frames with t < depth. If fillFrames is set to [],
    the acoustic context for all the initial frames will be shorter than depth.
    If fillFrames has length depth, the context for all the initial frames will
    have length depth. The net returned by netFor should contain no non-emitting
    cycles.

    This class internally expands the net specified by netFor, adding acoustic
    context as appropriate, to form a new "unrolled" net. Each transition in the
    original net has conditional log probability specified by durDist.logProb
    and each emission in the original net has conditional log probability
    specified by acDist.logProb. The conditional log probability of edges with
    label None is fixed at 0.0. As a consistency condition, for any node in the
    original net and for any acoustic context the sum of the probabilities of
    all edges leaving that node forwards must be 1.0. (During synthesis, this
    condition is checked for all nodes along the chosen path). The easiest and
    most natural way to satisfy this condition is as follows -- for each node
    with non-None edges leaving it, use the same phonetic context for all these
    edges, and have the phonetic output for the different edges correspond
    (bijectively) to the set of possible outputs given by durDist. For example,
    if durDist for a given phonetic context (and for any acoustic context) is a
    distribution over [0, 1] and we wish to use this phonetic context for a
    given node, then there should be two edges leaving this node, one with
    phonetic output 0 and one with phonetic output 1, and both with the given
    phonetic context.
    """
    def __init__(self, depth, netFor, fillFrames, durDist, acDist, pruneSpec,
                 tag = None):
        self.depth = depth
        self.netFor = netFor
        self.fillFrames = fillFrames
        self.durDist = durDist
        self.acDist = acDist
        # (FIXME : could argue pruneSpec should be specified as part of
        #   createAcc rather than part of dist itself. Would make it clumsier
        #   to use pruning during logProb computation, though, which we
        #   probably want to do.)
        self.pruneSpec = pruneSpec
        self.tag = tag

        assert len(self.fillFrames) <= self.depth

        self.ring = semiring.LogRealsField()

    def __repr__(self):
        return ('AutoregressiveNetDist(%r, %r, %r, %r, %r, %r, tag=%r)' %
                (self.depth, self.netFor, self.fillFrames, self.durDist,
                 self.acDist, self.pruneSpec, self.tag))

    def children(self):
        return [self.durDist, self.acDist]

    def mapChildren(self, mapChild):
        return AutoregressiveNetDist(self.depth, self.netFor, self.fillFrames,
                                     mapChild(self.durDist),
                                     mapChild(self.acDist),
                                     self.pruneSpec,
                                     tag = self.tag)

    def getNet(self, input):
        net0 = self.netFor(input)
        net1 = wnet.MappedLabelNet(
            lambda (phInput, phOutput): (False, phInput, phOutput),
            net0
        )
        net2 = wnet.FlatMappedNet(
            lambda phInput: wnet.TrivialNet((True, phInput)),
            net1
        )
        def deltaTime(label):
            return 0 if label is None or not label[0] else 1
        net = wnet.concretizeNetTopSort(net2, deltaTime)
        return net, deltaTime

    def getTimedNet(self, input, outSeq, preComputeLabelToWeight = False):
        net, deltaTime = self.getNet(input)
        timedNet = wnet.UnrolledNet(net, startTime = 0, endTime = len(outSeq),
                                    deltaTime = deltaTime)
        labelToWeight = self.getLabelToWeight(outSeq)

        if preComputeLabelToWeight:
            times = range(len(outSeq) + 1)
            times0 = zip(times, times)
            times1 = zip(times, times[1:])
            for node in wnet.nodeSetCompute(net, accessibleOnly = False):
                for label, nextNode in net.next(node, forwards = True):
                    delta = deltaTime(label)
                    assert delta == 0 or delta == 1
                    times0or1 = times0 if delta == 0 else times1
                    for labelStartTime, labelEndTime in times0or1:
                        labelToWeight((label, labelStartTime, labelEndTime))

        return timedNet, labelToWeight

    def getLabelToWeight(self, outSeq):
        numFilled = len(self.fillFrames)
        outSeqFilled = self.fillFrames + outSeq
        def timedLabelToLogProb((label, labelStartTime, labelEndTime)):
            if label is None:
                return 0.0
            else:
                acInput = outSeqFilled[
                    max(labelStartTime - self.depth + numFilled, 0):
                    (labelStartTime + numFilled)
                ]
                if not label[0]:
                    _, phInput, phOutput = label
                    return self.durDist.logProb((phInput, acInput), phOutput)
                else:
                    _, phInput = label
                    assert labelEndTime == labelStartTime + 1
                    acOutput = outSeq[labelStartTime]
                    return self.acDist.logProb((phInput, acInput), acOutput)
        return memoize(timedLabelToLogProb)

    def getAgenda(self, forwards):
        def negMap((time, nodeIndex)):
            return -time, -nodeIndex
        def pruneTrigger(nodePrevPop, nodeCurrPop):
            # compare times
            return (nodePrevPop[0] != nodeCurrPop[0])
        pruneThresh = (None if self.pruneSpec is None
                       else self.pruneSpec.betaThresh)
        agenda = wnet.PriorityQueueSumAgenda(self.ring, forwards,
                                             negMap = negMap,
                                             pruneThresh = pruneThresh,
                                             pruneTrigger = pruneTrigger)
        return agenda

    def logProb(self, (uttId, input), outSeq):
        timedNet, labelToWeight = self.getTimedNet(input, outSeq)
        totalLogProb = wnet.sum(timedNet, labelToWeight = labelToWeight,
                                ring = self.ring, getAgenda = self.getAgenda)
        return totalLogProb

    def logProbDerivOutput(self, (uttId, input), outSeq):
        # FIXME : complete
        notyetimplemented

    def arError(self, (uttId, input), outSeq, distError):
        # FIXME : complete (compute using expectation semiring?)
        notyetimplemented

    def createAcc(self, createAccChild, verbosity = 0):
        return AutoregressiveNetAcc(
            distPrev = self,
            durAcc = createAccChild(self.durDist),
            acAcc = createAccChild(self.acDist),
            verbosity = verbosity,
            tag = self.tag
        )

    def synth(self, (uttId, input), method = SynthMethod.Sample,
              actualOutput = None, maxLength = None):
        # (FIXME : align actualOutSeq and pass down to frames below? (What
        #   exactly do I mean?))
        # (FIXME : can we do anything simple and reasonable with durations for
        #   meanish case?)
        forwards = True
        net = self.netFor(input)
        actualOutSeq = actualOutput
        startNode = net.start(forwards)
        endNode = net.end(forwards)
        assert net.elem(startNode) is None
        assert net.elem(endNode) is None

        outSeq = []
        acInput = deque(self.fillFrames)
        node = startNode
        while node != endNode:
            nodedProbs = []
            for label, nextNode in net.next(node, forwards):
                if label is None:
                    nodedProbs.append((nextNode, 1.0))
                else:
                    phInput, phOutput = label
                    logProb = self.durDist.logProb((phInput, list(acInput)),
                                                   phOutput)
                    nodedProbs.append((nextNode, math.exp(logProb)))
            node = sampleDiscrete(nodedProbs)
            elem = net.elem(node)
            if elem is not None:
                phInput = elem
                acOutput = self.acDist.synth((phInput, list(acInput)), method)
                outSeq.append(acOutput)
                acInput.append(acOutput)
                if len(acInput) > self.depth:
                    acInput.popleft()
            if maxLength is not None and len(outSeq) > maxLength:
                raise SynthSeqTooLongError(
                    'maximum length %r exceeded during synth from'
                    ' AutoregressiveNetDist' % maxLength
                )

        return outSeq

    def paramsSingle(self):
        return []

    def parseSingle(self, params):
        return self, params

    def parseChildren(self, params, parseChild):
        paramsLeft = params
        durDist, paramsLeft = parseChild(self.durDist, paramsLeft)
        acDist, paramsLeft = parseChild(self.acDist, paramsLeft)
        distNew = AutoregressiveNetDist(self.depth, self.netFor,
                                        self.fillFrames,
                                        durDist, acDist,
                                        self.pruneSpec, tag = self.tag)
        return distNew, paramsLeft

@codeDeps(LinearGaussianAcc, LinearGaussianVec, LinearGaussianVecAcc,
    accNodeList
)
class FloorSetter(object):
    """Helper class for setting floors.

    __call__ takes a root acc as input, and mutably sets floors of any sub-accs
    which are of an appropriate type (currently just LinearGaussianAcc and
    LinearGaussianAccVec).

    The set of sub-nodes to consider is customizable by setting getNodes.

    If htsStyle is False then the variance floor of each LinearGaussianAcc is
    set to lgFloorMult times the variance of the dist estimated from that acc.

    If htsStyle is True then the variance floor of each LinearGaussianAcc is
    set to lgFloorMult times the variance of the output.

    If htsStyle is True then it is assumed that the last component of the input
    vector is the bias (this is weakly checked).
    """
    def __init__(self, lgFloorMult, htsStyle = False, getNodes = accNodeList,
                 verbosity = 1):
        self.lgFloorMult = lgFloorMult
        self.htsStyle = htsStyle
        self.getNodes = getNodes
        self.verbosity = verbosity

    def lgFloor(self, acc):
        assert isinstance(acc, LinearGaussianAcc)
        if self.htsStyle:
            assert len(acc.sumTarget) >= 1
            # weak check that last component of input is bias
            assert np.allclose(acc.sumOuter[-1, -1], acc.occ)
            accNew = LinearGaussianAcc(inputLength = 1)
            accNew.occ = acc.occ
            accNew.sumSqr = acc.sumSqr
            accNew.sumTarget = acc.sumTarget[-1:]
            accNew.sumOuter = acc.sumOuter[-1:, -1:]
        else:
            accNew = acc
        variance = accNew.estimateSingleAux()[0].variance
        return variance * self.lgFloorMult

    def lgFloorVec(self, acc):
        assert isinstance(acc, LinearGaussianVecAcc)
        if self.htsStyle:
            assert np.shape(acc.sumOuter)[1] >= 2
            # weak check that last component of input (i.e. second last
            #   component of combined inputOutput) is bias
            assert np.allclose(acc.sumOuter[:, -2, -2], acc.occ)

            accNew = LinearGaussianVecAcc(
                distPrev = LinearGaussianVec(
                    acc.distPrev.coeffVec[:, -1:],
                    acc.distPrev.varianceVec,
                    acc.distPrev.varianceFloorVec
                )
            )
            accNew.occ = acc.occ
            accNew.sumOuter = acc.sumOuter[:, -2:, -2:]
        else:
            accNew = acc
        varianceVec = accNew.estimateSingleAux()[0].varianceVec
        return varianceVec * self.lgFloorMult

    def __call__(self, accRoot):
        if self.verbosity >= 1:
            print ('flooring: setting floors with lgFloorMult = %s,'
                   ' htsStyle = %s' % (self.lgFloorMult, self.htsStyle))
        floorsSet = 0
        # (FIXME : slightly dodgy to change accs like this, but not too bad)
        for acc in self.getNodes(accRoot):
            if isinstance(acc, LinearGaussianAcc):
                varianceFloor = self.lgFloor(acc)
                if self.verbosity >= 2:
                    print ('flooring:    changing variance floor from %s to'
                           ' %s' % (acc.varianceFloor, varianceFloor))
                acc.varianceFloor = varianceFloor
                floorsSet += 1
            elif isinstance(acc, LinearGaussianVecAcc):
                varianceFloorVec = self.lgFloorVec(acc)
                if self.verbosity >= 2:
                    print ('flooring:    changing variance floors from %s to'
                           ' %s' % (acc.varianceFloorVec, varianceFloorVec))
                acc.varianceFloorVec = varianceFloorVec
                floorsSet += len(varianceFloorVec)
        if self.verbosity >= 1:
            print 'flooring: set %s floors' % floorsSet

@codeDeps(BinaryLogisticClassifier, ConstantClassifier, LinearGaussian,
    LinearGaussianVec, distNodeList
)
def reportFloored(distRoot, rootTag):
    dists = distNodeList(distRoot)
    taggedDistTypes = [('LG', LinearGaussian),
                       ('LGV', LinearGaussianVec),
                       ('CC', ConstantClassifier),
                       ('BLC', BinaryLogisticClassifier)]
    numFlooreds = [ np.array([0, 0]) for dtIndex, (dtTag, dt)
                                     in enumerate(taggedDistTypes) ]
    for dist in dists:
        for dtIndex, (dtTag, dt) in enumerate(taggedDistTypes):
            if isinstance(dist, dt):
                numFlooreds[dtIndex] += dist.flooredSingle()
    parts = [ '%s: %s of %s' % (dtTag, numFloored, numTot)
              for (dtTag, dt), (numFloored, numTot)
              in zip(taggedDistTypes, numFlooreds)
              if numTot > 0 ]
    summary = ', '.join(parts)
    if summary:
        print 'flooring: %s: %s' % (rootTag, summary)

@codeDeps(BinaryLogisticClassifier, LinearGaussian, LinearGaussianAcc,
    MixtureDist
)
def estimateInitialMixtureOfTwoExperts(acc):
    """Estimates an initial mixture of two experts from a LinearGaussianAcc.

    N.B. assumes last component of input vector is bias (only weakly checked).
    (N.B. is geometric -- not invariant with respect to scaling of individual
    summarizers (at least for depth > 0).)
    """
    assert isinstance(acc, LinearGaussianAcc)
    if acc.occ == 0.0:
        logging.warning('not mixing up LinearGaussian with occ == 0.0')
        return acc.estimateSingleAuxSafe()[0]
    sigmoidAbscissaAtOneStdev = 0.5
    occRecompute = acc.sumOuter[-1, -1]
    S = acc.sumOuter[:-1, :-1] / occRecompute
    mu = acc.sumOuter[-1, :-1] / occRecompute
    if abs(occRecompute - acc.occ) > 1e-10:
        raise RuntimeError('looks like last component of input vector is not'
                           ' bias (%r vs %r)' % (occRecompute, acc.occ))
    # FIXME : completely different behaviour for depth 0 case!
    #   Can we unify (or improve depth > 0 case with something HTK-like)?
    # FIXME : what about len(acc.sumOuter) == 0 case?
    # (FIXME : hard-coded flooring of 5.0)
    if len(acc.sumOuter) == 1:
        # HTK-style mixture incrementing
        coeff = np.array([0.0])
        coeffFloor = np.array([float('inf')])
        blc = BinaryLogisticClassifier(coeff, coeffFloor)
        dist = acc.estimateSingleAuxSafe()[0]
        mean, = dist.coeff
        variance = dist.variance
        dist0 = LinearGaussian(np.array([mean - 0.2 * math.sqrt(variance)]),
                               variance, acc.varianceFloor)
        dist1 = LinearGaussian(np.array([mean + 0.2 * math.sqrt(variance)]),
                               variance, acc.varianceFloor)
        return MixtureDist(blc, [dist0, dist1])
    else:
        l, U = la.eigh(S - np.outer(mu, mu))
        eigVal, (index, eigVec) = max(zip(l, enumerate(np.transpose(U))))
        if eigVal == 0.0:
            logging.warning('not mixing up LinearGaussian since eigenvalue'
                            ' 0.0')
            return acc.estimateSingleAuxSafe()[0]
        w = eigVec * sigmoidAbscissaAtOneStdev / math.sqrt(eigVal)
        w0 = -np.dot(w, mu)
        coeff = np.append(w, w0)
        coeffFloor = np.append(np.ones((len(w),)) * 5.0, float('inf'))
        coeff = np.minimum(coeff, coeffFloor)
        coeff = np.maximum(coeff, -coeffFloor)
        blc = BinaryLogisticClassifier(coeff, coeffFloor)
        dist0 = acc.estimateSingleAuxSafe()[0]
        dist1 = acc.estimateSingleAuxSafe()[0]
        return MixtureDist(blc, [dist0, dist1])
