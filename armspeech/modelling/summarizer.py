"""Some useful summarizers.

Summarizers condense (summarize) past input."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from __future__ import division

import numpy as np

from codedep import codeDeps

import armspeech.modelling.dist as d
import armspeech.numpy_settings

@codeDeps()
class ContextualVectorSummarizer(object):
    def __init__(self, vectorSummarizer):
        self.vectorSummarizer = vectorSummarizer

    def __repr__(self):
        return 'ContextualVectorSummarizer('+repr(self.vectorSummarizer)+')'

    def __call__(self, input, partialOutput, outIndex):
        context, vectorInput = input
        summary = self.vectorSummarizer(vectorInput, partialOutput, outIndex)
        return context, summary

@codeDeps(ContextualVectorSummarizer, d.createVectorAcc, d.createVectorDist)
class IdentitySummarizer(object):
    def __init__(self, order, outIndices = None):
        if outIndices is None:
            outIndices = range(order)
        self.order = order
        self.outIndices = outIndices

    def __repr__(self):
        return 'IdentitySummarizer('+repr(self.order)+', '+repr(self.outIndices)+')'

    def __call__(self, input, partialOutput, outIndex):
        return input

    def createDist(self, contextual, createDistForIndex):
        vectorSummarizer = ContextualVectorSummarizer(self) if contextual else self
        return d.createVectorDist(self.order, self.outIndices, vectorSummarizer, createDistForIndex)

    def createAcc(self, contextual, createAccForIndex):
        vectorSummarizer = ContextualVectorSummarizer(self) if contextual else self
        return d.createVectorAcc(self.order, self.outIndices, vectorSummarizer, createAccForIndex)

@codeDeps(ContextualVectorSummarizer, d.createVectorAcc, d.createVectorDist)
class VectorSeqSummarizer(object):
    def __init__(self, order, depths, strictAboutDepth = True):
        self.order = order
        self.depths = depths
        self.strictAboutDepth = strictAboutDepth

    def __repr__(self):
        return 'VectorSeqSummarizer('+repr(self.order)+', '+repr(self.depths)+')'

    def __call__(self, input, partialOutput, outIndex):
        depth = self.depths[outIndex]
        assert depth >= 0
        if self.strictAboutDepth and len(input) < depth:
            raise RuntimeError('input to summarize is too short (length of input '+str(len(input))+' < specified depth '+str(depth)+')')
        startSummaryIndex = max(len(input) - depth, 0)
        endSummaryIndex = len(input)
        summary = [ v[outIndex] for v in input[startSummaryIndex:endSummaryIndex] ]
        return summary

    def createDist(self, contextual, createDistForIndex):
        vectorSummarizer = ContextualVectorSummarizer(self) if contextual else self
        return d.createVectorDist(self.order, sorted(self.depths.keys()), vectorSummarizer, createDistForIndex)

    def createAcc(self, contextual, createAccForIndex):
        vectorSummarizer = ContextualVectorSummarizer(self) if contextual else self
        return d.createVectorAcc(self.order, sorted(self.depths.keys()), vectorSummarizer, createAccForIndex)

@codeDeps(ContextualVectorSummarizer, d.createVectorAcc, d.createVectorDist)
class IndexSpecSummarizer(object):
    def __init__(self, outIndices, fromOffset, toOffset, order, depth, powers = [1]):
        self.outIndices = outIndices
        self.fromOffset = fromOffset
        self.toOffset = toOffset
        self.order = order
        self.depth = depth
        self.powers = powers

        self.limits = dict()
        for outIndex in outIndices:
            inFromIndex = min(max(outIndex + fromOffset, 0), order)
            inUntilIndex = min(max(outIndex + toOffset + 1, 0), order)
            self.limits[outIndex] = inFromIndex, inUntilIndex

    def __repr__(self):
        return 'IndexSpecSummarizer('+repr(self.outIndices)+', '+repr(self.fromOffset)+', '+repr(self.toOffset)+', '+repr(self.order)+', '+repr(self.depth)+', '+repr(self.powers)+')'

    def __call__(self, input, partialOutput, outIndex):
        if not outIndex in self.limits or len(input) != self.depth:
            raise RuntimeError('invalid input to summarize: '+repr(input))
        inFromIndex, inUntilIndex = self.limits[outIndex]
        summary = []
        for pastVec in input:
            for index in range(inFromIndex, inUntilIndex):
                for power in self.powers:
                    summary.append(pastVec[index] ** power)
        for index in range(inFromIndex, min(outIndex, inUntilIndex)):
            for power in self.powers:
                summary.append(partialOutput[index] ** power)
        return np.array(summary)

    def vectorLength(self, outIndex):
        inFromIndex, inUntilIndex = self.limits[outIndex]
        summaryLength = max(inUntilIndex - inFromIndex, 0) * len(self.powers) * self.depth + max(min(outIndex, inUntilIndex) - inFromIndex, 0) * len(self.powers)
        return summaryLength

    def createDist(self, contextual, createDistForIndex):
        vectorSummarizer = ContextualVectorSummarizer(self) if contextual else self
        return d.createVectorDist(self.order, self.outIndices, vectorSummarizer, createDistForIndex)

    def createAcc(self, contextual, createAccForIndex):
        vectorSummarizer = ContextualVectorSummarizer(self) if contextual else self
        return d.createVectorAcc(self.order, self.outIndices, vectorSummarizer, createAccForIndex)
