"""General-purpose corpus abstraction."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import dist as d

class Corpus(object):
    def accumulate(self, acc):
        for uttId in self.trainUttIds:
            input, output = self.data(uttId)
            acc.add(input, output)

    def sum(self, uttIds, computeValue):
        return sum([ computeValue(*self.data(uttId)) for uttId in uttIds ])

    def logProb(self, dist, uttIds):
        def computeValue(input, output):
            return dist.logProb(input, output)
        return self.sum(uttIds, computeValue)

    def outError(self, dist, uttIds, vecError, frameToVec = lambda frame: frame, outputToFrameSeq = lambda output: output):
        def computeValue(input, actualOutput):
            synthOutput = dist.synth(input, d.SynthMethod.Meanish, actualOutput)
            synthFrameSeq = outputToFrameSeq(synthOutput)
            actualFrameSeq = outputToFrameSeq(actualOutput)
            if len(actualFrameSeq) != len(synthFrameSeq):
                raise RuntimeError('actual and synthesized sequences must have the same length to compute error')
            error = sum([ vecError(frameToVec(synthFrame), frameToVec(actualFrame)) for synthFrame, actualFrame in zip(synthFrameSeq, actualFrameSeq) ])
            return error
        return self.sum(uttIds, computeValue)

    def synth(self, dist, uttId, method = d.SynthMethod.Sample):
        input, actualOutput = self.data(uttId)
        return dist.synth(input, method, actualOutput)
