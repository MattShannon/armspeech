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

    def logProb_frames(self, dist, uttIds):
        lp = 0.0
        frames = 0
        for uttId in uttIds:
            input, output = self.data(uttId)
            lpD, framesD = dist.logProb_frames(input, output)
            lp += lpD
            frames += framesD
        return lp, frames

    def arError_frames(self, dist, uttIds, distError):
        error = 0.0
        frames = 0
        for uttId in uttIds:
            input, output = self.data(uttId)
            errorD, framesD = dist.arError_frames(input, output, distError)
            error += errorD
            frames += framesD
        return error, frames

    def arOutError_frames(self, dist, uttIds, vecError, frameToVec = lambda frame: frame):
        def distError(dist, input, actualFrame):
            synthFrame = dist.synth(input, d.SynthMethod.Meanish, actualFrame)
            return vecError(frameToVec(synthFrame), frameToVec(actualFrame))
        return self.arError_frames(dist, uttIds, distError)

    def outError_frames(self, dist, uttIds, vecError, frameToVec = lambda frame: frame, outputToFrameSeq = lambda output: output):
        error = 0.0
        frames = 0
        for uttId in uttIds:
            input, actualOutput = self.data(uttId)
            synthOutput = dist.synth(input, d.SynthMethod.Meanish, actualOutput)
            synthFrameSeq = outputToFrameSeq(synthOutput)
            actualFrameSeq = outputToFrameSeq(actualOutput)
            if len(actualFrameSeq) != len(synthFrameSeq):
                raise RuntimeError('actual and synthesized sequences must have the same length to compute error')
            errorD = sum([ vecError(frameToVec(synthFrame), frameToVec(actualFrame)) for synthFrame, actualFrame in zip(synthFrameSeq, actualFrameSeq) ])
            framesD = len(actualFrameSeq)
            error += errorD
            frames += framesD
        return error, frames

    def synth(self, dist, uttId, method = d.SynthMethod.Sample):
        input, actualOutput = self.data(uttId)
        return dist.synth(input, method, actualOutput)
