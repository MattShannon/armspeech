"""General-purpose corpus abstraction."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import dist as d

import math
import numpy as np
import numpy.linalg as la

mcdConstant = 10.0 / math.log(10.0) * math.sqrt(2.0)

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

    def cepDist_frames(self, dist, uttIds, extractVectorSeq):
        cd = 0.0
        frames = 0
        for uttId in uttIds:
            input, actualOutput = self.data(uttId)
            synthOutput = dist.synth(input, method = d.SynthMethod.Meanish, actualOutput = actualOutput)
            actualSeq = extractVectorSeq(actualOutput)
            synthSeq = extractVectorSeq(synthOutput)
            if len(actualSeq) != len(synthSeq):
                raise RuntimeError('actual and synthesized sequences must have the same length to compute MCD')
            cdD = mcdConstant * sum([ la.norm(np.asarray(actualFrame) - np.asarray(synthFrame)) for actualFrame, synthFrame in zip(actualSeq, synthSeq) ])
            framesD = len(actualSeq)
            cd += cdD
            frames += framesD
        return cd, frames

    def synth(self, dist, uttId, method = d.SynthMethod.Sample):
        input, actualOutput = self.data(uttId)
        return dist.synth(input, method, actualOutput)
