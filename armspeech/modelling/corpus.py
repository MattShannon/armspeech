"""General-purpose corpus abstraction."""

# Copyright 2011 Matt Shannon

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

    def synth(self, dist, uttId, method = d.SynthMethod.Sample):
        input, actualOutput = self.data(uttId)
        return dist.synth(input, method, actualOutput)
