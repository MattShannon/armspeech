"""General-purpose corpus abstraction."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import dist as d

class Corpus(object):
    def accumulate(self, acc):
        for uttId in self.trainUttIds:
            alignment, acousticSeq = self.data(uttId)
            acc.add(alignment, acousticSeq)

    def logProb_frames(self, dist, uttIds):
        lp = 0.0
        frames = 0
        for uttId in uttIds:
            alignment, acousticSeq = self.data(uttId)
            lpD, framesD = dist.logProb_frames(alignment, acousticSeq)
            lp += lpD
            frames += framesD
        return lp, frames

    def synth(self, dist, uttId, method = d.SynthMethod.Sample):
        alignment, actualAcousticSeq = self.data(uttId)
        return dist.synth(alignment, method, actualAcousticSeq)
