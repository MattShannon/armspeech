"""Helper functions specific to case where streams are (mgc, lf0, bap).

More specifically, specific to case where we have 3 streams, with first and
third streams non-"multispace" vectors and second stream a 0/1-dimensional
"multispace" vector.
"""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.modelling import summarizer
from codedep import codeDeps

import numpy as np
import armspeech.numpy_settings

@codeDeps(summarizer.IndexSpecSummarizer, summarizer.VectorSeqSummarizer)
class BasicArModelInfo(object):
    def __init__(self, phoneset, subLabels,
                 mgcOrder, bapOrder,
                 mgcDepth, lf0Depth, bapDepth,
                 mgcIndices, bapIndices,
                 mgcUseVec, bapUseVec):
        self.phoneset = phoneset
        self.subLabels = subLabels
        self.mgcOrder = mgcOrder
        self.bapOrder = bapOrder
        self.mgcDepth = mgcDepth
        self.lf0Depth = lf0Depth
        self.bapDepth = bapDepth
        self.mgcIndices = mgcIndices
        self.bapIndices = bapIndices
        self.mgcUseVec = mgcUseVec
        self.bapUseVec = bapUseVec

        self.maxDepth = max(self.mgcDepth, self.lf0Depth, self.bapDepth)
        self.streamDepths = {0: self.mgcDepth,
                             1: self.lf0Depth,
                             2: self.bapDepth}
        self.frameSummarizer = summarizer.VectorSeqSummarizer(
            order = 3,
            depths = self.streamDepths
        )

        self.mgcSummarizer = summarizer.IndexSpecSummarizer(
            self.mgcIndices, fromOffset = 0, toOffset = 0,
            order = self.mgcOrder, depth = self.mgcDepth
        )
        self.bapSummarizer = summarizer.IndexSpecSummarizer(
            self.bapIndices, fromOffset = 0, toOffset = 0,
            order = self.bapOrder, depth = self.bapDepth
        )

        if self.mgcUseVec:
            assert self.mgcIndices == range(self.mgcOrder)
        if self.bapUseVec:
            assert self.bapIndices == range(self.bapOrder)

@codeDeps()
def getZeroFrame(mgcOrder, bapOrder):
    return np.zeros((mgcOrder,)), None, np.zeros((bapOrder,))

@codeDeps()
def computeFirstFrameAverage(corpus, mgcOrder, bapOrder):
    mgcFirstFrameAverage = np.zeros((mgcOrder,))
    lf0FirstFrameProportionUnvoiced = 0.0
    bapFirstFrameAverage = np.zeros((bapOrder,))
    numUtts = 0
    for uttId in corpus.trainUttIds:
        (uttId, alignment), acousticSeq = corpus.data(uttId)
        mgcFrame, lf0Frame, bapFrame = acousticSeq[0]
        mgcFirstFrameAverage += mgcFrame
        lf0FirstFrameProportionUnvoiced += (1.0 if lf0Frame == (0, None) else 0.0)
        bapFirstFrameAverage += bapFrame
        numUtts += 1
    mgcFirstFrameAverage /= numUtts
    lf0FirstFrameProportionUnvoiced /= numUtts
    bapFirstFrameAverage /= numUtts
    # (FIXME : should probably just return average lf0 for voiced first frames
    #   if more than half of the first frames are voiced)
    assert lf0FirstFrameProportionUnvoiced >= 0.5
    return mgcFirstFrameAverage, None, bapFirstFrameAverage
