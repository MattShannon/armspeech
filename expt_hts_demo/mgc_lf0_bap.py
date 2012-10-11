"""Helper functions specific to case where streams are (mgc, lf0, bap).

More specifically, specific to case where we have 3 streams, with first and
third streams non-"multispace" vectors and second stream a 0/1-dimensional
"multispace" vector.
"""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import numpy as np
import armspeech.numpy_settings

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

@codeDeps()
def computeFrameMeanAndVariance(corpus, mgcOrder, bapOrder):
    mgcSum = np.zeros((mgcOrder,))
    mgcSumSqr = np.zeros((mgcOrder,))
    bapSum = np.zeros((bapOrder,))
    bapSumSqr = np.zeros((bapOrder,))
    numFrames = 0
    for uttId in corpus.trainUttIds:
        (uttId, alignment), acousticSeq = corpus.data(uttId)
        for mgcFrame, lf0Frame, bapFrame in acousticSeq:
            mgcFrame = np.asarray(mgcFrame)
            bapFrame = np.asarray(bapFrame)
            mgcSum += mgcFrame
            mgcSumSqr += mgcFrame * mgcFrame
            bapSum += bapFrame
            bapSumSqr += bapFrame * bapFrame
        numFrames += len(acousticSeq)
    mgcMean = mgcSum / numFrames
    bapMean = bapSum / numFrames
    mgcVariance = mgcSumSqr / numFrames - mgcMean * mgcMean
    bapVariance = bapSumSqr / numFrames - bapMean * bapMean
    return (mgcMean, bapMean), (mgcVariance, bapVariance)
