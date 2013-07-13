"""Example experiments."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.modelling.alignment as align
from armspeech.modelling.alignment import StandardizeAlignment
from armspeech.modelling import nodetree
import armspeech.modelling.dist as d
import armspeech.modelling.train as trn
from armspeech.modelling import summarizer
import armspeech.modelling.transform as xf
import armspeech.modelling.questions as ques
from armspeech.modelling import cluster
from armspeech.modelling import wnet
from armspeech.speech.features import stdCepDist, stdCepDistIncZero
from armspeech.speech import draw
from armspeech.util.util import identityFn, ConstantFn
from armspeech.util.util import getElem, ElemGetter, AttrGetter
from armspeech.util import persist
from armspeech.util.timing import timed, printTime
from armspeech.modelling.bisque import corpus as corpus_bisque
from armspeech.modelling.bisque import train as train_bisque
from armspeech.bisque.distribute import liftLocal, lit, lift
from codedep import codeDeps

import phoneset_cmu
import labels_hts_demo
import questions_hts_demo
import mgc_lf0_bap
import corpus_arctic

import os
import math
import numpy as np
import armspeech.numpy_settings
from collections import defaultdict

@codeDeps(d.AutoregressiveNetAcc, d.accNodeList, getElem,
    nodetree.findTaggedNodes, nodetree.nodeList
)
def reportLogLikeBreakdown(accRoot):
    def getAccLists(accRoot):
        def isStreamRoot(tag):
            return getElem(tag, 0, 2) == 'stream'
        def isNetAcc(node):
            return isinstance(node, d.AutoregressiveNetAcc)
        names = []
        accLists = []
        for streamRoot in nodetree.findTaggedNodes(accRoot, isStreamRoot):
            streamName = streamRoot.tag[1]
            names.append(streamName)
            accLists.append(d.accNodeList(streamRoot))
        names.append('net')
        accLists.append(nodetree.nodeList(accRoot, includeNode = isNetAcc))
        return names, accLists

    names, accLists = getAccLists(accRoot)
    accIdSets = [ set([ id(acc) for acc in accList ]) for accList in accLists ]

    logLikeTot = 0.0
    numNodesTot = 0
    logLikeDict = defaultdict(float)
    numNodesDict = defaultdict(int)
    for accNode in d.accNodeList(accRoot):
        accId = id(accNode)
        inSets = tuple([ (accId in accIdSet) for accIdSet in accIdSets ])
        logLike = accNode.logLikeSingle()
        logLikeTot += logLike
        numNodesTot += 1
        logLikeDict[inSets] += logLike
        numNodesDict[inSets] += 1

    assert np.allclose(logLikeTot, accRoot.logLike())
    assert np.allclose(sum(logLikeDict.values()), logLikeTot)

    sortedInSetsList = sorted(
        logLikeDict.keys(),
        key = lambda inSets: (
            # place "other" at end
            len(inSets) - sum(inSets) if sum(inSets) > 0 else -1,
            inSets
        ),
        reverse = True
    )

    count = accRoot.count()
    count = max(count, 1.0)

    print 'train: breakdown of log like (count = %s)' % count
    for inSets in sortedInSetsList:
        logLike = logLikeDict[inSets]
        numNodes = numNodesDict[inSets]
        inSetsNames = [ name for name, inSet in zip(names, inSets) if inSet ]
        inSetsDesc = ' and '.join(inSetsNames) if inSetsNames else 'other'
        print 'train:    %s: %s (%s acc nodes)' % (inSetsDesc, logLike / count,
                                                   numNodes)
    print 'train: %s: %s (%s acc nodes)' % ('total', logLikeTot / count,
                                            numNodesTot)

# (FIXME : re-jig how reportLogLikeBreakdown is used so it reports on current
#   dist (rather than previous dist) more often?)

@codeDeps(reportLogLikeBreakdown)
def getReportLogLikeBreakdown():
    return reportLogLikeBreakdown

@codeDeps(d.reportFloored, getElem, nodetree.findTaggedNodes)
def reportFlooredPerStream(dist):
    def isStreamRoot(tag):
        return getElem(tag, 0, 2) == 'stream'
    for streamRoot in nodetree.findTaggedNodes(dist, isStreamRoot):
        streamName = streamRoot.tag[1]
        d.reportFloored(streamRoot, rootTag = streamName)

@codeDeps(d.Rat)
def reportTrainAux((trainAux, trainAuxRat), trainFrames):
    print 'training aux = %s (%s) (%s frames)' % (trainAux / trainFrames, d.Rat.toString(trainAuxRat), trainFrames)

@codeDeps()
def evaluateLogProb(dist, corpus):
    trainLogProb = corpus.logProb(dist, corpus.trainUttIds)
    trainFrames = corpus.frames(corpus.trainUttIds)
    print 'train set log prob = %s (%s frames)' % (trainLogProb / trainFrames, trainFrames)
    testLogProb = corpus.logProb(dist, corpus.testUttIds)
    testFrames = corpus.frames(corpus.testUttIds)
    print 'test set log prob = %s (%s frames)' % (testLogProb / testFrames, testFrames)
    print
    return [('train log prob', (trainLogProb, trainFrames)),
            ('test log prob', (testLogProb, testFrames))]

@codeDeps(d.SynthMethod, stdCepDist)
def evaluateMgcArOutError(dist, corpus, vecError = stdCepDist, desc = 'MARCD'):
    def frameToVec(frame):
        mgcFrame, lf0Frame, bapFrame = frame
        return mgcFrame
    def distError(dist, input, actualFrame):
        synthFrame = dist.synth(input, d.SynthMethod.Meanish, actualFrame)
        return vecError(frameToVec(synthFrame), frameToVec(actualFrame))
    def computeValue(inputUtt, outputUtt):
        return dist.sum(inputUtt, outputUtt, distError)
    trainError = corpus.sum(corpus.trainUttIds, computeValue)
    trainFrames = corpus.frames(corpus.trainUttIds)
    print 'train set %s = %s (%s frames)' % (desc, trainError / trainFrames, trainFrames)
    testError = corpus.sum(corpus.testUttIds, computeValue)
    testFrames = corpus.frames(corpus.testUttIds)
    print 'test set %s = %s (%s frames)' % (desc, testError / testFrames, testFrames)
    print
    return [('train %s' % desc, (trainError, trainFrames)),
            ('test %s' % desc, (testError, testFrames))]

@codeDeps(stdCepDist)
def evaluateMgcOutError(dist, corpus, vecError = stdCepDist, desc = 'MCD'):
    def frameToVec(frame):
        mgcFrame, lf0Frame, bapFrame = frame
        return mgcFrame
    trainError = corpus.outError(dist, corpus.trainUttIds, vecError, frameToVec)
    trainFrames = corpus.frames(corpus.trainUttIds)
    print 'train set %s = %s (%s frames)' % (desc, trainError / trainFrames, trainFrames)
    testError = corpus.outError(dist, corpus.testUttIds, vecError, frameToVec)
    testFrames = corpus.frames(corpus.testUttIds)
    print 'test set %s = %s (%s frames)' % (desc, testError / testFrames, testFrames)
    print
    return [('train %s' % desc, (trainError, trainFrames)),
            ('test %s' % desc, (testError, testFrames))]

@codeDeps(d.SynthMethod)
def evaluateSynthesize(dist, corpus, synthOutDir, exptTag, afterSynth = None):
    corpus.synthComplete(dist, corpus.synthUttIds, d.SynthMethod.Sample, synthOutDir, exptTag+'.sample', afterSynth = afterSynth)
    corpus.synthComplete(dist, corpus.synthUttIds, d.SynthMethod.Meanish, synthOutDir, exptTag+'.meanish', afterSynth = afterSynth)

@codeDeps(draw.drawLabelledSeq, draw.partitionSeq)
def getDrawMgc(corpus, mgcIndices, figOutDir, ylims = None, includeGivenLabels = True, extraLabelSeqs = []):
    streamIndex = 0
    def drawMgc(synthOutput, uttId, exptTag):
        (uttId, alignment), trueOutput = corpus.data(uttId)

        alignmentToDraw = [ (start * corpus.framePeriod, end * corpus.framePeriod, label.phone) for start, end, label, subAlignment in alignment ]
        partitionedLabelSeqs = (draw.partitionSeq(alignmentToDraw, 2) if includeGivenLabels else []) + [ labelSeqSub for labelSeq in extraLabelSeqs for labelSeqSub in draw.partitionSeq(labelSeq, 2) ]

        trueSeqTime = (np.array(range(len(trueOutput))) + 0.5) * corpus.framePeriod
        synthSeqTime = (np.array(range(len(synthOutput))) + 0.5) * corpus.framePeriod

        for mgcIndex in mgcIndices:
            trueSeq = [ frame[streamIndex][mgcIndex] for frame in trueOutput ]
            synthSeq = [ frame[streamIndex][mgcIndex] for frame in synthOutput ]

            outPdf = os.path.join(figOutDir, uttId+'-mgc'+str(mgcIndex)+'-'+exptTag+'.pdf')
            draw.drawLabelledSeq([(trueSeqTime, trueSeq), (synthSeqTime, synthSeq)], partitionedLabelSeqs, outPdf = outPdf, figSizeRate = 10.0, ylims = ylims, labelColors = ['red', 'purple'])
    return drawMgc

@codeDeps(evaluateLogProb, evaluateMgcArOutError, evaluateMgcOutError,
    evaluateSynthesize, getDrawMgc, persist.savePickle, reportFlooredPerStream,
    stdCepDistIncZero
)
def evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag, vecError = stdCepDistIncZero):
    # FIXME : vecError default should probably be changed to stdCepDist eventually
    reportFlooredPerStream(dist)
    # (FIXME : perhaps this shouldn't really go in synthOutDir)
    persist.savePickle(os.path.join(synthOutDir, 'dist-%s.pkl' % exptTag),
                       dist)
    logProbResults = evaluateLogProb(dist, corpus)
    marcdResults = evaluateMgcOutError(dist, corpus, vecError = vecError)
    mcdResults = evaluateMgcArOutError(dist, corpus, vecError = vecError)
    evaluateSynthesize(dist, corpus, synthOutDir, exptTag, afterSynth = getDrawMgc(corpus, bmi.mgcSummarizer.outIndices, figOutDir))
    return logProbResults + marcdResults + mcdResults

@codeDeps(d.defaultEstimatePartial, nodetree.getDagMap,
    trn.mixupLinearGaussianEstimatePartial
)
def getMixupEstimate():
    return nodetree.getDagMap([trn.mixupLinearGaussianEstimatePartial,
                               d.defaultEstimatePartial])

@codeDeps(d.getDefaultCreateAcc, getMixupEstimate, trn.trainEM)
def mixup(dist, accumulate):
    print
    print 'MIXING UP'
    acc = d.getDefaultCreateAcc()(dist)
    accumulate(acc)
    logLikeInit = acc.logLike()
    framesInit = acc.count()
    print 'initial training log likelihood = %s (%s frames)' % (logLikeInit / framesInit, framesInit)
    dist = getMixupEstimate()(acc)
    dist = trn.trainEM(dist, accumulate, deltaThresh = 1e-4, minIterations = 4, maxIterations = 8, verbosity = 2)
    return dist

@codeDeps(getMixupEstimate, getReportLogLikeBreakdown, liftLocal, lit,
    train_bisque.accumulateJobSet, train_bisque.estimateJobSet,
    train_bisque.trainEMJobSet
)
def mixupJobSet(distArt, corpusArt, uttIdChunkArts):
    accArts = train_bisque.accumulateJobSet(distArt, corpusArt, uttIdChunkArts)
    distArt = train_bisque.estimateJobSet(
        distArt,
        accArts,
        estimateArt = liftLocal(getMixupEstimate)(),
        afterAccArt = liftLocal(getReportLogLikeBreakdown)(),
        verbosityArt = lit(2)
    )
    distArt = train_bisque.trainEMJobSet(
        distArt,
        corpusArt,
        uttIdChunkArts,
        numIterationsLit = lit(8),
        afterAccArt = liftLocal(getReportLogLikeBreakdown)(),
        verbosityArt = lit(2)
    )
    return distArt

@codeDeps(d.DebugDist, d.LinearGaussian, d.MappedInputDist, d.StudentDist,
    d.TransformedOutputDist, nodetree.defaultMapPartial, nodetree.getDagMap,
    xf.ConstantTransform, xf.DotProductTransform, xf.ShiftOutputTransform
)
def convertToStudentResiduals(dist, studentTag = None, debugTag = None, subtractMeanTag = None):
    def studentResidualsMapPartial(dist, mapChild):
        if isinstance(dist, d.LinearGaussian):
            subDist = d.StudentDist(df = 10000.0, precision = 1.0 / dist.variance).withTag(studentTag)
            if debugTag is not None:
                subDist = d.DebugDist(None, subDist).withTag(debugTag)
            subtractMeanTransform = xf.ShiftOutputTransform(xf.DotProductTransform(-dist.coeff)).withTag(subtractMeanTag)
            distNew = d.TransformedOutputDist(subtractMeanTransform,
                d.MappedInputDist(xf.ConstantTransform(np.array([])),
                    subDist
                )
            )
            return distNew
    studentResidualsMap = nodetree.getDagMap([studentResidualsMapPartial, nodetree.defaultMapPartial])

    return studentResidualsMap(dist)

@codeDeps(d.DebugDist, d.LinearGaussian, d.MappedInputDist,
    d.TransformedOutputDist, nodetree.defaultMapPartial, nodetree.getDagMap,
    xf.ConstantTransform, xf.DotProductTransform, xf.IdentityTransform,
    xf.ShiftOutputTransform, xf.SimpleOutputTransform, xf.SumTransform1D,
    xf.TanhTransform1D
)
def convertToTransformedGaussianResiduals(dist, residualTransformTag = None, debugTag = None, subtractMeanTag = None):
    def transformedGaussianResidualsMapPartial(dist, mapChild):
        if isinstance(dist, d.LinearGaussian):
            residualTransform = xf.SimpleOutputTransform(xf.SumTransform1D([
                xf.IdentityTransform()
            ,   xf.TanhTransform1D(np.array([0.0, 0.5, -1.0]))
            ,   xf.TanhTransform1D(np.array([0.0, 0.5, 0.0]))
            ,   xf.TanhTransform1D(np.array([0.0, 0.5, 1.0]))
            ]), checkDerivPositive1D = True).withTag(residualTransformTag)

            subDist = d.TransformedOutputDist(residualTransform,
                d.LinearGaussian(np.array([]), dist.variance, varianceFloor = 0.0)
            )
            if debugTag is not None:
                subDist = d.DebugDist(None, subDist).withTag(debugTag)
            subtractMeanTransform = xf.ShiftOutputTransform(xf.DotProductTransform(-dist.coeff)).withTag(subtractMeanTag)
            distNew = d.TransformedOutputDist(subtractMeanTransform,
                d.MappedInputDist(xf.ConstantTransform(np.array([])),
                    subDist
                )
            )
            return distNew
    transformedGaussianResidualsMap = nodetree.getDagMap([transformedGaussianResidualsMapPartial, nodetree.defaultMapPartial])

    return transformedGaussianResidualsMap(dist)

@codeDeps(d.LinearGaussian, d.MappedInputDist, xf.AddBias)
def createLinearGaussianVectorDist(indexSpecSummarizer):
    """Creates a linear-Gaussian vector dist.

    Expects input acInput where acInput is a sequence of vectors.
    Expects output a vector.
    """
    def distFor(outIndex):
        inputLength = indexSpecSummarizer.vectorLength(outIndex) + 1
        return d.MappedInputDist(xf.AddBias(),
            d.LinearGaussian(
                coeff = np.zeros((inputLength,)),
                variance = 1.0,
                varianceFloor = 0.0
            )
        )

    return indexSpecSummarizer.createDist(False, distFor)

@codeDeps(d.LinearGaussianVec, d.MappedInputDist, xf.AddBiasVec)
def createLinearGaussianVecDist(order, vecLength):
    """Creates a LinearGaussianVec dist.

    Expects input acInput where acInput is a sequence of vectors.
    Expects output a vector.
    """
    return d.MappedInputDist(xf.AddBiasVec(),
        d.LinearGaussianVec(
            coeffVec = np.zeros((order, vecLength + 1)),
            varianceVec = np.ones((order,)),
            varianceFloorVec = np.zeros((order,))
        )
    )

@codeDeps(d.GaussianVec)
def createGaussianVecDist(order):
    """Creates a GaussianVec dist.

    Expects input arbitrary.
    Expects output a vector.
    """
    return d.GaussianVec(
        meanVec = np.zeros((order,)),
        varianceVec = np.ones((order,)),
        varianceFloorVec = np.zeros((order,))
    )

@codeDeps(d.LinearGaussian, d.MappedInputDist, xf.AddBias)
def createLinearGaussianWithTimingVectorDist(indexSpecSummarizer):
    """Creates a linear-Gaussian vector dist where input includes timing info.

    Expects input (timingInfo, acInput) where timingInfo is
    (framesBefore, framesAfter) and acInput is a sequence of vectors.
    Expects output a vector.
    """
    extraLength = 2
    def distFor(outIndex):
        inputLength = (indexSpecSummarizer.vectorLength(outIndex) +
                       extraLength + 1)
        # from (timingInfo, acInputSingle) to (timingInfo + acInputSingle)
        #   where acInputSingle is a vector which is the acoustic context for a
        #   single outIndex
        return d.MappedInputDist(np.concatenate,
            d.MappedInputDist(xf.AddBias(),
                d.LinearGaussian(
                    coeff = np.zeros((inputLength,)),
                    variance = 1.0,
                    varianceFloor = 0.0
                )
            )
        )

    return indexSpecSummarizer.createDist(True, distFor)

@codeDeps(d.BinaryLogisticClassifier, d.FixedValueDist,
    d.IdentifiableMixtureDist, d.LinearGaussian, d.MappedInputDist, xf.AddBias,
    xf.Msd01ToVector
)
def createBlcBasedLf0Dist(lf0Depth):
    """Creates a BinaryLogisticClassifier-based dist suitable for use with lf0.

    Expects input acInput where acInput is a sequence of MSD elements.
    Expects output an MSD element.
    Here an MSD element is a pair which is either (0, None) or (1, value) where
    value is a scalar value.
    """
    return d.MappedInputDist(xf.Msd01ToVector(),
        d.MappedInputDist(xf.AddBias(),
            d.IdentifiableMixtureDist(
                d.BinaryLogisticClassifier(
                    coeff = np.zeros((2 * lf0Depth + 1,)),
                    coeffFloor = np.ones((2 * lf0Depth + 1,)) * 5.0
                ),
                [
                    d.FixedValueDist(None),
                    d.LinearGaussian(
                        coeff = np.zeros((2 * lf0Depth + 1,)),
                        variance = 1.0,
                        varianceFloor = 0.0
                    )
                ]
            )
        )
    )

@codeDeps(d.ConstantClassifier)
def createCcBinaryDist(ccProbFloor):
    """Creates a ConstantClassifier duration dist.

    Expects input acInput where acInput is a sequence of acoustic frames.
    Expects output (0 or 1).
    """
    return d.ConstantClassifier(
        probs = np.array([0.5, 0.5]),
        probFloors = np.array([ccProbFloor, ccProbFloor])
    )

@codeDeps()
def tupleMap0(((a, b), c)):
    return a, (b, c)

@codeDeps()
def tupleMap1(((label, subLabel), acInput)):
    return subLabel, (label, acInput)

@codeDeps()
def tupleMap1Phone(((label, subLabel), acInput)):
    return subLabel, (label.phone, acInput)

@codeDeps()
def tupleMap2((((label, subLabel), framesLeft), acInput)):
    return subLabel, ((label, framesLeft), acInput)

@codeDeps(d.MappedInputDist, d.createDiscreteDist, d.isolateDist, getElem,
    nodetree.defaultMapPartial, nodetree.findTaggedNode, nodetree.getDagMap,
    tupleMap1Phone
)
def globalToMonophoneMap(dist, bmi, tupleMap = tupleMap1Phone):
    """Converts a global dist to a monophone dist.

    Expects stream dist tag to be ('stream', streamName).
    Expects acoustic vector dist tag to be 'acVec'.
    There should be a single acoustic vector dist below each stream dist.
    Input to stream dist should match input to tupleMap; in default
    tupleMap1Phone case this input should be ((label, subLabel), acInput).
    Output of tupleMap should be (subLabel, (phone, acInput)).
    Input to acoustic vector dist should be acInput.
    Here acInput is arbitrary.

    The returned dist has the same tags and inputs for stream dists and
    acoustic vector dists.
    """
    def globalToMonophoneMapPartial(dist, mapChild):
        if getElem(dist.tag, 0, 2) == 'stream':
            acVecDist = nodetree.findTaggedNode(dist,
                                                lambda tag: tag == 'acVec')
            return d.MappedInputDist(tupleMap,
                d.createDiscreteDist(bmi.subLabels, lambda subLabel:
                    d.createDiscreteDist(bmi.phoneset.phoneList, lambda phone:
                        d.isolateDist(acVecDist)
                    )
                )
            ).withTag(dist.tag)

    return nodetree.getDagMap([globalToMonophoneMapPartial,
                               nodetree.defaultMapPartial])(dist)

@codeDeps(ElemGetter, d.AutoGrowingDiscreteAcc, d.MappedInputAcc,
    d.createDiscreteAcc, d.defaultCreateAccPartial, d.getDefaultCreateAcc,
    getElem, nodetree.findTaggedNode, nodetree.getDagMap, tupleMap1
)
def globalToFullCtxCreateAcc(dist, bmi, agTags = None, tupleMap = tupleMap1):
    """Converts a global dist to a full context acc.

    Expects stream dist tag to be ('stream', streamName).
    Expects acoustic vector dist tag to be 'acVec'.
    There should be a single acoustic vector dist below each stream dist.
    Input to stream dist should match input to tupleMap; in default
    tupleMap1 case this input should be ((label, subLabel), acInput).
    Output of tupleMap should be (subLabel, (phInput, acInput)).
    Input to acoustic vector dist should be acInput.
    Here acInput is arbitrary.
    Here phInput should be discrete and hashable but is otherwise arbitrary.

    The returned acc has the same tags and inputs for stream accs and
    acoustic vector accs.
    Below each stream acc an AutoGrowingDiscreteAcc is added with tag
    ('agAcc', streamName, subLabel) and input (phInput, acInput).

    If agTags is not None it should be a list of tags which are of the form
    ('agAcc', streamName, subLabel).
    In this case only the corresponding streams and subLabels have an
    AutoGrowingDiscreteAcc created.
    (This is mainly for debugging).
    """
    getAcInput = ElemGetter(1, 2)

    def globalToFullCtxCreateAccPartial(dist, createAccChild):
        if getElem(dist.tag, 0, 2) == 'stream':
            _, streamName = dist.tag
            acVecDist = nodetree.findTaggedNode(dist,
                                                lambda tag: tag == 'acVec')
            def createAcc():
                return d.getDefaultCreateAcc()(acVecDist)
            return d.MappedInputAcc(tupleMap,
                d.createDiscreteAcc(bmi.subLabels, lambda subLabel:
                    (
                        d.AutoGrowingDiscreteAcc(createAcc).withTag(
                            ('agAcc', streamName, subLabel)
                        )
                    ) if (
                        agTags is None or
                        ('agAcc', streamName, subLabel) in agTags
                    ) else (
                        d.MappedInputAcc(getAcInput,
                            createAcc()
                        )
                    )
                )
            ).withTag(('stream', streamName))

    return nodetree.getDagMap([globalToFullCtxCreateAccPartial,
                               d.defaultCreateAccPartial])(dist)

@codeDeps(ElemGetter, align.AlignmentToPhoneticSeq, align.StandardizeAlignment,
    createLinearGaussianVecDist, createLinearGaussianVectorDist,
    d.AutoregressiveSequenceDist, d.MappedInputDist, d.OracleDist,
    mgc_lf0_bap.computeFirstFrameAverage
)
def getInitDist1(bmi, corpus, alignmentSubLabels = 'sameAsBmi'):
    """Produces an initial dist.

    Expects input an alignment.
    Expects output a sequence of acoustic frames.
    Has stream tag ('stream', streamName) with input (phInput, acInput).
    Has acoustic vector tag 'acVec' with input acInput.
    Here phInput is typically a (label, subLabel) pair and acInput is typically
    a sequence of vectors.
    """
    initialFrame = mgc_lf0_bap.computeFirstFrameAverage(corpus, bmi.mgcOrder,
                                                        bmi.bapOrder)
    if alignmentSubLabels == 'sameAsBmi':
        alignmentSubLabels = bmi.subLabels
    alignmentToPhoneticSeq = align.AlignmentToPhoneticSeq(
        mapAlignment = align.StandardizeAlignment(
            corpus.subLabels, alignmentSubLabels
        )
    )
    createLeafDists = [
        (
            (lambda: createLinearGaussianVecDist(bmi.mgcOrder, bmi.mgcDepth))
            if bmi.mgcUseVec else
            (lambda: createLinearGaussianVectorDist(bmi.mgcSummarizer))
        ),
        d.OracleDist,
        d.OracleDist,
    ]
    getAcInput = ElemGetter(1, 2)
    dist = d.AutoregressiveSequenceDist(
        bmi.maxDepth,
        alignmentToPhoneticSeq,
        [ initialFrame for i in range(bmi.maxDepth) ],
        bmi.frameSummarizer.createDist(True, lambda streamIndex:
            # input is (phInput, acInput)
            d.MappedInputDist(getAcInput,
                createLeafDists[streamIndex]().withTag('acVec')
            ).withTag(('stream', corpus.streams[streamIndex].name))
        )
    )
    return dist

@codeDeps(ElemGetter, align.AlignmentToPhoneticSeqWithTiming,
    align.StandardizeAlignment, createLinearGaussianWithTimingVectorDist,
    d.AutoregressiveSequenceDist, d.MappedInputDist, d.OracleDist,
    mgc_lf0_bap.computeFirstFrameAverage, tupleMap0
)
def getInitDist2(bmi, corpus, alignmentSubLabels = 'sameAsBmi'):
    """Produces an initial dist which uses timings.

    Expects input an alignment.
    Expects output a sequence of acoustic frames.
    Has stream tag ('stream', streamName) with input
    (phInput, (timingInfo, acInput)) where timingInfo is
    (framesBefore, framesAfter).
    Has acoustic vector tag 'acVec' with input (timingInfo, acInput).
    Here phInput is typically a (label, subLabel) pair and acInput is typically
    a sequence of vectors.
    """
    initialFrame = mgc_lf0_bap.computeFirstFrameAverage(corpus, bmi.mgcOrder,
                                                        bmi.bapOrder)
    if alignmentSubLabels == 'sameAsBmi':
        alignmentSubLabels = bmi.subLabels
    alignmentToPhoneticSeq = align.AlignmentToPhoneticSeqWithTiming(
        mapAlignment = align.StandardizeAlignment(
            corpus.subLabels, alignmentSubLabels
        )
    )
    createLeafDists = [
        lambda: createLinearGaussianWithTimingVectorDist(bmi.mgcSummarizer),
        d.OracleDist,
        d.OracleDist,
    ]
    dropPhInput = ElemGetter(1, 2)
    dist = d.AutoregressiveSequenceDist(
        bmi.maxDepth,
        alignmentToPhoneticSeq,
        [ initialFrame for i in range(bmi.maxDepth) ],
        bmi.frameSummarizer.createDist(True, lambda streamIndex:
            # from ((phInput, timingInfo), acInput)
            #   to (phInput, (timingInfo, acInput))
            d.MappedInputDist(tupleMap0,
                d.MappedInputDist(dropPhInput,
                    createLeafDists[streamIndex]().withTag('acVec')
                ).withTag(('stream', corpus.streams[streamIndex].name))
            )
        )
    )
    return dist

@codeDeps(ElemGetter, align.AlignmentToPhoneticSeqWithTiming,
    align.StandardizeAlignment, createLinearGaussianVecDist,
    createLinearGaussianVectorDist, d.AutoregressiveSequenceDist,
    d.MappedInputDist, d.OracleDist, mgc_lf0_bap.computeFirstFrameAverage
)
def getInitDist3(bmi, corpus, alignmentSubLabels = 'sameAsBmi'):
    """Produces an initial dist.

    Expects input an alignment.
    Expects output a sequence of acoustic frames.
    Has stream tag ('stream', streamName) with input
    ((phInput, framesLeft), acInput).
    Has acoustic vector tag 'acVec' with input acInput.
    Here phInput is typically a (label, subLabel) pair, acInput is typically
    a sequence of vectors, and framesLeft is typically the number of frames
    remaining in the current subLabel.
    """
    initialFrame = mgc_lf0_bap.computeFirstFrameAverage(corpus, bmi.mgcOrder,
                                                        bmi.bapOrder)
    if alignmentSubLabels == 'sameAsBmi':
        alignmentSubLabels = bmi.subLabels
    alignmentToPhoneticSeq = align.AlignmentToPhoneticSeqWithTiming(
        mapAlignment = align.StandardizeAlignment(
            corpus.subLabels, alignmentSubLabels
        ),
        # (framesBefore, framesAfter) -> framesAfter
        mapTiming = ElemGetter(1, 2)
    )
    createLeafDists = [
        (
            (lambda: createLinearGaussianVecDist(bmi.mgcOrder, bmi.mgcDepth))
            if bmi.mgcUseVec else
            (lambda: createLinearGaussianVectorDist(bmi.mgcSummarizer))
        ),
        d.OracleDist,
        d.OracleDist,
    ]
    getAcInput = ElemGetter(1, 2)
    dist = d.AutoregressiveSequenceDist(
        bmi.maxDepth,
        alignmentToPhoneticSeq,
        [ initialFrame for i in range(bmi.maxDepth) ],
        bmi.frameSummarizer.createDist(True, lambda streamIndex:
            # input is ((phInput, framesLeft), acInput)
            d.MappedInputDist(getAcInput,
                createLeafDists[streamIndex]().withTag('acVec')
            ).withTag(('stream', corpus.streams[streamIndex].name))
        )
    )
    return dist

@codeDeps(ElemGetter, createCcBinaryDist, d.MappedInputDist)
def getInitBinaryDurDist1():
    """Produces an initial binary dist (typically used for duration modelling).

    Expects input (phInput, acInput).
    Expects output (0, 1).
    Has stream tag ('stream', 'dur') with input (phInput, acInput).
    Has acoustic vector tag 'acVec' with input acInput.
    Here phInput is typically a (label, subLabel) pair and acInput is typically
    a sequence of vectors.
    """
    getAcInput = ElemGetter(1, 2)
    return d.MappedInputDist(getAcInput,
        createCcBinaryDist(ccProbFloor = 3e-5).withTag('acVec')
    ).withTag(('stream', 'dur'))

@codeDeps(corpus_arctic.getCorpusSynthFewer, corpus_arctic.getTrainUttIds,
    labels_hts_demo.getParseLabel
)
def getCorpus():
    return corpus_arctic.getCorpusSynthFewer(
        trainUttIds = corpus_arctic.getTrainUttIds(),
        parseLabel = labels_hts_demo.getParseLabel(),
        subLabels = None,
        mgcOrder = 40,
        dataDir = '## TBA: fill-in data dir here ##',
        labDir = '## TBA: fill-in lab dir for phone-level alignments here ##',
        scriptsDir = 'scripts',
    )

@codeDeps(corpus_arctic.getCorpusSynthFewer, corpus_arctic.getTrainUttIds,
    labels_hts_demo.getParseLabel
)
def getCorpusWithSubLabels():
    return corpus_arctic.getCorpusSynthFewer(
        trainUttIds = corpus_arctic.getTrainUttIds(),
        parseLabel = labels_hts_demo.getParseLabel(),
        subLabels = list(range(5)),
        mgcOrder = 40,
        dataDir = '## TBA: fill-in data dir here ##',
        labDir = '## TBA: fill-in lab dir for state-level alignments here ##',
        scriptsDir = 'scripts',
    )

@codeDeps(mgc_lf0_bap.BasicArModelInfo, phoneset_cmu.CmuPhoneset)
def getBmiForCorpus(corpus, subLabels = 'sameAsCorpus'):
    if subLabels == 'sameAsCorpus':
        subLabels = corpus.subLabels
    return mgc_lf0_bap.BasicArModelInfo(
        phoneset = phoneset_cmu.CmuPhoneset(),
        subLabels = subLabels,
        mgcOrder = corpus.mgcOrder,
        bapOrder = corpus.bapOrder,
        mgcDepth = 3,
        lf0Depth = 2,
        bapDepth = 3,
        mgcIndices = [0],
        bapIndices = [],
        mgcUseVec = False,
        bapUseVec = False,
    )

@codeDeps()
def minusPrevAc((ph, ac)):
    return -ac[-1]

# (FIXME : this somewhat unnecessarily uses lots of memory)
@codeDeps(StandardizeAlignment, align.AlignmentToPhoneticSeq,
    d.AutoregressiveSequenceDist, d.DebugDist, d.MappedOutputDist, d.OracleDist,
    d.getDefaultCreateAcc, getBmiForCorpus, getCorpusWithSubLabels,
    mgc_lf0_bap.computeFirstFrameAverage, minusPrevAc, nodetree.findTaggedNode,
    printTime, questions_hts_demo.getTriphoneQuestionGroups, timed,
    xf.ShiftOutputTransform
)
def doDumpCorpus(outDir):
    print
    print 'DUMPING CORPUS'
    printTime('started dumpCorpus')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    questionGroups = questions_hts_demo.getTriphoneQuestionGroups(bmi.phoneset)

    def getQuestionAnswers(label):
        questionAnswers = []
        for labelValuer, questions in questionGroups:
            labelValue = labelValuer(label)
            for question in questions:
                questionAnswers.append(question(labelValue))
        return questionAnswers

    alignmentToPhoneticSeq = align.AlignmentToPhoneticSeq(
        mapAlignment = StandardizeAlignment(corpus.subLabels, bmi.subLabels),
        mapLabel = getQuestionAnswers
    )

    initialFrame = mgc_lf0_bap.computeFirstFrameAverage(corpus, bmi.mgcOrder,
                                                        bmi.bapOrder)

    # FIXME : only works if no cross-stream stuff happening. Make more robust somehow.
    shiftToPrevTransform = xf.ShiftOutputTransform(minusPrevAc)

    dist = d.AutoregressiveSequenceDist(
        bmi.maxDepth,
        alignmentToPhoneticSeq,
        [ initialFrame for i in range(bmi.maxDepth) ],
        bmi.frameSummarizer.createDist(True, lambda streamIndex:
            {
                0:
                    bmi.mgcSummarizer.createDist(True, lambda outIndex:
                        d.MappedOutputDist(shiftToPrevTransform,
                            d.DebugDist(None,
                                d.OracleDist()
                            ).withTag(('debug-mgc', outIndex))
                        )
                    )
            ,   1:
                    d.OracleDist()
            ,   2:
                    d.OracleDist()
            }[streamIndex]
        )
    )

    trainAcc = d.getDefaultCreateAcc()(dist)
    timed(corpus.accumulate)(trainAcc, uttIds = corpus.trainUttIds)

    testAcc = d.getDefaultCreateAcc()(dist)
    timed(corpus.accumulate)(testAcc, uttIds = corpus.testUttIds)

    for desc, acc in [('train', trainAcc), ('test', testAcc)]:
        for outIndex in bmi.mgcSummarizer.outIndices:
            debugAcc = nodetree.findTaggedNode(acc, lambda tag: tag == ('debug-mgc', outIndex))

            dumpInputFn = os.path.join(outDir, 'corpus-'+desc+'-in.mgc'+str(outIndex)+'.mat')
            with open(dumpInputFn, 'w') as f:
                print 'dump: dumping corpus input to:', dumpInputFn
                for (questionAnswers, subLabel), acInput in debugAcc.memo.inputs:
                    f.write(' '.join(
                        [ ('1' if questionAnswer else '0') for questionAnswer in questionAnswers ] +
                        [ str(subLabel) ]+
                        [ str(x) for x in acInput ]
                    )+'\n')

            dumpOutputFn = os.path.join(outDir, 'corpus-'+desc+'-out.mgc'+str(outIndex)+'.mat')
            with open(dumpOutputFn, 'w') as f:
                print 'dump: dumping corpus output to:', dumpOutputFn
                for acOutput in debugAcc.memo.outputs:
                    f.write(str(acOutput)+'\n')

    printTime('finished dumpCorpus')

@codeDeps(d.FloorSetter, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getInitDist1, mixup, printTime,
    trn.expectationMaximization
)
def doGlobalSystem(synthOutDir, figOutDir):
    print
    print 'TRAINING GLOBAL SYSTEM'
    printTime('started global')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist1(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'global')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'global.2mix')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'global.4mix')

    printTime('finished global')

@codeDeps(corpus_bisque.getUttIdChunkArts, d.FloorSetter, evaluateVarious,
    getBmiForCorpus, getCorpusWithSubLabels, getInitDist1, lift, liftLocal, lit,
    mixupJobSet, train_bisque.expectationMaximizationJobSet
)
def doGlobalSystemJobSet(synthOutDirArt, figOutDirArt):
    corpusArt = liftLocal(getCorpusWithSubLabels)()
    bmiArt = liftLocal(getBmiForCorpus)(corpusArt)

    uttIdChunkArts = corpus_bisque.getUttIdChunkArts(corpusArt,
                                                     numChunksLit = lit(2))

    distArt = lift(getInitDist1)(bmiArt, corpusArt)

    # train global dist while setting floors
    distArt = train_bisque.expectationMaximizationJobSet(
        distArt,
        corpusArt,
        uttIdChunkArts,
        afterAccArt = liftLocal(d.FloorSetter)(lgFloorMult = lit(1e-3)),
        verbosityArt = lit(2),
    )
    results1Art = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('global')
    )

    distArt = mixupJobSet(distArt, corpusArt, uttIdChunkArts)
    results2Art = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('global.2mix')
    )

    distArt = mixupJobSet(distArt, corpusArt, uttIdChunkArts)
    results4Art = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('global.4mix')
    )

    return results1Art, results2Art, results4Art

@codeDeps(d.FloorSetter, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getInitDist1, globalToMonophoneMap, mixup,
    printTime, reportLogLikeBreakdown, trn.expectationMaximization
)
def doMonophoneSystem(synthOutDir, figOutDir):
    print
    print 'TRAINING MONOPHONE SYSTEM'
    printTime('started mono')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist1(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    print 'DEBUG: converting global dist to monophone dist'
    dist = globalToMonophoneMap(dist, bmi)

    dist = trn.expectationMaximization(dist, corpus.accumulate,
                                       afterAcc = reportLogLikeBreakdown,
                                       verbosity = 2)[0]
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'mono')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'mono.2mix')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'mono.4mix')

    printTime('finished mono')

@codeDeps(corpus_bisque.getUttIdChunkArts, d.FloorSetter, evaluateVarious,
    getBmiForCorpus, getCorpusWithSubLabels, getInitDist1,
    getReportLogLikeBreakdown, globalToMonophoneMap, lift, liftLocal, lit,
    mixupJobSet, train_bisque.expectationMaximizationJobSet
)
def doMonophoneSystemJobSet(synthOutDirArt, figOutDirArt):
    corpusArt = liftLocal(getCorpusWithSubLabels)()
    bmiArt = liftLocal(getBmiForCorpus)(corpusArt)

    uttIdChunkArts = corpus_bisque.getUttIdChunkArts(corpusArt,
                                                     numChunksLit = lit(2))

    distArt = lift(getInitDist1)(bmiArt, corpusArt)

    # train global dist while setting floors
    distArt = train_bisque.expectationMaximizationJobSet(
        distArt,
        corpusArt,
        uttIdChunkArts,
        afterAccArt = liftLocal(d.FloorSetter)(lgFloorMult = lit(1e-3)),
        verbosityArt = lit(2),
    )

    distArt = lift(globalToMonophoneMap)(distArt, bmiArt)

    distArt = train_bisque.expectationMaximizationJobSet(
        distArt,
        corpusArt,
        uttIdChunkArts,
        afterAccArt = liftLocal(getReportLogLikeBreakdown)(),
        verbosityArt = lit(2),
    )
    results1Art = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('mono')
    )

    distArt = mixupJobSet(distArt, corpusArt, uttIdChunkArts)
    results2Art = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('mono.2mix')
    )

    distArt = mixupJobSet(distArt, corpusArt, uttIdChunkArts)
    results4Art = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('mono.4mix')
    )

    return results1Art, results2Art, results4Art

@codeDeps(d.FloorSetter, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getInitDist2, globalToMonophoneMap, printTime,
    reportLogLikeBreakdown, trn.expectationMaximization
)
def doTimingInfoSystem(synthOutDir, figOutDir):
    print
    print 'TRAINING MONOPHONE SYSTEM WITH TIMING INFO'
    printTime('started timingInfo')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist2(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    print 'DEBUG: converting global dist to monophone dist'
    dist = globalToMonophoneMap(dist, bmi)

    dist = trn.expectationMaximization(dist, corpus.accumulate,
                                       afterAcc = reportLogLikeBreakdown,
                                       verbosity = 2)[0]
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'timingInfo')

    printTime('finished timingInfo')

@codeDeps(ques.TupleIdLabelValuer, ques.getEqualityQuestions,
    ques.getThreshQuestions
)
def getEqualityThreshQGFramesLeft(sortedValues, threshes = None):
    values = sortedValues
    if threshes is None:
        threshes = sortedValues[1:]
    return (ques.TupleIdLabelValuer(1, shortRepr = 'frames_left'),
            (ques.getEqualityQuestions(values) +
             ques.getThreshQuestions(threshes)))

@codeDeps(cluster.ClusteringSpec, cluster.MdlUtilitySpec,
    cluster.decisionTreeCluster, d.AutoGrowingDiscreteAcc, d.FloorSetter,
    d.defaultEstimatePartial, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getInitDist1, globalToFullCtxCreateAcc, mixup,
    nodetree.findTaggedNode, nodetree.getDagMap, printTime,
    questions_hts_demo.getFullContextQuestionGroups, timed,
    trn.expectationMaximization
)
def doDecisionTreeClusteredSystem(synthOutDir, figOutDir, mdlFactor = 0.3):
    print
    print 'DECISION TREE CLUSTERING'
    printTime('started clustered')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist1(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    agTags = [
        ('agAcc', stream.name, subLabel)
        for stream in corpus.streams
        for subLabel in bmi.subLabels
    ]

    print 'DEBUG: converting global dist to full ctx acc'
    accOverall = globalToFullCtxCreateAcc(dist, bmi, agTags = agTags)

    print 'DEBUG: accumulating for decision tree clustering'
    timed(corpus.accumulate)(accOverall)

    questionGroups = questions_hts_demo.getFullContextQuestionGroups(bmi.phoneset)

    clusteringSpecDict = dict()
    for agTag in agTags:
        utilitySpec = cluster.MdlUtilitySpec(mdlFactor, minCount = 10.0)
        clusteringSpecDict[agTag] = cluster.ClusteringSpec(
            utilitySpec, questionGroups, verbosity = 3
        )

    subDistDict = dict()
    for agTag in agTags:
        clusteringSpec = clusteringSpecDict[agTag]
        agAcc = nodetree.findTaggedNode(accOverall, lambda tag: tag == agTag)
        print 'cluster: clustering for tag %s' % (agTag,)
        subDist = timed(cluster.decisionTreeCluster)(
            clusteringSpec, agAcc.accDict.keys(),
            lambda label: agAcc.accDict[label], agAcc.createAcc
        )
        subDistDict[agTag] = subDist

    def decisionTreeClusterEstimatePartial(acc, estimateChild):
        if isinstance(acc, d.AutoGrowingDiscreteAcc):
            agTag = acc.tag
            return subDistDict[agTag]
    decisionTreeClusterEstimate = nodetree.getDagMap([decisionTreeClusterEstimatePartial, d.defaultEstimatePartial])

    dist = decisionTreeClusterEstimate(accOverall)
    print
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'clustered')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'clustered.2mix')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'clustered.4mix')

    printTime('finished clustered')

@codeDeps(cluster.ClusteringSpec, cluster.MdlUtilitySpec,
    cluster.decisionTreeCluster, d.AutoGrowingDiscreteAcc, d.FloorSetter,
    d.defaultEstimatePartial, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getEqualityThreshQGFramesLeft, getInitDist3,
    globalToFullCtxCreateAcc, mixup, nodetree.findTaggedNode,
    nodetree.getDagMap, printTime, ques.AttrLabelValuer,
    ques.TupleAttrLabelValuer, questions_hts_demo.getFullContextQuestionGroups,
    timed, trn.expectationMaximization, tupleMap2
)
def doDecisionTreeClusteredFramesRemainingSystem(synthOutDir, figOutDir,
                                                 mdlFactor = 0.3,
                                                 questionMaxDur = 20):
    print
    print 'DECISION TREE CLUSTERING WITH FRAMES REMAINING INFO'
    printTime('started clustered_frames_remaining')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist3(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    labelQuestionGroups = questions_hts_demo.getFullContextQuestionGroups(
        bmi.phoneset
    )
    questionGroups = []
    for labelValuer, questions in labelQuestionGroups:
        assert isinstance(labelValuer, ques.AttrLabelValuer)
        labelValuerNew = ques.TupleAttrLabelValuer(
            0, labelValuer.labelKey, shortRepr = labelValuer.labelKey
        )
        questionGroups.append((labelValuerNew, questions))
    questionGroups.append(
        getEqualityThreshQGFramesLeft(range(0, questionMaxDur))
    )

    agTags = [
        ('agAcc', stream.name, subLabel)
        for stream in corpus.streams
        for subLabel in bmi.subLabels
    ]

    print 'DEBUG: converting global dist to full ctx acc'
    accOverall = globalToFullCtxCreateAcc(dist, bmi, agTags = agTags,
                                          tupleMap = tupleMap2)

    print 'DEBUG: accumulating for decision tree clustering'
    timed(corpus.accumulate)(accOverall)

    clusteringSpecDict = dict()
    for agTag in agTags:
        utilitySpec = cluster.MdlUtilitySpec(mdlFactor, minCount = 10.0)
        clusteringSpecDict[agTag] = cluster.ClusteringSpec(
            utilitySpec, questionGroups, verbosity = 3
        )

    subDistDict = dict()
    for agTag in agTags:
        clusteringSpec = clusteringSpecDict[agTag]
        agAcc = nodetree.findTaggedNode(accOverall, lambda tag: tag == agTag)
        print 'cluster: clustering for tag %s' % (agTag,)
        subDist = timed(cluster.decisionTreeCluster)(
            clusteringSpec, agAcc.accDict.keys(),
            lambda label: agAcc.accDict[label], agAcc.createAcc
        )
        subDistDict[agTag] = subDist

    def decisionTreeClusterEstimatePartial(acc, estimateChild):
        if isinstance(acc, d.AutoGrowingDiscreteAcc):
            agTag = acc.tag
            return subDistDict[agTag]
    decisionTreeClusterEstimate = nodetree.getDagMap([decisionTreeClusterEstimatePartial, d.defaultEstimatePartial])

    dist = decisionTreeClusterEstimate(accOverall)
    print
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'clustered_frames_remaining')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'clustered_frames_remaining.2mix')

    dist = mixup(dist, corpus.accumulate)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'clustered_frames_remaining.4mix')

    printTime('finished clustered_frames_remaining')

@codeDeps(cluster.ClusteringSpec, cluster.MdlUtilitySpec,
    cluster.decisionTreeClusterInGreedyOrderWithTest, d.FloorSetter,
    getBmiForCorpus, getCorpusWithSubLabels, getInitDist1,
    globalToFullCtxCreateAcc, nodetree.findTaggedNode, printTime,
    questions_hts_demo.getFullContextQuestionGroups, timed,
    trn.expectationMaximization
)
def doDecisionTreeClusteredInvestigateMdl(synthOutDir, figOutDir,
                                          mdlFactor = 0.3,
                                          streamIndicesToUse = [0],
                                          subLabelsToUse = None):
    print
    print 'DECISION TREE CLUSTERING INVESTIGATING MDL'
    printTime('started clustered_investigate_mdl')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist1(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    if subLabelsToUse is None:
        subLabelsToUse = bmi.subLabels
    agTags = [
        ('agAcc', corpus.streams[streamIndex].name, subLabel)
        for streamIndex in streamIndicesToUse
        for subLabel in subLabelsToUse
    ]

    print 'DEBUG: converting global dist to full ctx acc'
    accOverall = globalToFullCtxCreateAcc(dist, bmi, agTags = agTags)

    print 'DEBUG: accumulating for decision tree clustering'
    timed(corpus.accumulate)(accOverall)

    accTestOverall = globalToFullCtxCreateAcc(dist, bmi, agTags = agTags)
    timed(corpus.accumulate)(accTestOverall, uttIds = corpus.testUttIds)

    questionGroups = questions_hts_demo.getFullContextQuestionGroups(bmi.phoneset)

    clusteringSpecDict = dict()
    for agTag in agTags:
        growerSpec = cluster.MdlUtilitySpec(mdlFactor, minCount = 10.0)
        clusteringSpecDict[agTag] = cluster.ClusteringSpec(
            growerSpec, questionGroups, verbosity = 0
        )

    for agTag in agTags:
        clusteringSpec = clusteringSpecDict[agTag]
        agAcc = nodetree.findTaggedNode(accOverall, lambda tag: tag == agTag)
        agAccTest = nodetree.findTaggedNode(accTestOverall,
                                            lambda tag: tag == agTag)
        print
        print 'cluster: clustering for tag %s' % (agTag,)

        outMat = os.path.join(figOutDir, 'delta-%s-%s-s%s.mat' % agTag)
        print '(writing output to %s)' % outMat
        with open(outMat, 'w') as f:
            print ('(train frames = %s, test frames = %s)' %
                   (agAcc.occ, agAccTest.occ))
            f.write('# delta log probs due to splits\n')
            f.write('# format: [train delta] [test delta] (both per frame)\n')
            f.write('# frames: %s %s\n' % (agAcc.occ, agAccTest.occ))
            for (
                deltaNumLeaves, deltaTrain, deltaTest
            ) in cluster.decisionTreeClusterInGreedyOrderWithTest(
                clusteringSpec,
                agAcc.accDict.keys(),
                agAccTest.accDict.keys(),
                lambda label: agAcc.accDict[label],
                lambda label: agAccTest.accDict[label],
                agAcc.createAcc
            ):
                print ('delta: %s %s %s' %
                       (deltaNumLeaves,
                        deltaTrain / agAcc.occ,
                        deltaTest / agAccTest.occ))
                f.write('%s %s %s\n' %
                        (deltaNumLeaves,
                         deltaTrain / agAcc.occ,
                         deltaTest / agAccTest.occ))

    printTime('finished clustered_investigate_mdl')

@codeDeps(AttrGetter, ConstantFn, StandardizeAlignment,
    align.AlignmentToPhoneticSeq, convertToStudentResiduals,
    convertToTransformedGaussianResiduals, d.AutoregressiveSequenceDist,
    d.DebugDist, d.LinearGaussian, d.MappedInputDist, d.OracleDist,
    d.TransformedInputDist, d.TransformedOutputDist, d.createDiscreteDist,
    d.getByTagParamSpec, d.getDefaultParamSpec, draw.drawFor1DInput,
    draw.drawLogPdf, draw.drawWarping, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getElem, mgc_lf0_bap.computeFirstFrameAverage,
    nodetree.findTaggedNode, nodetree.findTaggedNodes, printTime, timed,
    trn.trainCG, trn.trainCGandEM, trn.trainEM, tupleMap1, xf.AddBias,
    xf.IdentityTransform, xf.MinusPrev, xf.ShiftOutputTransform,
    xf.SimpleOutputTransform, xf.SumTransform1D, xf.TanhTransform1D,
    xf.VectorizeTransform
)
def doTransformSystem(synthOutDir, figOutDir, globalPhone = True, studentResiduals = True, numTanhTransforms = 3):
    print
    print 'TRAINING TRANSFORM SYSTEM'
    printTime('started xf')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'globalPhone =', globalPhone
    print 'studentResiduals =', studentResiduals
    # mgcDepth affects what pictures we can draw
    print 'mgcDepth =', bmi.mgcDepth
    print 'numTanhTransforms =', numTanhTransforms

    # N.B. would be perverse to have globalPhone == True with numSubLabels != 1, but not disallowed
    phoneList = ['global'] if globalPhone else bmi.phoneset.phoneList

    print 'numSubLabels =', len(bmi.subLabels)

    alignmentToPhoneticSeq = align.AlignmentToPhoneticSeq(
        mapAlignment = StandardizeAlignment(corpus.subLabels, bmi.subLabels),
        mapLabel = (ConstantFn('global') if globalPhone
                    else AttrGetter('phone'))
    )

    initialFrame = mgc_lf0_bap.computeFirstFrameAverage(corpus, bmi.mgcOrder,
                                                        bmi.bapOrder)

    # FIXME : only works if no cross-stream stuff happening. Make more robust somehow.
    shiftToPrevTransform = xf.ShiftOutputTransform(xf.MinusPrev())

    mgcOutputTransform = dict()
    mgcInputTransform = dict()
    for outIndex in bmi.mgcSummarizer.outIndices:
        xmin, xmax = corpus.mgcLims[outIndex]
        bins = np.linspace(xmin, xmax, numTanhTransforms + 1)
        binCentres = bins[:-1] + 0.5 * np.diff(bins)
        width = (xmax - xmin) / numTanhTransforms / 2.0
        tanhOutputTransforms = [ xf.TanhTransform1D(np.array([0.0, width, binCentre])) for binCentre in binCentres ]
        outputWarp = xf.SumTransform1D([xf.IdentityTransform()] + tanhOutputTransforms).withTag(('mgcOutputWarp', outIndex))
        mgcOutputTransform[outIndex] = xf.SimpleOutputTransform(outputWarp, checkDerivPositive1D = True).withTag(('mgcOutputTransform', outIndex))
        tanhInputTransforms = [ xf.TanhTransform1D(np.array([0.0, width, binCentre])) for binCentre in binCentres ]
        inputWarp = xf.SumTransform1D([xf.IdentityTransform()] + tanhInputTransforms).withTag(('mgcInputWarp', outIndex))
        mgcInputTransform[outIndex] = xf.VectorizeTransform(inputWarp).withTag(('mgcInputTransform', outIndex))

    dist = d.AutoregressiveSequenceDist(
        bmi.maxDepth,
        alignmentToPhoneticSeq,
        [ initialFrame for i in range(bmi.maxDepth) ],
        bmi.frameSummarizer.createDist(True, lambda streamIndex:
            {
                0:
                    d.MappedInputDist(tupleMap1,
                        d.createDiscreteDist(bmi.subLabels, lambda subLabel:
                            d.createDiscreteDist(phoneList, lambda phone:
                                bmi.mgcSummarizer.createDist(False, lambda outIndex:
                                    d.DebugDist(None,
                                        d.TransformedOutputDist(mgcOutputTransform[outIndex],
                                            d.TransformedInputDist(mgcInputTransform[outIndex],
                                                #d.MappedOutputDist(shiftToPrevTransform,
                                                    d.DebugDist(None,
                                                        d.MappedInputDist(xf.AddBias(),
                                                            # arbitrary dist to get things rolling
                                                            d.LinearGaussian(np.zeros((bmi.mgcSummarizer.vectorLength(outIndex) + 1,)), 1.0, varianceFloor = 0.0)
                                                        )
                                                    ).withTag('debug-xfed')
                                                #)
                                            )
                                        )
                                    ).withTag(('debug-orig', phone, subLabel, streamIndex, outIndex))
                                )
                            )
                        )
                    )
            ,   1:
                    d.OracleDist()
            ,   2:
                    d.OracleDist()
            }[streamIndex]
        )
    )

    def drawVarious(dist, id, simpleResiduals = False, debugResiduals = False):
        assert not (simpleResiduals and debugResiduals)
        acc = d.getDefaultParamSpec().createAccG(dist)
        corpus.accumulate(acc)
        streamIndex = 0
        for outIndex in bmi.mgcSummarizer.outIndices:
            lims = corpus.mgcLims[outIndex]
            streamId = corpus.streams[streamIndex].name+str(outIndex)
            outputTransform = nodetree.findTaggedNode(dist, lambda tag: tag == ('mgcOutputTransform', outIndex))
            inputTransform = nodetree.findTaggedNode(dist, lambda tag: tag == ('mgcInputTransform', outIndex))
            # (FIXME : replace with looking up warp (=sub-transform) directly once tree structure for transforms is done)
            outputWarp = outputTransform.transform
            inputWarp = inputTransform.transform1D

            outPdf = os.path.join(figOutDir, 'warping-'+id+'-'+streamId+'.pdf')
            draw.drawWarping([outputWarp, inputWarp], outPdf = outPdf, xlims = lims, title = outPdf)

            for phone in phoneList:
                for subLabel in bmi.subLabels:
                    accOrig = nodetree.findTaggedNode(acc, lambda tag: tag == ('debug-orig', phone, subLabel, streamIndex, outIndex))
                    distOrig = nodetree.findTaggedNode(dist, lambda tag: tag == ('debug-orig', phone, subLabel, streamIndex, outIndex))

                    debugAcc = accOrig
                    subDist = distOrig.dist
                    if bmi.mgcDepth == 1:
                        outPdf = os.path.join(figOutDir, 'scatter-'+id+'-orig-'+str(phone)+'-state'+str(subLabel)+'-'+streamId+'.pdf')
                        draw.drawFor1DInput(debugAcc, subDist, outPdf = outPdf, xlims = lims, ylims = lims, title = outPdf)

                    debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-xfed')
                    subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-xfed').dist
                    if bmi.mgcDepth == 1:
                        outPdf = os.path.join(figOutDir, 'scatter-'+id+'-xfed-'+str(phone)+'-state'+str(subLabel)+'-'+streamId+'.pdf')
                        draw.drawFor1DInput(debugAcc, subDist, outPdf = outPdf, xlims = map(inputWarp, lims), ylims = map(outputWarp, lims), title = outPdf)

                    if simpleResiduals:
                        debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-xfed')
                        subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-xfed').dist
                        residuals = np.array([ subDist.dist.residual(xf.AddBias()(input), output) for input, output in zip(debugAcc.memo.inputs, debugAcc.memo.outputs) ])
                        f = lambda x: -0.5 * math.log(2.0 * math.pi) - 0.5 * x * x
                        outPdf = os.path.join(figOutDir, 'residualLogPdf-'+id+'-'+str(phone)+'-state'+str(subLabel)+'-'+streamId+'.pdf')
                        if len(residuals) > 0:
                            draw.drawLogPdf(residuals, bins = 20, fns = [f], outPdf = outPdf, title = outPdf)

                    if debugResiduals:
                        debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-residual')
                        subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-residual').dist
                        f = lambda output: subDist.logProb([], output)
                        outPdf = os.path.join(figOutDir, 'residualLogPdf-'+id+'-'+str(phone)+'-state'+str(subLabel)+'-'+streamId+'.pdf')
                        if len(debugAcc.memo.outputs) > 0:
                            draw.drawLogPdf(debugAcc.memo.outputs, bins = 20, fns = [f], outPdf = outPdf, title = outPdf)

                    # (FIXME : replace with looking up sub-transform directly once tree structure for transforms is done)
                    residualTransforms = nodetree.findTaggedNodes(distOrig, lambda tag: tag == 'residualTransform')
                    assert len(residualTransforms) <= 1
                    for residualTransform in residualTransforms:
                        outPdf = os.path.join(figOutDir, 'residualTransform-'+id+'-'+str(phone)+'-state'+str(subLabel)+'-'+streamId+'.pdf')
                        draw.drawWarping([residualTransform.transform], outPdf = outPdf, xlims = [-2.5, 2.5], title = outPdf)

    dist = timed(trn.trainEM)(dist, corpus.accumulate, minIterations = 2, maxIterations = 2)
    timed(drawVarious)(dist, id = 'xf_init', simpleResiduals = True)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'xf_init')

    print
    print 'ESTIMATING "GAUSSIANIZATION" TRANSFORMS'
    def afterEst(dist, it):
        #timed(drawVarious)(dist, id = 'xf-it'+str(it))
        pass
    # (FIXME : change mgcInputTransform to mgcInputWarp and mgcOutputTransform to mgcOutputWarp once tree structure for transforms is done)
    mgcWarpParamSpec = d.getByTagParamSpec(lambda tag: getElem(tag, 0, 2) == 'mgcInputTransform' or getElem(tag, 0, 2) == 'mgcOutputTransform')
    dist = timed(trn.trainCGandEM)(dist, corpus.accumulate, ps = mgcWarpParamSpec, iterations = 5, length = -25, afterEst = afterEst, verbosity = 2)
    timed(drawVarious)(dist, id = 'xf', simpleResiduals = True)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'xf')

    if studentResiduals:
        print
        print 'USING STUDENT RESIDUALS'
        dist = convertToStudentResiduals(dist, studentTag = 'student', debugTag = 'debug-residual', subtractMeanTag = 'subtractMean')
        residualParamSpec = d.getByTagParamSpec(lambda tag: tag == 'student')
        subtractMeanParamSpec = d.getByTagParamSpec(lambda tag: tag == 'subtractMean')
    else:
        print
        print 'USING TRANSFORMED-GAUSSIAN RESIDUALS'
        dist = convertToTransformedGaussianResiduals(dist, residualTransformTag = 'residualTransform', debugTag = 'debug-residual', subtractMeanTag = 'subtractMean')
        residualParamSpec = d.getByTagParamSpec(lambda tag: tag == 'residualTransform')
        subtractMeanParamSpec = d.getByTagParamSpec(lambda tag: tag == 'subtractMean')
    timed(drawVarious)(dist, id = 'xf.res_init', debugResiduals = True)
    dist = timed(trn.trainCG)(dist, corpus.accumulate, ps = residualParamSpec, length = -50, verbosity = 2)
    dist = timed(trn.trainCG)(dist, corpus.accumulate, ps = subtractMeanParamSpec, length = -50, verbosity = 2)
    timed(drawVarious)(dist, id = 'xf.res', debugResiduals = True)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'xf.res')

    print
    print 'ESTIMATING ALL PARAMETERS'
    dist = timed(trn.trainCG)(dist, corpus.accumulate, ps = d.getDefaultParamSpec(), length = -200, verbosity = 2)
    timed(drawVarious)(dist, id = 'xf.res.xf', debugResiduals = True)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'xf.res.xf')

    printTime('finished xf')

@codeDeps(wnet.FlatMappedNet, wnet.SequenceNet, wnet.probLeftToRightZeroNet)
class BinaryDurNetFor(object):
    def __init__(self, subLabels):
        self.subLabels = subLabels

    def __call__(self, alignment):
        labelSeq = [ label for startTime, endTime, label, subAlignment in alignment ]
        net = wnet.FlatMappedNet(
            lambda label: wnet.probLeftToRightZeroNet(
                [ (label, subLabel) for subLabel in self.subLabels ],
                [ [ ((label, subLabel), adv) for adv in [0, 1] ] for subLabel in self.subLabels ]
            ),
            wnet.SequenceNet(labelSeq, None)
        )
        return net

@codeDeps(BinaryDurNetFor, d.AutoregressiveNetDist,
    d.AutoregressiveSequenceDist
)
def seqDistToBinaryDurNetDistMap(seqDist, bmi, durDist, pruneSpec):
    """Converts an AutoregressiveSequenceDist to an AutoregressiveNetDist.

    acDist of new AutoregressiveNetDist is seqDist.dist.
    durDist should be a binary dist.
    """
    assert isinstance(seqDist, d.AutoregressiveSequenceDist)
    netFor = BinaryDurNetFor(bmi.subLabels)
    acDist = seqDist.dist
    return d.AutoregressiveNetDist(seqDist.depth, netFor,
                                   seqDist.fillFrames, durDist, acDist,
                                   pruneSpec)

@codeDeps(d.SimplePruneSpec, getInitBinaryDurDist1, globalToMonophoneMap,
    seqDistToBinaryDurNetDistMap
)
def globalSeqDistToMonoNetDistMap1(seqDist, bmi):
    """Converts global AutoregressiveSequenceDist to monophone AutoregressiveNetDist."""
    durDist = getInitBinaryDurDist1()
    pruneSpec = d.SimplePruneSpec(betaThresh = 500.0, logOccThresh = 20.0)
    dist = seqDistToBinaryDurNetDistMap(seqDist, bmi, durDist, pruneSpec)
    return globalToMonophoneMap(dist, bmi)

@codeDeps(d.SimplePruneSpec, getInitBinaryDurDist1, globalToMonophoneMap,
    seqDistToBinaryDurNetDistMap
)
def monoSeqDistToMonoNetDistMap1(seqDist, bmi):
    """Converts monophone AutoregressiveSequenceDist to monophone AutoregressiveNetDist."""
    durDist = globalToMonophoneMap(getInitBinaryDurDist1(), bmi)
    pruneSpec = d.SimplePruneSpec(betaThresh = 500.0, logOccThresh = 20.0)
    return seqDistToBinaryDurNetDistMap(seqDist, bmi, durDist, pruneSpec)

@codeDeps(d.FloorSetter, d.getVerboseNetCreateAcc, evaluateVarious,
    getBmiForCorpus, getCorpus, getInitDist1, globalSeqDistToMonoNetDistMap1,
    printTime, reportLogLikeBreakdown, timed, trn.expectationMaximization,
    trn.trainEM
)
def doFlatStartSystem(synthOutDir, figOutDir, numSubLabels = 5):
    print
    print 'TRAINING FLAT-START SYSTEM'
    printTime('started flatStart')

    corpus = getCorpus()
    bmi = getBmiForCorpus(corpus, subLabels = list(range(numSubLabels)))

    assert corpus.subLabels is None
    assert bmi.subLabels is not None
    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist1(bmi, corpus, alignmentSubLabels = None)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    print 'DEBUG: converting global dist to monophone net dist'
    dist = globalSeqDistToMonoNetDistMap1(dist, bmi)

    print 'DEBUG: estimating monophone net dist'
    dist = trn.trainEM(dist, timed(corpus.accumulate),
                       createAcc = d.getVerboseNetCreateAcc(),
                       deltaThresh = 1e-4,
                       minIterations = 4, maxIterations = 10,
                       afterAcc = reportLogLikeBreakdown,
                       verbosity = 2)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'flatStart.mono')

    printTime('finished flatStart')

@codeDeps(d.FloorSetter, d.getVerboseNetCreateAcc, evaluateVarious,
    getBmiForCorpus, getCorpusWithSubLabels, getInitDist1, globalToMonophoneMap,
    monoSeqDistToMonoNetDistMap1, printTime, reportLogLikeBreakdown, timed,
    trn.expectationMaximization, trn.trainEM
)
def doMonophoneNetSystem(synthOutDir, figOutDir):
    print
    print 'TRAINING MONOPHONE NET SYSTEM'
    printTime('started monoNet')

    corpus = getCorpusWithSubLabels()
    bmi = getBmiForCorpus(corpus)

    print 'numSubLabels =', len(bmi.subLabels)

    dist = getInitDist1(bmi, corpus)

    # train global dist while setting floors
    dist, _, _, _ = trn.expectationMaximization(
        dist,
        corpus.accumulate,
        afterAcc = d.FloorSetter(lgFloorMult = 1e-3),
        verbosity = 2,
    )

    print 'DEBUG: converting global dist to monophone dist'
    dist = globalToMonophoneMap(dist, bmi)

    dist = trn.expectationMaximization(dist, corpus.accumulate,
                                       afterAcc = reportLogLikeBreakdown,
                                       verbosity = 2)[0]
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'mono')

    print 'DEBUG: converting monophone seq dist to monophone net dist'
    dist = monoSeqDistToMonoNetDistMap1(dist, bmi)

    print 'DEBUG: estimating monophone net dist'
    dist = trn.trainEM(dist, timed(corpus.accumulate),
                       createAcc = d.getVerboseNetCreateAcc(),
                       deltaThresh = 1e-4,
                       minIterations = 4, maxIterations = 4,
                       afterAcc = reportLogLikeBreakdown,
                       verbosity = 2)
    results = evaluateVarious(dist, bmi, corpus, synthOutDir, figOutDir, exptTag = 'monoNet.mono')

    printTime('finished monoNet')

@codeDeps(corpus_bisque.getUttIdChunkArts, d.FloorSetter,
    d.getVerboseNetCreateAcc, evaluateVarious, getBmiForCorpus,
    getCorpusWithSubLabels, getInitDist1, getReportLogLikeBreakdown,
    globalToMonophoneMap, lift, liftLocal, lit, monoSeqDistToMonoNetDistMap1,
    train_bisque.expectationMaximizationJobSet, train_bisque.trainEMJobSet
)
def doMonophoneNetSystemJobSet(synthOutDirArt, figOutDirArt):
    corpusArt = liftLocal(getCorpusWithSubLabels)()
    bmiArt = liftLocal(getBmiForCorpus)(corpusArt)

    uttIdChunkArts = corpus_bisque.getUttIdChunkArts(corpusArt,
                                                     numChunksLit = lit(2))

    uttIdChunkArtsNet = corpus_bisque.getUttIdChunkArts(corpusArt,
                                                        numChunksLit = lit(10))

    distArt = lift(getInitDist1)(bmiArt, corpusArt)

    # train global dist while setting floors
    distArt = train_bisque.expectationMaximizationJobSet(
        distArt,
        corpusArt,
        uttIdChunkArts,
        afterAccArt = liftLocal(d.FloorSetter)(lgFloorMult = lit(1e-3)),
        verbosityArt = lit(2),
    )

    distArt = lift(globalToMonophoneMap)(distArt, bmiArt)

    distArt = train_bisque.expectationMaximizationJobSet(
        distArt,
        corpusArt,
        uttIdChunkArts,
        afterAccArt = liftLocal(getReportLogLikeBreakdown)(),
        verbosityArt = lit(2),
    )
    resultsSeqArt = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('mono')
    )

    distArt = lift(monoSeqDistToMonoNetDistMap1)(distArt, bmiArt)

    distArt = train_bisque.trainEMJobSet(
        distArt,
        corpusArt,
        uttIdChunkArtsNet,
        numIterationsLit = lit(4),
        createAccArt = liftLocal(d.getVerboseNetCreateAcc)(),
        afterAccArt = liftLocal(getReportLogLikeBreakdown)(),
        verbosityArt = lit(2)
    )
    resultsNetArt = lift(evaluateVarious)(
        distArt, bmiArt, corpusArt, synthOutDirArt, figOutDirArt,
        exptTag = lit('monoNet.mono')
    )

    return resultsSeqArt, resultsNetArt

@codeDeps(doDecisionTreeClusteredFramesRemainingSystem,
    doDecisionTreeClusteredInvestigateMdl, doDecisionTreeClusteredSystem,
    doDumpCorpus, doFlatStartSystem, doGlobalSystem, doMonophoneNetSystem,
    doMonophoneSystem, doTimingInfoSystem, doTransformSystem
)
def run(outDir):
    synthOutDir = os.path.join(outDir, 'synth')
    figOutDir = os.path.join(outDir, 'fig')
    os.makedirs(synthOutDir)
    os.makedirs(figOutDir)
    print 'CONFIG: outDir =', outDir

    doDumpCorpus(outDir)

    doGlobalSystem(synthOutDir, figOutDir)

    doMonophoneSystem(synthOutDir, figOutDir)

    doTimingInfoSystem(synthOutDir, figOutDir)

    doDecisionTreeClusteredSystem(synthOutDir, figOutDir)

    doDecisionTreeClusteredFramesRemainingSystem(synthOutDir, figOutDir)

    doDecisionTreeClusteredInvestigateMdl(synthOutDir, figOutDir,
                                          mdlFactor = 0.2)

    doTransformSystem(synthOutDir, figOutDir)

    doFlatStartSystem(synthOutDir, figOutDir)

    doMonophoneNetSystem(synthOutDir, figOutDir)
