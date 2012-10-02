"""Example experiments."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.modelling import nodetree
import armspeech.modelling.dist as d
import armspeech.modelling.train as trn
from armspeech.modelling import summarizer
import armspeech.modelling.transform as xf
from armspeech.modelling import cluster
from armspeech.modelling import wnet
from armspeech.speech.features import stdCepDist, stdCepDistIncZero
from armspeech.speech import draw
from armspeech.util.timing import timed, printTime

import phoneset_cmu
import questions_hts_demo
import corpus_arctic

import os
import math
import numpy as np
import armspeech.numpy_settings

def writeDistFile(distFile, dist):
    """Writes repr of a dist to a file.

    N.B. Just for logging (actual serialization and deserialization of dists
    should use pickle instead).
    """
    distString = repr(dist)
    with open(distFile, 'w') as f:
        f.write(distString)
        f.write('\n')
    print 'DEBUG: wrote dist to file', distFile

def getFirst(x, default = None):
    try:
        t = x[0]
    except TypeError, AttributeError:
        return default
    else:
        return t

def getStandardizeAlignment(subLabels):
    numSubLabels = len(subLabels)
    haveWarned = [False]
    def standardizeAlignment(alignment):
        """Returns a standardized, state-level alignment.

        If numSubLabels is 1, outputs 1-state state-level alignment (whether
        input alignment is state-level or phone-level). If numSubLabels is not
        1 and input alignment is phone-level, then uses uniform segmentation to
        obtain a crudge state-level alignment with the desired number of
        states. If numSubLabels is not 1 and input alignment is state-level,
        output is just the input alignment, after checking that this has the
        desired number of states.
        """
        alignmentOut = []
        for phoneStartTime, phoneEndTime, label, subAlignment in alignment:
            if numSubLabels == 1:
                alignmentOut.append((phoneStartTime, phoneEndTime, label, [(phoneStartTime, phoneEndTime, 0, None)]))
            else:
                if subAlignment is None:
                    if not haveWarned[0]:
                        print 'NOTE: only phone-level alignment specified, so will use uniform segmentation to obtain a state-level alignment (not ideal)'
                        haveWarned[0] = True
                    phoneDur = (phoneEndTime - phoneStartTime) * 1.0
                    subAlignmentOut = []
                    for subLabelIndex, subLabel in enumerate(subLabels):
                        startTime = int(phoneDur * subLabelIndex / numSubLabels + 0.5) + phoneStartTime
                        endTime = int(phoneDur * (subLabelIndex + 1) / numSubLabels + 0.5) + phoneStartTime
                        subAlignmentOut.append((startTime, endTime, subLabel, None))
                    alignmentOut.append((phoneStartTime, phoneEndTime, label, subAlignmentOut))
                    assert endTime == phoneEndTime
                else:
                    subLabelsFromAlignment = [ subLabel for _, _, subLabel, _ in subAlignment ]
                    if subLabelsFromAlignment != subLabels:
                        raise RuntimeError('mismatched subLabels ('+repr(subLabels)+' desired, '+repr(subLabelsFromAlignment)+' actual)')
                    alignmentOut.append((phoneStartTime, phoneEndTime, label, subAlignment))
        return alignmentOut
    return standardizeAlignment

def reportFloored(distRoot, rootTag):
    dists = d.distNodeList(distRoot)
    taggedDistTypes = [('LG', d.LinearGaussian), ('CC', d.ConstantClassifier), ('BLC', d.BinaryLogisticClassifier)]
    numFlooreds = [ np.array([0, 0]) for distTypeIndex, (distTypeTag, distType) in enumerate(taggedDistTypes) ]
    for dist in dists:
        for distTypeIndex, (distTypeTag, distType) in enumerate(taggedDistTypes):
            if isinstance(dist, distType):
                numFlooreds[distTypeIndex] += dist.flooredSingle()
    summary = ', '.join([ distTypeTag+(': %s of %s' % tuple(numFlooreds[distTypeIndex])) for distTypeIndex, (distTypeTag, distType) in enumerate(taggedDistTypes) if numFlooreds[distTypeIndex][1] > 0 ])
    if summary:
        print 'flooring: %s: %s' % (rootTag, summary)

def run(dataDir, labDir, scriptsDir, outDir):
    synthOutDir = os.path.join(outDir, 'synth')
    distOutDir = os.path.join(outDir, 'dist')
    figOutDir = os.path.join(outDir, 'fig')
    os.makedirs(synthOutDir)
    os.makedirs(distOutDir)
    os.makedirs(figOutDir)
    print 'CONFIG: outDir =', outDir

    phoneset = phoneset_cmu.CmuPhoneset()
    corpus = corpus_arctic.getCorpusSynthFewer(trainUttIds = corpus_arctic.trainUttIds, mgcOrder = 40, dataDir = dataDir, labDir = labDir, scriptsDir = scriptsDir)

    mgcStream, lf0Stream, bapStream = corpus.streams
    mgcStreamDepth, lf0StreamDepth, bapStreamDepth = 3, 2, 3
    maxDepth = max(mgcStreamDepth, lf0StreamDepth, bapStreamDepth)
    streamDepths = {0: mgcStreamDepth, 1: lf0StreamDepth, 2: bapStreamDepth }
    frameSummarizer = summarizer.VectorSeqSummarizer(order = len(corpus.streams), depths = streamDepths)

    mgcSummarizer = summarizer.IndexSpecSummarizer([0], fromOffset = 0, toOffset = 0, order = mgcStream.order, depth = mgcStreamDepth)
    bapSummarizer = summarizer.IndexSpecSummarizer([], fromOffset = 0, toOffset = 0, order = bapStream.order, depth = bapStreamDepth)

    zeroFrame = np.zeros((mgcStream.order,)), None, np.zeros((bapStream.order,))
    def computeFirstFrameAverages():
        mgcFirstFrameAverage = np.zeros((mgcStream.order,))
        lf0FirstFrameProportionUnvoiced = 0.0
        bapFirstFrameAverage = np.zeros((bapStream.order,))
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
        return mgcFirstFrameAverage, lf0FirstFrameProportionUnvoiced, bapFirstFrameAverage
    mgcFirstFrameAverage, lf0FirstFrameProportionUnvoiced, bapFirstFrameAverage = computeFirstFrameAverages()
    # not crucial that this is true
    # (FIXME : perhaps shouldn't be an assert)
    assert lf0FirstFrameProportionUnvoiced >= 0.5
    firstFrameAverage = mgcFirstFrameAverage, None, bapFirstFrameAverage

    def computeFrameMeanAndVariance():
        mgcSum = np.zeros((mgcStream.order,))
        mgcSumSqr = np.zeros((mgcStream.order,))
        bapSum = np.zeros((bapStream.order,))
        bapSumSqr = np.zeros((bapStream.order,))
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


    def reportTrainAux((trainAux, trainAuxRat), trainFrames):
        print 'training aux = %s (%s) (%s frames)' % (trainAux / trainFrames, d.Rat.toString(trainAuxRat), trainFrames)

    def evaluateLogProb(dist, corpus):
        trainLogProb = corpus.logProb(dist, corpus.trainUttIds)
        trainFrames = corpus.frames(corpus.trainUttIds)
        print 'train set log prob = %s (%s frames)' % (trainLogProb / trainFrames, trainFrames)
        testLogProb = corpus.logProb(dist, corpus.testUttIds)
        testFrames = corpus.frames(corpus.testUttIds)
        print 'test set log prob = %s (%s frames)' % (testLogProb / testFrames, testFrames)
        print

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

    def evaluateSynthesize(dist, corpus, exptTag, afterSynth = None):
        corpus.synthComplete(dist, corpus.synthUttIds, d.SynthMethod.Sample, synthOutDir, exptTag+'.sample', afterSynth = afterSynth)
        corpus.synthComplete(dist, corpus.synthUttIds, d.SynthMethod.Meanish, synthOutDir, exptTag+'.meanish', afterSynth = afterSynth)

    def mixup(dist, accumulate):
        acc = d.getDefaultCreateAcc()(dist)
        accumulate(acc)
        logLikeInit = acc.logLike()
        framesInit = acc.count()
        print 'initial training log likelihood = %s (%s frames)' % (logLikeInit / framesInit, framesInit)
        dist = nodetree.getDagMap([trn.mixupLinearGaussianEstimatePartial, d.defaultEstimatePartial])(acc)
        dist = trn.trainEM(dist, accumulate, deltaThresh = 1e-4, minIterations = 4, maxIterations = 8, verbosity = 2)
        return dist

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

    def reportFlooredPerStream(dist):
        for stream in corpus.streams:
            distRoot = nodetree.findTaggedNode(dist, lambda tag: tag == ('stream', stream.name))
            reportFloored(distRoot, rootTag = stream.name)

    def getDrawMgc(ylims = None, includeGivenLabels = True, extraLabelSeqs = []):
        streamIndex = 0
        def drawMgc(synthOutput, uttId, exptTag):
            (uttId, alignment), trueOutput = corpus.data(uttId)

            alignmentToDraw = [ (start * corpus.framePeriod, end * corpus.framePeriod, label.phone) for start, end, label, subAlignment in alignment ]
            partitionedLabelSeqs = (draw.partitionSeq(alignmentToDraw, 2) if includeGivenLabels else []) + [ labelSeqSub for labelSeq in extraLabelSeqs for labelSeqSub in draw.partitionSeq(labelSeq, 2) ]

            trueSeqTime = (np.array(range(len(trueOutput))) + 0.5) * corpus.framePeriod
            synthSeqTime = (np.array(range(len(synthOutput))) + 0.5) * corpus.framePeriod

            for mgcIndex in mgcSummarizer.outIndices:
                trueSeq = [ frame[streamIndex][mgcIndex] for frame in trueOutput ]
                synthSeq = [ frame[streamIndex][mgcIndex] for frame in synthOutput ]

                outPdf = os.path.join(figOutDir, uttId+'-mgc'+str(mgcIndex)+'-'+exptTag+'.pdf')
                draw.drawLabelledSeq([(trueSeqTime, trueSeq), (synthSeqTime, synthSeq)], partitionedLabelSeqs, outPdf = outPdf, figSizeRate = 10.0, ylims = ylims, colors = ['red', 'purple'])
        return drawMgc

    # (FIXME : this somewhat unnecessarily uses lots of memory)
    def doDumpCorpus(numSubLabels = 1):
        print
        print 'DUMPING CORPUS'
        printTime('started dumpCorpus')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        standardizeAlignment = getStandardizeAlignment(subLabels)

        def alignmentToPhoneticSeq(alignment):
            for phoneStartTime, phoneEndTime, label, subAlignment in standardizeAlignment(alignment):
                questionAnswers = []
                for labelValuer, questions in questionGroups:
                    labelValue = labelValuer(label)
                    for question in questions:
                        questionAnswers.append(question(labelValue))
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        yield questionAnswers, subLabel

        questionGroups = questions_hts_demo.getTriphoneQuestionGroups(phoneset)

        # FIXME : only works if no cross-stream stuff happening. Make more robust somehow.
        shiftToPrevTransform = xf.ShiftOutputTransform(lambda x: -x[1][-1])

        dist = d.AutoregressiveSequenceDist(
            maxDepth,
            lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            [ firstFrameAverage for i in range(maxDepth) ],
            frameSummarizer.createDist(True, lambda streamIndex:
                {
                    0:
                        mgcSummarizer.createDist(True, lambda outIndex:
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

        def accumulate(acc, uttIds):
            for uttId in uttIds:
                input, output = corpus.data(uttId)
                acc.add(input, output)

        trainAcc = d.getDefaultCreateAcc()(dist)
        timed(accumulate)(trainAcc, corpus.trainUttIds)

        testAcc = d.getDefaultCreateAcc()(dist)
        timed(accumulate)(testAcc, corpus.testUttIds)

        for desc, acc in [('train', trainAcc), ('test', testAcc)]:
            for outIndex in mgcSummarizer.outIndices:
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

    def trainGlobalSystem(standardizeAlignment, lgVarianceFloorMult):
        def alignmentToPhoneticIter(alignment):
            for phoneStartTime, phoneEndTime, label, subAlignment in standardizeAlignment(alignment):
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        yield label, subLabel
        alignmentToPhoneticSeq = lambda alignment: list(alignmentToPhoneticIter(alignment))

        acc = d.AutoregressiveSequenceAcc(
            maxDepth,
            alignmentToPhoneticSeq,
            [ firstFrameAverage for i in range(maxDepth) ],
            frameSummarizer.createAcc(True, lambda streamIndex:
                {
                    0:
                        d.MappedInputAcc(lambda (phInput, acInput): acInput,
                            mgcSummarizer.createAcc(False, lambda outIndex:
                                d.MappedInputAcc(xf.AddBias(),
                                    d.LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + 1).withTag('setFloor')
                                )
                            )
                        ).withTag(('stream', corpus.streams[streamIndex].name))
                ,   1:
                        d.MappedInputAcc(lambda (phInput, acInput): acInput,
                            #d.MappedInputAcc(xf.Msd01ToVector(),
                            #    d.MappedInputAcc(xf.AddBias(),
                            #        d.IdentifiableMixtureAcc(
                            #            d.BinaryLogisticClassifierAcc(d.BinaryLogisticClassifier(coeff = np.zeros((2 * lf0StreamDepth + 1,)), coeffFloor = np.ones((2 * lf0StreamDepth + 1,)) * 5.0)),
                            #            [
                            #                d.FixedValueAcc(None),
                            #                d.LinearGaussianAcc(inputLength = 2 * lf0StreamDepth + 1).withTag('setFloor')
                            #            ]
                            #        )
                            #    )
                            #)
                            d.OracleAcc()
                        ).withTag(('stream', corpus.streams[streamIndex].name))
                ,   2:
                        d.MappedInputAcc(lambda (phInput, acInput): acInput,
                            d.OracleAcc()
                        ).withTag(('stream', corpus.streams[streamIndex].name))
                }[streamIndex]
            )
        )

        print 'DEBUG: estimating global dist'
        timed(corpus.accumulate)(acc)
        dist, (trainAux, trainAuxRat) = d.getDefaultEstimateTotAux()(acc)
        reportTrainAux((trainAux, trainAuxRat), acc.count())

        print 'lgVarianceFloorMult =', lgVarianceFloorMult

        def setFloorMapPartial(dist, mapChild):
            if isinstance(dist, d.LinearGaussian) and dist.tag == 'setFloor':
                return d.LinearGaussian(dist.coeff, dist.variance, dist.variance * lgVarianceFloorMult)

        print 'DEBUG: setting floors'
        dist = nodetree.getDagMap([setFloorMapPartial, nodetree.defaultMapPartial])(dist)

        return dist

    def doGlobalSystem(numSubLabels = 1):
        print
        print 'TRAINING GLOBAL SYSTEM'
        printTime('started global')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        lgVarianceFloorMult = 1e-3
        print 'lgVarianceFloorMult =', lgVarianceFloorMult

        dist = trainGlobalSystem(getStandardizeAlignment(subLabels), lgVarianceFloorMult)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'global.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'global', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'global.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'global.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'global.4mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'global.4mix', afterSynth = getDrawMgc())

        printTime('finished global')

    def doMonophoneSystem(numSubLabels = 1):
        print
        print 'TRAINING MONOPHONE SYSTEM'
        printTime('started mono')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        lgVarianceFloorMult = 1e-3
        print 'lgVarianceFloorMult =', lgVarianceFloorMult

        def globalToMonophoneMapPartial(dist, mapChild):
            if isinstance(dist, d.MappedInputDist) and getFirst(dist.tag) == 'stream':
                subDist = dist.dist
                return d.MappedInputDist(lambda ((label, subLabel), acInput): (subLabel, (label.phone, acInput)),
                    d.createDiscreteDist(subLabels, lambda subLabel:
                        d.createDiscreteDist(phoneset.phoneList, lambda phone:
                            d.isolateDist(subDist)
                        )
                    )
                ).withTag(dist.tag)

        globalDist = trainGlobalSystem(getStandardizeAlignment(subLabels), lgVarianceFloorMult)
        dist = globalDist

        print 'DEBUG: converting global dist to monophone dist'
        dist = nodetree.getDagMap([globalToMonophoneMapPartial, nodetree.defaultMapPartial])(dist)

        print 'DEBUG: estimating monophone dist'
        dist, trainLogLike, (trainAux, trainAuxRat), trainFrames = trn.expectationMaximization(dist, timed(corpus.accumulate), verbosity = 3)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'mono.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'mono', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'mono.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'mono.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'mono.4mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'mono.4mix', afterSynth = getDrawMgc())

        printTime('finished mono')

    def doTimingInfoSystem(numSubLabels = 1):
        print
        print 'TRAINING MONOPHONE SYSTEM WITH TIMING INFO'
        printTime('started timingInfo')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        standardizeAlignment = getStandardizeAlignment(subLabels)

        extraLength = 2

        def alignmentToPhoneticSeq(alignment):
            for startTime, endTime, label, subAlignment in standardizeAlignment(alignment):
                phone = label.phone
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        framesBefore = time - startTime
                        framesAfter = endTime - time - 1
                        yield phone, subLabel, [framesBefore, framesAfter]
        def convertTimingInfo(input):
            (phone, subLabel, extra), acousticContext = input
            assert len(extra) == extraLength
            return subLabel, (phone, (extra, acousticContext))

        acc = d.AutoregressiveSequenceAcc(
            maxDepth,
            lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            [ firstFrameAverage for i in range(maxDepth) ],
            frameSummarizer.createAcc(True, lambda streamIndex:
                {
                    0:
                        d.MappedInputAcc(convertTimingInfo,
                            d.createDiscreteAcc(subLabels, lambda subLabel:
                                d.createDiscreteAcc(phoneset.phoneList, lambda phone:
                                    mgcSummarizer.createAcc(True, lambda outIndex:
                                        d.MappedInputAcc(np.concatenate,
                                            d.MappedInputAcc(xf.AddBias(),
                                                d.LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + extraLength + 1, varianceFloor = 0.0)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                ,   1:
                        d.OracleAcc()
                ,   2:
                        d.OracleAcc()
                }[streamIndex]
            )
        )

        timed(corpus.accumulate)(acc)
        dist, (trainAux, trainAuxRat) = d.getDefaultEstimateTotAux()(acc)
        reportTrainAux((trainAux, trainAuxRat), acc.count())

        writeDistFile(os.path.join(distOutDir, 'timingInfo.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'timingInfo', afterSynth = getDrawMgc())

        printTime('finished timingInfo')

    def doDecisionTreeClusteredSystem(numSubLabels = 1, mdlFactor = 0.3):
        print
        print 'DECISION TREE CLUSTERING'
        printTime('started clustered')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        lgVarianceFloorMult = 1e-3
        print 'lgVarianceFloorMult =', lgVarianceFloorMult

        questionGroups = questions_hts_demo.getFullContextQuestionGroups(phoneset)

        def globalToFullCtxCreateAccPartial(dist, createAccChild):
            if isinstance(dist, d.MappedInputDist) and getFirst(dist.tag) == 'stream':
                rootDist = dist.dist
                def createAcc():
                    leafAcc = d.getDefaultCreateAcc()(rootDist)
                    return leafAcc
                return d.MappedInputAcc(lambda ((label, subLabel), acInput): (subLabel, (label, acInput)),
                    d.createDiscreteAcc(subLabels, lambda subLabel:
                        d.AutoGrowingDiscreteAcc(createAcc)
                    )
                ).withTag(dist.tag)

        globalDist = trainGlobalSystem(getStandardizeAlignment(subLabels), lgVarianceFloorMult)
        dist = globalDist

        print 'DEBUG: converting global dist to full ctx acc'
        acc = nodetree.getDagMap([globalToFullCtxCreateAccPartial, d.defaultCreateAccPartial])(dist)

        print 'DEBUG: accumulating for decision tree clustering'
        timed(corpus.accumulate)(acc)

        def decisionTreeClusterEstimatePartial(acc, estimateChild):
            if isinstance(acc, d.AutoGrowingDiscreteAcc):
                return timed(cluster.decisionTreeCluster)(acc.accDict.keys(), lambda label: acc.accDict[label], acc.createAcc, questionGroups, thresh = None, mdlFactor = mdlFactor, minCount = 10.0, maxCount = None, verbosity = 3)
        decisionTreeClusterEstimate = nodetree.getDagMap([decisionTreeClusterEstimatePartial, d.defaultEstimatePartial])

        dist = decisionTreeClusterEstimate(acc)
        print
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'clustered.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'clustered', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'clustered.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'clustered.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'clustered.4mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'clustered.4mix', afterSynth = getDrawMgc())

        printTime('finished clustered')

    def doTransformSystem(globalPhone = True, studentResiduals = True, numTanhTransforms = 3, numSubLabels = 1):
        print
        print 'TRAINING TRANSFORM SYSTEM'
        printTime('started xf')

        print 'globalPhone =', globalPhone
        print 'studentResiduals =', studentResiduals
        # mgcStreamDepth affects what pictures we can draw
        print 'mgcStreamDepth =', mgcStreamDepth
        print 'numTanhTransforms =', numTanhTransforms

        # N.B. would be perverse to have globalPhone == True with numSubLabels != 1, but not disallowed
        phoneList = ['global'] if globalPhone else phoneset.phoneList

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        standardizeAlignment = getStandardizeAlignment(subLabels)

        def alignmentToPhoneticSeq(alignment):
            for phoneStartTime, phoneEndTime, label, subAlignment in standardizeAlignment(alignment):
                phone = label.phone
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        yield 'global' if globalPhone else phone, subLabel

        # FIXME : only works if no cross-stream stuff happening. Make more robust somehow.
        shiftToPrevTransform = xf.ShiftOutputTransform(xf.MinusPrev())

        mgcOutputTransform = dict()
        mgcInputTransform = dict()
        for outIndex in mgcSummarizer.outIndices:
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
            maxDepth,
            lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            [ firstFrameAverage for i in range(maxDepth) ],
            frameSummarizer.createDist(True, lambda streamIndex:
                {
                    0:
                        d.MappedInputDist(lambda ((label, subLabel), acInput): (subLabel, (label, acInput)),
                            d.createDiscreteDist(subLabels, lambda subLabel:
                                d.createDiscreteDist(phoneList, lambda phone:
                                    mgcSummarizer.createDist(False, lambda outIndex:
                                        d.DebugDist(None,
                                            d.TransformedOutputDist(mgcOutputTransform[outIndex],
                                                d.TransformedInputDist(mgcInputTransform[outIndex],
                                                    #d.MappedOutputDist(shiftToPrevTransform,
                                                        d.DebugDist(None,
                                                            d.MappedInputDist(xf.AddBias(),
                                                                # arbitrary dist to get things rolling
                                                                d.LinearGaussian(np.zeros((mgcSummarizer.vectorLength(outIndex) + 1,)), 1.0, varianceFloor = 0.0)
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
            for outIndex in mgcSummarizer.outIndices:
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
                    for subLabel in subLabels:
                        accOrig = nodetree.findTaggedNode(acc, lambda tag: tag == ('debug-orig', phone, subLabel, streamIndex, outIndex))
                        distOrig = nodetree.findTaggedNode(dist, lambda tag: tag == ('debug-orig', phone, subLabel, streamIndex, outIndex))

                        debugAcc = accOrig
                        subDist = distOrig.dist
                        if mgcStreamDepth == 1:
                            outPdf = os.path.join(figOutDir, 'scatter-'+id+'-orig-'+str(phone)+'-state'+str(subLabel)+'-'+streamId+'.pdf')
                            draw.drawFor1DInput(debugAcc, subDist, outPdf = outPdf, xlims = lims, ylims = lims, title = outPdf)

                        debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-xfed')
                        subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-xfed').dist
                        if mgcStreamDepth == 1:
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
        writeDistFile(os.path.join(distOutDir, 'xf_init.dist'), dist)
        timed(drawVarious)(dist, id = 'xf_init', simpleResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf_init', afterSynth = getDrawMgc())

        print
        print 'ESTIMATING "GAUSSIANIZATION" TRANSFORMS'
        def afterEst(dist, it):
            #timed(drawVarious)(dist, id = 'xf-it'+str(it))
            pass
        # (FIXME : change mgcInputTransform to mgcInputWarp and mgcOutputTransform to mgcOutputWarp once tree structure for transforms is done)
        mgcWarpParamSpec = d.getByTagParamSpec(lambda tag: getFirst(tag) == 'mgcInputTransform' or getFirst(tag) == 'mgcOutputTransform')
        dist = timed(trn.trainCGandEM)(dist, corpus.accumulate, ps = mgcWarpParamSpec, iterations = 5, length = -25, afterEst = afterEst, verbosity = 2)
        writeDistFile(os.path.join(distOutDir, 'xf.dist'), dist)
        timed(drawVarious)(dist, id = 'xf', simpleResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf', afterSynth = getDrawMgc())

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
        writeDistFile(os.path.join(distOutDir, 'xf.res.dist'), dist)
        timed(drawVarious)(dist, id = 'xf.res', debugResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf.res', afterSynth = getDrawMgc())

        print
        print 'ESTIMATING ALL PARAMETERS'
        dist = timed(trn.trainCG)(dist, corpus.accumulate, ps = d.getDefaultParamSpec(), length = -200, verbosity = 2)
        writeDistFile(os.path.join(distOutDir, 'xf.res.xf.dist'), dist)
        timed(drawVarious)(dist, id = 'xf.res.xf', debugResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf.res.xf', afterSynth = getDrawMgc())

        printTime('finished xf')

    def doFlatStartSystem(numSubLabels = 5):
        print
        print 'TRAINING FLAT-START SYSTEM'
        printTime('started flatStart')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        lgVarianceFloorMult = 1e-3
        ccProbFloor = 3e-5
        print 'lgVarianceFloorMult =', lgVarianceFloorMult
        print 'ccProbFloor =', ccProbFloor

        globalDist = trainGlobalSystem(getStandardizeAlignment(subLabels), lgVarianceFloorMult)

        print 'DEBUG: converting global dist to monophone net dist'
        def netFor(alignment):
            labelSeq = [ label for startTime, endTime, label, subAlignment in alignment ]
            net = wnet.FlatMappedNet(
                lambda label: wnet.probLeftToRightZeroNet(
                    [ (label, subLabel) for subLabel in subLabels ],
                    [ [ ((label, subLabel), adv) for adv in [0, 1] ] for subLabel in subLabels ]
                ),
                wnet.SequenceNet(labelSeq, None)
            )
            return net
        def globalToMonophoneMapPartial(dist, mapChild):
            if isinstance(dist, d.MappedInputDist) and getFirst(dist.tag) == 'stream':
                subDist = dist.dist
                return d.MappedInputDist(lambda ((label, subLabel), acInput): (subLabel, (label.phone, acInput)),
                    d.createDiscreteDist(subLabels, lambda subLabel:
                        d.createDiscreteDist(phoneset.phoneList, lambda phone:
                            d.isolateDist(subDist)
                        )
                    )
                ).withTag(dist.tag)
        acDist = nodetree.getDagMap([globalToMonophoneMapPartial, nodetree.defaultMapPartial])(globalDist.dist)
        durDist = d.MappedInputDist(lambda ((label, subLabel), acInput): (subLabel, (label.phone, acInput)),
            d.createDiscreteDist(subLabels, lambda subLabel:
                d.createDiscreteDist(phoneset.phoneList, lambda phone:
                    d.ConstantClassifier(probs = np.array([0.5, 0.5]), probFloors = np.array([ccProbFloor, ccProbFloor]))
                )
            )
        ).withTag(('stream', 'dur'))
        pruneSpec = d.SimplePruneSpec(betaThresh = 500.0, logOccThresh = 20.0)
        dist = d.AutoregressiveNetDist(maxDepth, netFor, [ firstFrameAverage for i in range(maxDepth) ], durDist, acDist, pruneSpec)

        print 'DEBUG: estimating monophone net dist'
        dist = trn.trainEM(dist, timed(corpus.accumulate), deltaThresh = 1e-4, minIterations = 4, maxIterations = 10, verbosity = 2)
        reportFlooredPerStream(dist)
        writeDistFile(os.path.join(distOutDir, 'flatStart.mono.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'flatStart.mono', afterSynth = getDrawMgc())

        printTime('finished flatStart')

    doDumpCorpus()

    doGlobalSystem()

    doMonophoneSystem()

    doTimingInfoSystem()

    doDecisionTreeClusteredSystem()

    doTransformSystem()

    doFlatStartSystem()
