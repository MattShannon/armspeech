#!/usr/bin/python -u

"""Example experiments."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division
from __future__ import with_statement

from armspeech.modelling import nodetree
import armspeech.modelling.dist as d
import armspeech.modelling.train as trn
from armspeech.modelling import summarizer
import armspeech.modelling.transform as xf
from armspeech.modelling import cluster
from armspeech.speech.features import stdCepDist, stdCepDistIncZero
from armspeech.speech import draw
from armspeech.util.timing import timed, printTime

import phoneset_cmu
import questions_hts_demo
import corpus_arctic

import os
import sys
import argparse
import tempfile
import traceback
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')

def readDistFile(distFile):
    with open(distFile, 'r') as f:
        distString = f.read().strip()
    dist = eval(distString)
    print 'DEBUG: read dist from file', distFile
    return dist

# FIXME : roundtripping using writeDistFile (using readDistFile to load a distribution
#   file saved with writeDistFile) requires every distribution to output all relevant
#   details, including any functions / closures. This isn't currently the case.
#   Even if this was the case, might want to just use pickle instead -- less fragile,
#   doesn't lose precision outputting floats (though maybe this could be overcome by
#   using better routines than the numpy default ones) and doesn't waste space
#   representing multiple copies of shared immutable objects.
# FIXME : better serialization and deserialization of dists using pickle?
def writeDistFile(distFile, dist):
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
    haveWarned = False
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
                    if not haveWarned:
                        print 'NOTE: only phone-level alignment specified, so will use uniform segmentation to obtain a state-level alignment (not ideal)'
                        haveWarned = True
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

def main(rawArgs):
    parser = argparse.ArgumentParser(
        description = 'Runs example experiments that use the same data as the HTS demo.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataDir', dest = 'dataDir', default = 'expt_hts_demo/data', metavar = 'DDIR',
        help = 'base directory for speech parameter files (mgc should be in "mgc" subdirectory, etc)'
    )
    parser.add_argument(
        '--labDir', dest = 'labDir', default = 'expt_hts_demo/data/labels/full', metavar = 'LDIR',
        help = 'directory containing full-context label files'
    )
    parser.add_argument(
        '--scriptsDir', dest = 'scriptsDir', default = 'scripts', metavar = 'SDIR',
        help = 'directory containing scripts used for audio generation (HTS-demo-style Config.pm and gen_wave.pl)'
    )
    parser.add_argument(
        '--baseOutDir', dest = 'baseOutDir', default = 'expt_hts_demo', metavar = 'ODIR',
        help = 'base directory used for experiment output (unique subdirectory is autogenerated)'
    )
    args = parser.parse_args(rawArgs[1:])

    outDir = tempfile.mkdtemp(dir = args.baseOutDir, prefix = 'out.')
    synthOutDir = os.path.join(outDir, 'synth')
    distOutDir = os.path.join(outDir, 'dist')
    figOutDir = os.path.join(outDir, 'fig')
    os.makedirs(synthOutDir)
    os.makedirs(distOutDir)
    os.makedirs(figOutDir)
    print 'CONFIG: outDir =', outDir

    phoneset = phoneset_cmu
    corpus = corpus_arctic.getCorpusSynthFewer(trainUttIds = corpus_arctic.trainUttIds, mgcOrder = 40, dataDir = args.dataDir, labDir = args.labDir, scriptsDir = args.scriptsDir)

    mgcStream, lf0Stream, bapStream = corpus.streams
    mgcStreamDepth, lf0StreamDepth, bapStreamDepth = 3, 2, 3
    streamDepths = {0: mgcStreamDepth, 1: lf0StreamDepth, 2: bapStreamDepth }
    frameSummarizer = summarizer.VectorSeqSummarizer(order = len(corpus.streams), depths = streamDepths)

    mgcSummarizer = summarizer.IndexSpecSummarizer([0], fromOffset = 0, toOffset = 0, order = mgcStream.order, depth = mgcStreamDepth)
    bapSummarizer = summarizer.IndexSpecSummarizer([], fromOffset = 0, toOffset = 0, order = bapStream.order, depth = bapStreamDepth)


    def reportTrainLogLike(trainLogLike, trainOcc):
        print 'training log likelihood =', trainLogLike / trainOcc, '('+str(trainOcc)+' frames)'

    def evaluateLogProb(dist, corpus):
        trainLogProb, trainOcc = corpus.logProb_frames(dist, corpus.trainUttIds)
        print 'train set log prob =', trainLogProb / trainOcc, '('+str(trainOcc)+' frames)'
        testLogProb, testOcc = corpus.logProb_frames(dist, corpus.testUttIds)
        print 'test set log prob =', testLogProb / testOcc, '('+str(testOcc)+' frames)'
        print

    def evaluateMgcArOutError(dist, corpus, vecError = stdCepDist, desc = 'MARCD'):
        def frameToVec(frame):
            mgcFrame, lf0Frame, bapFrame = frame
            return mgcFrame
        trainError, trainOcc = corpus.arOutError_frames(dist, corpus.trainUttIds, vecError, frameToVec)
        print 'train set %s = %s (%s frames)' % (desc, trainError / trainOcc, trainOcc)
        testError, testOcc = corpus.arOutError_frames(dist, corpus.testUttIds, vecError, frameToVec)
        print 'test set %s = %s (%s frames)' % (desc, testError / testOcc, testOcc)
        print

    def evaluateMgcOutError(dist, corpus, vecError = stdCepDist, desc = 'MCD'):
        def frameToVec(frame):
            mgcFrame, lf0Frame, bapFrame = frame
            return mgcFrame
        trainError, trainOcc = corpus.outError_frames(dist, corpus.trainUttIds, vecError, frameToVec)
        print 'train set %s = %s (%s frames)' % (desc, trainError / trainOcc, trainOcc)
        testError, testOcc = corpus.outError_frames(dist, corpus.testUttIds, vecError, frameToVec)
        print 'test set %s = %s (%s frames)' % (desc, testError / testOcc, testOcc)
        print

    def evaluateSynthesize(dist, corpus, exptTag, afterSynth = None):
        corpus.synthComplete(dist, corpus.synthUttIds, d.SynthMethod.Sample, synthOutDir, exptTag+'.sample', afterSynth = afterSynth)
        corpus.synthComplete(dist, corpus.synthUttIds, d.SynthMethod.Meanish, synthOutDir, exptTag+'.meanish', afterSynth = afterSynth)

    def mixup(dist, accumulate):
        acc = d.defaultCreateAcc(dist)
        accumulate(acc)
        dist, trainLogLike, trainOcc = trn.mixupLinearGaussianEstimate(acc)
        reportTrainLogLike(trainLogLike, trainOcc)
        dist, trainLogLike, trainOcc = trn.trainEM(dist, accumulate, deltaThresh = 1e-4, minIterations = 4, maxIterations = 8, verbosity = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
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
                    d.LinearGaussian(np.array([]), dist.variance)
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

    def getDrawMgc(ylims = None, includeGivenLabels = True, extraLabelSeqs = []):
        streamIndex = 0
        def drawMgc(synthOutput, uttId, exptTag):
            alignment, trueOutput = corpus.data(uttId)

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

        questionGroups = questions_hts_demo.getTriphoneQuestionGroups()

        # FIXME : only works if no cross-stream stuff happening. Make more robust somehow.
        shiftToPrevTransform = xf.ShiftOutputTransform(lambda x: -x[1][-1])

        dist = d.MappedInputDist(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            d.AutoregressiveSequenceDist(10,
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
        )

        def accumulate(acc, uttIds):
            for uttId in uttIds:
                input, output = corpus.data(uttId)
                acc.add(input, output)

        trainAcc = d.defaultCreateAcc(dist)
        timed(accumulate)(trainAcc, corpus.trainUttIds)

        testAcc = d.defaultCreateAcc(dist)
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

    def doMonophoneSystem(numSubLabels = 1):
        print
        print 'TRAINING MONOPHONE SYSTEM'
        printTime('started mono')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        standardizeAlignment = getStandardizeAlignment(subLabels)

        def alignmentToPhoneticSeq(alignment):
            for phoneStartTime, phoneEndTime, label, subAlignment in standardizeAlignment(alignment):
                phone = label.phone
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        yield phone, subLabel

        acc = d.MappedInputAcc(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            d.AutoregressiveSequenceAcc(10,
                frameSummarizer.createAcc(True, lambda streamIndex:
                    {
                        0:
                            d.MappedInputAcc(lambda ((label, subLabel), acInput): (subLabel, (label, acInput)),
                                d.createDiscreteAcc(subLabels, lambda subLabel:
                                    d.createDiscreteAcc(phoneset.phoneList, lambda phone:
                                        mgcSummarizer.createAcc(False, lambda outIndex:
                                            d.MappedInputAcc(xf.AddBias(),
                                                d.LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + 1)
                                            )
                                        )
                                    )
                                )
                            )
                    ,   1:
                            d.MappedInputAcc(lambda ((label, subLabel), acInput): (subLabel, (label, acInput)),
                                d.createDiscreteAcc(subLabels, lambda subLabel:
                                    d.createDiscreteAcc(phoneset.phoneList, lambda phone:
                                        #d.MappedInputAcc(xf.Msd01ToVector(),
                                        #    d.MappedInputAcc(xf.AddBias(),
                                        #        d.IdentifiableMixtureAcc(
                                        #            d.BinaryLogisticClassifierAcc(d.BinaryLogisticClassifier(np.zeros([2 * lf0StreamDepth + 1]))),
                                        #            [
                                        #                d.FixedValueAcc(None),
                                        #                d.LinearGaussianAcc(inputLength = 2 * lf0StreamDepth + 1)
                                        #            ]
                                        #        )
                                        #    )
                                        #)
                                        d.OracleAcc()
                                    )
                                )
                            )
                    ,   2:
                            d.MappedInputAcc(lambda ((label, subLabel), acInput): (subLabel, (label, acInput)),
                                d.createDiscreteAcc(subLabels, lambda subLabel:
                                    d.createDiscreteAcc(phoneset.phoneList, lambda phone:
                                        d.OracleAcc()
                                    )
                                )
                            )
                    }[streamIndex]
                )
            )
        )

        timed(corpus.accumulate)(acc)

        dist, trainLogLike, trainOcc = timed(d.defaultEstimate)(acc)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'mono.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'mono', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'mono.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'mono.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'mono.4mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'mono.4mix', afterSynth = getDrawMgc())

        printTime('finished mono')

    def doTimingInfoSystem(numSubLabels = 1):
        print
        print 'TRAINING MONOPHONE SYSTEM WITH TIMING INFO'
        printTime('started timinginfo')

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

        acc = d.MappedInputAcc(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            d.AutoregressiveSequenceAcc(10,
                frameSummarizer.createAcc(True, lambda streamIndex:
                    {
                        0:
                            d.MappedInputAcc(convertTimingInfo,
                                d.createDiscreteAcc(subLabels, lambda subLabel:
                                    d.createDiscreteAcc(phoneset.phoneList, lambda phone:
                                        mgcSummarizer.createAcc(True, lambda outIndex:
                                            d.MappedInputAcc(np.concatenate,
                                                d.MappedInputAcc(xf.AddBias(),
                                                    d.LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + extraLength + 1)
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
        )

        timed(corpus.accumulate)(acc)

        dist, trainLogLike, trainOcc = timed(d.defaultEstimate)(acc)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'timinginfo.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'timinginfo', afterSynth = getDrawMgc())

        printTime('finished timinginfo')

    def doDecisionTreeClusteredSystem(numSubLabels = 1, mdlFactor = 0.3):
        print
        print 'DECISION TREE CLUSTERING'
        printTime('started clustered')

        print 'numSubLabels =', numSubLabels
        subLabels = list(range(numSubLabels))

        standardizeAlignment = getStandardizeAlignment(subLabels)

        def alignmentToPhoneticSeq(alignment):
            for phoneStartTime, phoneEndTime, label, subAlignment in standardizeAlignment(alignment):
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        yield label, subLabel

        questionGroups = questions_hts_demo.getFullContextQuestionGroups()

        acc = d.MappedInputAcc(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            d.AutoregressiveSequenceAcc(10,
                frameSummarizer.createAcc(True, lambda streamIndex:
                    {
                        0:
                            d.MappedInputAcc(lambda ((label, subLabel), acInput): (subLabel, (label, acInput)),
                                d.createDiscreteAcc(subLabels, lambda subLabel:
                                    d.AutoGrowingDiscreteAcc(lambda:
                                        mgcSummarizer.createAcc(False, lambda outIndex:
                                            d.MappedInputAcc(xf.AddBias(),
                                                d.LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + 1)
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
        )

        timed(corpus.accumulate)(acc)

        def decisionTreeClusterEstimatePartial(acc, estimateChild):
            if isinstance(acc, d.AutoGrowingDiscreteAcc):
                return timed(cluster.decisionTreeCluster)(acc.accDict.keys(), lambda label: acc.accDict[label], acc.createAcc, questionGroups, thresh = None, mdlFactor = 0.3, minOcc = 10.0, maxOcc = None, verbosity = 3)
        decisionTreeClusterEstimate = d.getEstimate([decisionTreeClusterEstimatePartial, d.defaultEstimatePartial])

        dist, trainLogLike, trainOcc = decisionTreeClusterEstimate(acc)
        print
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'clustered.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'clustered', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'clustered.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'clustered.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
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

        dist = d.MappedInputDist(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            d.AutoregressiveSequenceDist(10,
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
                                                                    d.LinearGaussian(np.zeros([mgcSummarizer.vectorLength(outIndex) + 1]), 1.0)
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
        )

        def drawVarious(dist, id, simpleResiduals = False, debugResiduals = False):
            assert not (simpleResiduals and debugResiduals)
            acc = d.defaultParamSpec.createAccG(dist)
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

        dist, trainLogLike, trainOcc = timed(trn.trainEM)(dist, corpus.accumulate, estimate = d.defaultEstimate, minIterations = 2, maxIterations = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'xf_init.dist'), dist)
        timed(drawVarious)(dist, id = 'xf_init', simpleResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf_init', afterSynth = getDrawMgc())

        print
        print 'ESTIMATING "GAUSSIANIZATION" TRANSFORMS'
        def afterEst(dist, logLike, occ, it):
            reportTrainLogLike(logLike, occ)
            #timed(drawVarious)(dist, id = 'xf-it'+str(it))
        # (FIXME : change mgcInputTransform to mgcInputWarp and mgcOutputTransform to mgcOutputWarp once tree structure for transforms is done)
        mgcWarpParamSpec = d.getByTagParamSpec(lambda tag: getFirst(tag) == 'mgcInputTransform' or getFirst(tag) == 'mgcOutputTransform')
        dist, trainLogLike, trainOcc = timed(trn.trainCGandEM)(dist, corpus.accumulate, ps = mgcWarpParamSpec, iterations = 5, length = -25, afterEst = afterEst, verbosity = 2)
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
        dist, trainLogLike, trainOcc = timed(trn.trainCG)(dist, corpus.accumulate, ps = residualParamSpec, length = -50, verbosity = 2)
        dist, trainLogLike, trainOcc = timed(trn.trainCG)(dist, corpus.accumulate, ps = subtractMeanParamSpec, length = -50, verbosity = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'xf.res.dist'), dist)
        timed(drawVarious)(dist, id = 'xf.res', debugResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf.res', afterSynth = getDrawMgc())

        print
        print 'ESTIMATING ALL PARAMETERS'
        dist, trainLogLike, trainOcc = timed(trn.trainCG)(dist, corpus.accumulate, ps = d.defaultParamSpec, length = -200, verbosity = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'xf.res.xf.dist'), dist)
        timed(drawVarious)(dist, id = 'xf.res.xf', debugResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateMgcOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateMgcArOutError(dist, corpus, vecError = stdCepDistIncZero)
        evaluateSynthesize(dist, corpus, 'xf.res.xf', afterSynth = getDrawMgc())

        printTime('finished xf')

    try:
        doDumpCorpus()

        doMonophoneSystem()

        doTimingInfoSystem()

        doDecisionTreeClusteredSystem()

        doTransformSystem()
    except:
        traceback.print_exc()
        print
        print '(to delete dir:)'
        print 'rm -r', outDir
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)
