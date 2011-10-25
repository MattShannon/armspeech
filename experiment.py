#!/usr/bin/python -u

"""Example experiments."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division
from __future__ import with_statement

import nodetree
from dist import *
from model import *
from summarizer import *
from transform import *
import draw
from timing import timed, printTime

import phoneset_cmu
import questions_htsDemo
import corpus_arctic

import getopt
import os
import sys
import tempfile
import traceback
from numpy import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readDistFile(distFile):
    with open(distFile, 'r') as f:
        distString = f.read().strip()
    dist = eval(distString)
    print 'DEBUG: read dist from file', distFile
    return dist

# FIXME : roundtripping using writeDistFile (using readDistFile to load a distribution
#   file saved with writeDistFile) requires every distribution to output all relevant
#   details, including any functions / closures.  This isn't currently the case.
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

def main(rawArgs):
    optList, args = getopt.gnu_getopt(rawArgs[1:], '', ['dataDir=', 'labDir=', 'scriptsDir=', 'outDir='])
    opts = dict(optList)

    if len(args) != 0:
        raise RuntimeError('this program takes no non-optional arguments')

    dataDir = opts.get('--dataDir', 'data')
    labDir = opts.get('--labDir', 'data/labels/full')
    scriptsDir = opts.get('--scriptsDir', 'scripts')

    outDir = opts.get('--outDir', tempfile.mkdtemp(dir = '.', prefix = 'out.'))
    synthOutDir = os.path.join(outDir, 'synth')
    distOutDir = os.path.join(outDir, 'dist')
    figOutDir = os.path.join(outDir, 'fig')
    # FIXME : cope with these dirs already existing
    os.makedirs(synthOutDir)
    os.makedirs(distOutDir)
    os.makedirs(figOutDir)
    print 'CONFIG: outDir =', outDir

    phoneset = phoneset_cmu
    corpus = corpus_arctic.getCorpusSynthFewer(trainUttIds = corpus_arctic.trainUttIds, mgcOrder = 40, dataDir = dataDir, labDir = labDir, scriptsDir = scriptsDir)

    mgcStream, lf0Stream, bapStream = corpus.streams
    mgcStreamDepth, lf0StreamDepth, bapStreamDepth = 3, 2, 3
    streamDepths = {0: mgcStreamDepth, 1: lf0StreamDepth, 2: bapStreamDepth }
    frameSummarizer = VectorSeqSummarizer(order = len(corpus.streams), depths = streamDepths)

    mgcSummarizer = IndexSpecSummarizer([0], fromOffset = 0, toOffset = 0, order = mgcStream.order, depth = mgcStreamDepth)
    bapSummarizer = IndexSpecSummarizer([], fromOffset = 0, toOffset = 0, order = bapStream.order, depth = bapStreamDepth)


    def reportTrainLogLike(trainLogLike, trainOcc):
        print 'training log likelihood =', trainLogLike / trainOcc, '('+str(trainOcc)+' frames)'

    def evaluateLogProb(dist, corpus):
        trainLogProb, trainOcc = corpus.logProb_frames(dist, corpus.trainUttIds)
        print 'train set log prob =', trainLogProb / trainOcc, '('+str(trainOcc)+' frames)'
        testLogProb, testOcc = corpus.logProb_frames(dist, corpus.testUttIds)
        print 'test set log prob =', testLogProb / testOcc, '('+str(testOcc)+' frames)'
        print

    def evaluateSynthesize(dist, corpus, exptTag, afterSynth = None):
        corpus.synthComplete(dist, corpus.synthUttIds, SynthMethod.Sample, synthOutDir, exptTag+'.sample', afterSynth = afterSynth)
        corpus.synthComplete(dist, corpus.synthUttIds, SynthMethod.Meanish, synthOutDir, exptTag+'.meanish', afterSynth = afterSynth)

    def mixup(dist, accumulate):
        acc = defaultCreateAcc(dist)
        accumulate(acc)
        dist, trainLogLike, trainOcc = mixupLinearGaussianEstimate(acc)
        reportTrainLogLike(trainLogLike, trainOcc)
        dist, trainLogLike, trainOcc = trainEM(dist, accumulate, deltaThresh = 1e-4, minIterations = 4, maxIterations = 8, verbosity = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        return dist

    def convertToStudentResiduals(dist, studentTag = None, debugTag = None, subtractMeanTag = None):
        def studentResidualsMapPartial(dist, mapChild):
            if isinstance(dist, LinearGaussian):
                subDist = StudentDist(df = 10000.0, precision = 1.0 / dist.variance).withTag(studentTag)
                if debugTag != None:
                    subDist = DebugDist(None, subDist).withTag(debugTag)
                subtractMeanTransform = ShiftOutputTransform(DotProductTransform(-dist.coeff)).withTag(subtractMeanTag)
                distNew = TransformedOutputDist(subtractMeanTransform,
                    MappedInputDist(ConstantTransform(array([])),
                        subDist
                    )
                )
                return distNew
        studentResidualsMap = nodetree.getDagMap([studentResidualsMapPartial, nodetree.defaultMapPartial])

        return studentResidualsMap(dist)

    def convertToTransformedGaussianResiduals(dist, residualTransformTag = None, debugTag = None, subtractMeanTag = None):
        def transformedGaussianResidualsMapPartial(dist, mapChild):
            if isinstance(dist, LinearGaussian):
                residualTransform = SimpleOutputTransform(SumTransform1D([
                    IdentityTransform()
                ,   TanhTransform1D(array([0.0, 0.5, -1.0]))
                ,   TanhTransform1D(array([0.0, 0.5, 0.0]))
                ,   TanhTransform1D(array([0.0, 0.5, 1.0]))
                ]), checkDerivPositive1D = True).withTag(residualTransformTag)

                subDist = TransformedOutputDist(residualTransform,
                    LinearGaussian(array([]), dist.variance)
                )
                if debugTag != None:
                    subDist = DebugDist(None, subDist).withTag(debugTag)
                subtractMeanTransform = ShiftOutputTransform(DotProductTransform(-dist.coeff)).withTag(subtractMeanTag)
                distNew = TransformedOutputDist(subtractMeanTransform,
                    MappedInputDist(ConstantTransform(array([])),
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

            alignmentToDraw = [ (start * corpus.framePeriod, end * corpus.framePeriod, label.phone) for start, end, label in alignment ]
            partitionedLabelSeqs = (draw.partitionSeq(alignmentToDraw, 2) if includeGivenLabels else []) + [ labelSeqSub for labelSeq in extraLabelSeqs for labelSeqSub in draw.partitionSeq(labelSeq, 2) ]

            trueSeqTime = (array(range(len(trueOutput))) + 0.5) * corpus.framePeriod
            synthSeqTime = (array(range(len(synthOutput))) + 0.5) * corpus.framePeriod

            for mgcIndex in mgcSummarizer.outIndices:
                trueSeq = [ frame[streamIndex][mgcIndex] for frame in trueOutput ]
                synthSeq = [ frame[streamIndex][mgcIndex] for frame in synthOutput ]

                outPdf = os.path.join(figOutDir, uttId+'-mgc'+str(mgcIndex)+'-'+exptTag+'.pdf')
                draw.drawLabelledSeq([(trueSeqTime, trueSeq), (synthSeqTime, synthSeq)], partitionedLabelSeqs, outPdf = outPdf, figSizeRate = 10.0, ylims = ylims, colors = ['red', 'purple'])
        return drawMgc

    def doMonophoneSystem():
        print
        print 'TRAINING MONOPHONE SYSTEM'
        printTime('started mono')

        def alignmentToPhoneticSeq(alignment):
            for startTime, endTime, label in alignment:
                phone = label.phone
                for time in range(startTime, endTime):
                    yield phone

        acc = MappedInputAcc(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            SequenceAcc(10,
                createDiscreteAcc(phoneset.phoneList, lambda phone:
                    frameSummarizer.createAcc(False, lambda streamIndex:
                        {
                            0:
                                mgcSummarizer.createAcc(False, lambda outIndex:
                                    MappedInputAcc(AddBias(),
                                        LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + 1)
                                    )
                                )
                        ,   1:
                                #MappedInputAcc(Msd01ToVector(),
                                #    MappedInputAcc(AddBias(),
                                #        IdentifiableMixtureAcc(
                                #            BinaryLogisticClassifierAcc(BinaryLogisticClassifier(zeros([ 2 * lf0StreamDepth + 1 ]))),
                                #            [
                                #                FixedValueAcc(None),
                                #                LinearGaussianAcc(inputLength = 2 * lf0StreamDepth + 1)
                                #            ]
                                #        )
                                #    )
                                #)
                                OracleAcc()
                        ,   2:
                                OracleAcc()
                        }[streamIndex]
                    )
                )
            )
        )

        timed(corpus.accumulate)(acc)

        dist, trainLogLike, trainOcc = timed(defaultEstimate)(acc)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'mono.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'mono', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'mono.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'mono.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'mono.4mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'mono.4mix', afterSynth = getDrawMgc())

        printTime('finished mono')

    def doTimingInfoSystem():
        print
        print 'TRAINING MONOPHONE SYSTEM WITH TIMING INFO'
        printTime('started timinginfo')

        extraLength = 2

        def alignmentToPhoneticSeq(alignment):
            for startTime, endTime, label in alignment:
                phone = label.phone
                for time in range(startTime, endTime):
                    framesBefore = time - startTime
                    framesAfter = endTime - time - 1
                    yield phone, [framesBefore, framesAfter]
        def convertTimingInfo(input):
            (phone, extra), acousticContext = input
            assert len(extra) == extraLength
            return phone, (extra, acousticContext)

        acc = MappedInputAcc(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            SequenceAcc(10,
                MappedInputAcc(convertTimingInfo,
                    createDiscreteAcc(phoneset.phoneList, lambda phone:
                        frameSummarizer.createAcc(True, lambda streamIndex:
                            {
                                0:
                                    mgcSummarizer.createAcc(True, lambda outIndex:
                                        MappedInputAcc(concatenate,
                                            MappedInputAcc(AddBias(),
                                                LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + extraLength + 1)
                                            )
                                        )
                                    )
                            ,   1:
                                    OracleAcc()
                            ,   2:
                                    OracleAcc()
                            }[streamIndex]
                        )
                    )
                )
            )
        )

        timed(corpus.accumulate)(acc)

        dist, trainLogLike, trainOcc = timed(defaultEstimate)(acc)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'timinginfo.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'timinginfo', afterSynth = getDrawMgc())

        printTime('finished timinginfo')

    def doDecisionTreeClusteredSystem():
        print
        print 'DECISION TREE CLUSTERING'
        printTime('started clustered')

        def alignmentToPhoneticSeq(alignment):
            for startTime, endTime, label in alignment:
                for time in range(startTime, endTime):
                    yield label

        questionGroups = questions_htsDemo.getFullContextQuestionGroups()

        acc = MappedInputAcc(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            SequenceAcc(10,
                frameSummarizer.createAcc(True, lambda streamIndex:
                    {
                        0:
                            AutoGrowingDiscreteAcc(lambda:
                                mgcSummarizer.createAcc(False, lambda outIndex:
                                    MappedInputAcc(AddBias(),
                                        LinearGaussianAcc(inputLength = mgcSummarizer.vectorLength(outIndex) + 1)
                                    )
                                )
                            )
                    ,   1:
                            OracleAcc()
                    ,   2:
                            OracleAcc()
                    }[streamIndex]
                )
            )
        )

        timed(corpus.accumulate)(acc)

        def decisionTreeClusterEstimatePartial(acc, estimateChild):
            if isinstance(acc, AutoGrowingDiscreteAcc):
                return timed(acc.decisionTreeCluster)(questionGroups, thresh = 500.0, minOcc = 10.0, maxOcc = None, verbosity = 3)
        decisionTreeClusterEstimate = getEstimate([decisionTreeClusterEstimatePartial, defaultEstimatePartial])

        dist, trainLogLike, trainOcc = decisionTreeClusterEstimate(acc)
        print
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'clustered.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'clustered', afterSynth = getDrawMgc())

        print
        print 'MIXING UP (to 2 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'clustered.2mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'clustered.2mix', afterSynth = getDrawMgc())
        print
        print 'MIXING UP (to 4 components)'
        dist = mixup(dist, corpus.accumulate)
        writeDistFile(os.path.join(distOutDir, 'clustered.4mix.dist'), dist)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'clustered.4mix', afterSynth = getDrawMgc())

        printTime('finished clustered')

    def doTransformSystem():
        print
        print 'TRAINING TRANSFORM SYSTEM'
        printTime('started xf')

        globalPhone = True
        studentResiduals = True
        # mgcStreamDepth affects what pictures we can draw
        print 'DEBUG: mgcStreamDepth =', mgcStreamDepth
        numTanhTransforms = 3

        phoneList = ['global'] if globalPhone else phoneset.phoneList

        def alignmentToPhoneticSeq(alignment):
            for startTime, endTime, label in alignment:
                phone = label.phone
                for time in range(startTime, endTime):
                    yield 'global' if globalPhone else phone

        # FIXME : only works if no cross-stream stuff happening.  Make more robust somehow.
        shiftToPrevTransform = ShiftOutputTransform(MinusPrev())

        mgcOutputTransform = dict()
        mgcInputTransform = dict()
        for outIndex in mgcSummarizer.outIndices:
            xmin, xmax = corpus.mgcLims[outIndex]
            bins = linspace(xmin, xmax, numTanhTransforms + 1)
            binCentres = bins[:-1] + 0.5 * diff(bins)
            width = (xmax - xmin) / numTanhTransforms / 2.0
            tanhOutputTransforms = [ TanhTransform1D(array([0.0, width, binCentre])) for binCentre in binCentres ]
            outputWarp = SumTransform1D([IdentityTransform()] + tanhOutputTransforms).withTag(('mgcOutputWarp', outIndex))
            mgcOutputTransform[outIndex] = SimpleOutputTransform(outputWarp, checkDerivPositive1D = True).withTag(('mgcOutputTransform', outIndex))
            tanhInputTransforms = [ TanhTransform1D(array([0.0, width, binCentre])) for binCentre in binCentres ]
            inputWarp = SumTransform1D([IdentityTransform()] + tanhInputTransforms).withTag(('mgcInputWarp', outIndex))
            mgcInputTransform[outIndex] = VectorizeTransform(inputWarp).withTag(('mgcInputTransform', outIndex))

        dist = MappedInputDist(lambda alignment: list(alignmentToPhoneticSeq(alignment)),
            SequenceDist(10,
                createDiscreteDist(phoneList, lambda phone:
                    frameSummarizer.createDist(False, lambda streamIndex:
                        {
                            0:
                                mgcSummarizer.createDist(False, lambda outIndex:
                                    DebugDist(None,
                                        TransformedOutputDist(mgcOutputTransform[outIndex],
                                            TransformedInputDist(mgcInputTransform[outIndex],
                                                #MappedOutputDist(shiftToPrevTransform,
                                                    DebugDist(None,
                                                        MappedInputDist(AddBias(),
                                                            # arbitrary dist to get things rolling
                                                            LinearGaussian(zeros([mgcSummarizer.vectorLength(outIndex) + 1]), 1.0)
                                                        )
                                                    ).withTag('debug-xfed')
                                                #)
                                            )
                                        )
                                    ).withTag(('debug-orig', phone, streamIndex, outIndex))
                                )
                        ,   1:
                                OracleDist()
                        ,   2:
                                OracleDist()
                        }[streamIndex]
                    )
                )
            )
        )

        def drawVarious(dist, id, simpleResiduals = False, debugResiduals = False):
            assert not (simpleResiduals and debugResiduals)
            acc = defaultParamSpec.createAccG(dist)
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
                draw.drawWarping([ outputWarp, inputWarp ], outPdf = outPdf, xlims = lims, title = outPdf)

                for phone in phoneList:
                    accOrig = nodetree.findTaggedNode(acc, lambda tag: tag == ('debug-orig', phone, streamIndex, outIndex))
                    distOrig = nodetree.findTaggedNode(dist, lambda tag: tag == ('debug-orig', phone, streamIndex, outIndex))

                    debugAcc = accOrig
                    subDist = distOrig.dist
                    if mgcStreamDepth == 1:
                        outPdf = os.path.join(figOutDir, 'scatter-'+id+'-orig-'+str(phone)+'-'+streamId+'.pdf')
                        draw.drawFor1DInput(debugAcc, subDist, outPdf = outPdf, xlims = lims, ylims = lims, title = outPdf)

                    debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-xfed')
                    subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-xfed').dist
                    if mgcStreamDepth == 1:
                        outPdf = os.path.join(figOutDir, 'scatter-'+id+'-xfed-'+str(phone)+'-'+streamId+'.pdf')
                        draw.drawFor1DInput(debugAcc, subDist, outPdf = outPdf, xlims = map(inputWarp, lims), ylims = map(outputWarp, lims), title = outPdf)

                    if simpleResiduals:
                        debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-xfed')
                        subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-xfed').dist
                        residuals = array([ subDist.dist.residual(AddBias()(input), output) for input, output in zip(debugAcc.memo.inputs, debugAcc.memo.outputs) ])
                        f = lambda x: -0.5 * log(2.0 * pi) - 0.5 * x * x
                        outPdf = os.path.join(figOutDir, 'residualLogPdf-'+id+'-'+str(phone)+'-'+streamId+'.pdf')
                        if len(residuals) > 0:
                            draw.drawLogPdf(residuals, bins = 20, fns = [f], outPdf = outPdf, title = outPdf)

                    if debugResiduals:
                        debugAcc = nodetree.findTaggedNode(accOrig, lambda tag: tag == 'debug-residual')
                        subDist = nodetree.findTaggedNode(distOrig, lambda tag: tag == 'debug-residual').dist
                        f = lambda output: subDist.logProb([], output)
                        outPdf = os.path.join(figOutDir, 'residualLogPdf-'+id+'-'+str(phone)+'-'+streamId+'.pdf')
                        if len(debugAcc.memo.outputs) > 0:
                            draw.drawLogPdf(debugAcc.memo.outputs, bins = 20, fns = [f], outPdf = outPdf, title = outPdf)

                    # (FIXME : replace with looking up sub-transform directly once tree structure for transforms is done)
                    residualTransforms = nodetree.findTaggedNodes(distOrig, lambda tag: tag == 'residualTransform')
                    assert len(residualTransforms) <= 1
                    for residualTransform in residualTransforms:
                        outPdf = os.path.join(figOutDir, 'residualTransform-'+id+'-'+str(phone)+'-'+streamId+'.pdf')
                        draw.drawWarping([residualTransform.transform], outPdf = outPdf, xlims = [-2.5, 2.5], title = outPdf)

        dist, trainLogLike, trainOcc = timed(trainEM)(dist, corpus.accumulate, estimate = defaultEstimate, minIterations = 2, maxIterations = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'xf_init.dist'), dist)
        timed(drawVarious)(dist, id = 'xf_init', simpleResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'xf_init', afterSynth = getDrawMgc())

        print
        print 'ESTIMATING "GAUSSIANIZATION" TRANSFORMS'
        def afterEst(dist, logLike, occ, it):
            reportTrainLogLike(logLike, occ)
            #timed(drawVarious)(dist, id = 'xf-it'+str(it))
        # (FIXME : change mgcInputTransform to mgcInputWarp and mgcOutputTransform to mgcOutputWarp once tree structure for transforms is done)
        mgcWarpParamSpec = getByTagParamSpec(lambda tag: getFirst(tag) == 'mgcInputTransform' or getFirst(tag) == 'mgcOutputTransform')
        dist, trainLogLike, trainOcc = timed(trainCGandEM)(dist, corpus.accumulate, ps = mgcWarpParamSpec, iterations = 5, length = -25, afterEst = afterEst, verbosity = 2)
        writeDistFile(os.path.join(distOutDir, 'xf.dist'), dist)
        timed(drawVarious)(dist, id = 'xf', simpleResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'xf', afterSynth = getDrawMgc())

        if studentResiduals:
            print
            print 'USING STUDENT RESIDUALS'
            dist = convertToStudentResiduals(dist, studentTag = 'student', debugTag = 'debug-residual', subtractMeanTag = 'subtractMean')
            residualParamSpec = getByTagParamSpec(lambda tag: tag == 'student')
            subtractMeanParamSpec = getByTagParamSpec(lambda tag: tag == 'subtractMean')
        else:
            print
            print 'USING TRANSFORMED-GAUSSIAN RESIDUALS'
            dist = convertToTransformedGaussianResiduals(dist, residualTransformTag = 'residualTransform', debugTag = 'debug-residual', subtractMeanTag = 'subtractMean')
            residualParamSpec = getByTagParamSpec(lambda tag: tag == 'residualTransform')
            subtractMeanParamSpec = getByTagParamSpec(lambda tag: tag == 'subtractMean')
        timed(drawVarious)(dist, id = 'xf.res_init', debugResiduals = True)
        dist, trainLogLike, trainOcc = timed(trainCG)(dist, corpus.accumulate, ps = residualParamSpec, length = -50, verbosity = 2)
        dist, trainLogLike, trainOcc = timed(trainCG)(dist, corpus.accumulate, ps = subtractMeanParamSpec, length = -50, verbosity = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'xf.res.dist'), dist)
        timed(drawVarious)(dist, id = 'xf.res', debugResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'xf.res', afterSynth = getDrawMgc())

        print
        print 'ESTIMATING ALL PARAMETERS'
        dist, trainLogLike, trainOcc = timed(trainCG)(dist, corpus.accumulate, ps = defaultParamSpec, length = -200, verbosity = 2)
        reportTrainLogLike(trainLogLike, trainOcc)
        writeDistFile(os.path.join(distOutDir, 'xf.res.xf.dist'), dist)
        timed(drawVarious)(dist, id = 'xf.res.xf', debugResiduals = True)
        evaluateLogProb(dist, corpus)
        evaluateSynthesize(dist, corpus, 'xf.res.xf', afterSynth = getDrawMgc())

        printTime('finished xf')

    try:
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
