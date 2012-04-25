"""Unit tests for distributions, accumulators and model training."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import nodetree
import dist as d
import train as trn
import summarizer
import transform as xf
import wnet
from armspeech.util.mathhelp import logSum
from armspeech.util.iterhelp import chunkList

import test_dist_questions
import test_transform

import unittest
import sys
from collections import deque
import math
import random
import numpy as np
from numpy.random import randn, randint
import numpy.linalg as la
from scipy import stats
import string

def logProb_frames(dist, trainData):
    lp = 0.0
    frames = 0
    for input, output in trainData:
        lp += dist.logProb(input, output)
        frames += 1
    return lp, frames

def assert_allclose(actual, desired, rtol = 1e-7, atol = 1e-14, msg = 'items not almost equal'):
    if np.shape(actual) != np.shape(desired) or not np.allclose(actual, desired, rtol, atol):
        raise AssertionError(msg+'\n ACTUAL:  '+repr(actual)+'\n DESIRED: '+repr(desired))

def randBool():
    return randint(0, 2) == 0

def randTag():
    return 'tag'+str(randint(0, 1000000))

def simpleInputGen(dimIn, bias = False):
    while True:
        ret = randn(dimIn)
        if bias:
            ret[-1] = 1.0
        yield ret

# (FIXME : add tests to test full range of shapes for transform stuff)
# (FIXME : add tests for logProbDerivInput and logProbDerivOutput for dists where either input or output has a discrete component (applies to lots of below tests, e.g. logProbDerivOutput for IdentifiableMixtureDist))
# (FIXME : add tests for Transformed(Input|Output)Learn(Dist|Transform)AccEM (for the Transform ones, have to first add a transform that can be re-estimated using EM))
# (FIXME : deep test for Transformed(Input|Output)Dist doesn't seem to converge to close to true dist in terms of parameters. Multiple local minima? Or just very insensitive to details? For more complicated transforms might the test procedure never converge?)

def gen_LinearGaussian(dimIn = 3, bias = False):
    coeff = randn(dimIn)
    variance = math.exp(randn())
    dist = d.LinearGaussian(coeff, variance).withTag(randTag())
    return dist, simpleInputGen(dimIn, bias = bias)

def gen_StudentDist(dimIn = 3):
    df = math.exp(randn() + 1.0)
    precision = math.exp(randn())
    dist = d.StudentDist(df, precision).withTag(randTag())
    return dist, simpleInputGen(dimIn)

def gen_ConstantClassifier(numClasses = 5):
    logProbs = randn(numClasses)
    logProbs -= logSum(logProbs)
    dist = d.ConstantClassifier(logProbs).withTag(randTag())
    return dist, simpleInputGen(0)

def gen_BinaryLogisticClassifier(dimIn = 3, bias = False, useZeroCoeff = False):
    if useZeroCoeff:
        w = np.zeros([dimIn])
    else:
        w = randn(dimIn)
    dist = d.BinaryLogisticClassifier(w).withTag(randTag())
    return dist, simpleInputGen(dimIn, bias = bias)

def gen_classifier(numClasses, dimIn, bias = False):
    """Generates a random classifier with vector input."""
    if numClasses == 2 and randBool():
        return gen_BinaryLogisticClassifier(dimIn = dimIn, bias = bias)
    else:
        return gen_ConstantClassifier(numClasses)

def gen_MixtureDist(dimIn):
    return gen_MixtureOfTwoExperts(dimIn = 3)

def gen_MixtureOfTwoExperts(dimIn = 3, bias = False):
    blc, blcGen = gen_BinaryLogisticClassifier(dimIn, bias = bias)
    dist0 = gen_LinearGaussian(dimIn)[0]
    dist1 = gen_LinearGaussian(dimIn)[0]
    dist = d.MixtureDist(blc, [dist0, dist1]).withTag(randTag())
    return dist, blcGen

def gen_IdentifiableMixtureDist(dimIn = 3, blcUseZeroCoeff = False):
    blc, blcGen = gen_BinaryLogisticClassifier(dimIn, useZeroCoeff = blcUseZeroCoeff)
    dist0 = d.FixedValueDist(None)
    dist1 = gen_LinearGaussian(dimIn)[0]
    dist = d.IdentifiableMixtureDist(blc, [dist0, dist1]).withTag(randTag())
    return dist, blcGen

def gen_VectorDist(order = 10, depth = 3):
    depths = dict([ (outIndex, depth) for outIndex in range(order) ])
    vectorSummarizer = summarizer.VectorSeqSummarizer(order, depths)
    dist = vectorSummarizer.createDist(False, lambda outIndex:
        d.MappedInputDist(np.array,
            gen_LinearGaussian(depths[outIndex])[0]
        )
    ).withTag(randTag())
    def getInputGen():
        while True:
            yield randn(depth, order)
    return dist, getInputGen()

def gen_DiscreteDist(keys = ['a', 'b', 'c'], dimIn = 3):
    dist = d.createDiscreteDist(keys, lambda key:
        gen_LinearGaussian(dimIn)[0]
    ).withTag(randTag())
    def getInputGen():
        while True:
            yield random.choice(keys), randn(dimIn)
    return dist, getInputGen()

def gen_shared_DiscreteDist(keys = ['a', 'b', 'c'], dimIn = 3):
    subDist = gen_LinearGaussian(dimIn)[0]
    dist = d.createDiscreteDist(keys, lambda key:
        subDist
    ).withTag(randTag())
    def getInputGen():
        while True:
            yield random.choice(keys), randn(dimIn)
    return dist, getInputGen()

def gen_DecisionTree_with_LinearGaussian_leaves(splitProb = 0.49, dimIn = 3):
    labels = test_dist_questions.phoneList
    questionGroups = test_dist_questions.getQuestionGroups()

    def decisionTree(labelsLeft):
        fullQuestionsLeft = []
        for labelValuer, questions in questionGroups:
            for question in questions:
                yesNonEmpty = False
                noNonEmpty = False
                for label in labelsLeft:
                    if question(labelValuer(label)):
                        yesNonEmpty = True
                    else:
                        noNonEmpty = True
                    if yesNonEmpty and noNonEmpty:
                        break
                if yesNonEmpty and noNonEmpty:
                    fullQuestionsLeft.append((labelValuer, question))

        if random.random() > splitProb or not fullQuestionsLeft:
            return d.DecisionTreeLeaf(gen_LinearGaussian(dimIn)[0])
        else:
            fullQuestion = random.choice(fullQuestionsLeft)
            labelValuer, question = fullQuestion
            labelsLeftYes = []
            labelsLeftNo = []
            for label in labelsLeft:
                if question(labelValuer(label)):
                    labelsLeftYes.append(label)
                else:
                    labelsLeftNo.append(label)
            return d.DecisionTreeNode(fullQuestion, decisionTree(labelsLeftYes), decisionTree(labelsLeftNo))
    def getInputGen():
        while True:
            yield random.choice(labels), randn(dimIn)
    return decisionTree(labels).withTag(randTag()), getInputGen()

def gen_MappedInputDist(dimIn = 3, dimOut = 2):
    transform = test_transform.gen_genericTransform([dimIn], [dimOut])
    subDist = gen_LinearGaussian(dimOut)[0]
    return d.MappedInputDist(transform, subDist).withTag(randTag()), simpleInputGen(dimIn)

def gen_MappedOutputDist(dimInput = 3):
    outputTransform = test_transform.gen_genericOutputTransform([dimInput], [])
    subDist, inputGen = gen_LinearGaussian(dimInput)
    return d.MappedOutputDist(outputTransform, subDist).withTag(randTag()), inputGen

def gen_TransformedInputDist(dimIn = 3, dimOut = 2):
    transform = test_transform.gen_genericTransform([dimIn], [dimOut])
    subDist = gen_LinearGaussian(dimOut)[0]
    return d.TransformedInputDist(transform, subDist).withTag(randTag()), simpleInputGen(dimIn)

def gen_TransformedOutputDist(dimInput = 3):
    outputTransform = test_transform.gen_genericOutputTransform([dimInput], [])
    subDist, inputGen = gen_LinearGaussian(dimInput)
    return d.TransformedOutputDist(outputTransform, subDist).withTag(randTag()), inputGen

def gen_nestedTransformDist(dimInputs = [3, 4, 2]):
    assert len(dimInputs) >= 1
    dimIn = dimInputs[-1]
    dist = gen_LinearGaussian(dimIn)[0]
    if randint(0, 2) == 0:
        outputTransform = test_transform.gen_genericOutputTransform([dimIn], [])
        if randint(0, 2) == 0:
            dist = d.MappedOutputDist(outputTransform, dist)
        else:
            dist = d.TransformedOutputDist(outputTransform, dist)
    for dimIn, dimOut in reversed(zip(dimInputs, dimInputs[1:])):
        transform = test_transform.gen_genericTransform([dimIn], [dimOut])
        if randint(0, 2) == 0:
            dist = d.MappedInputDist(transform, dist)
        else:
            dist = d.TransformedInputDist(transform, dist)
        if randint(0, 2) == 0:
            outputTransform = test_transform.gen_genericOutputTransform([dimIn], [])
            if randint(0, 2) == 0:
                dist = d.MappedOutputDist(outputTransform, dist)
            else:
                dist = d.TransformedOutputDist(outputTransform, dist)
    return dist.withTag(randTag()), simpleInputGen(dimInputs[0])

def gen_PassThruDist(dimIn = 3):
    subDist, inputGen = gen_LinearGaussian(dimIn)
    return d.PassThruDist(subDist).withTag(randTag()), inputGen

def gen_DebugDist(maxOcc = None, dimIn = 3):
    subDist, inputGen = gen_LinearGaussian(dimIn)
    return d.DebugDist(maxOcc, subDist).withTag(randTag()), inputGen

def gen_autoregressive_dist(depth = 2):
    dist = d.MappedInputDist(np.asarray,
        d.MappedInputDist(xf.AddBias(),
            gen_LinearGaussian(dimIn = depth + 1)[0]
        )
    )
    return dist, None
def autoregressive_1D_is_stable(dist, depth, starts = 5, stepsIntoFuture = 100, bigThresh = 1.0e6):
    for start in range(starts):
        input = deque(randn(depth))
        for step in range(stepsIntoFuture):
            assert len(input) == depth
            output = dist.synth(list(input))
            if abs(output) > bigThresh:
                return False
            input.append(output)
            input.popleft()
    return True
def gen_stable_autoregressive_dist(depth = 2):
    while True:
        dist = gen_autoregressive_dist(depth)[0]
        if autoregressive_1D_is_stable(dist, depth):
            break
    return dist, None

def add_autoregressive_style_labels(concreteNet, genLabels):
    net = concreteNet
    numNodes = net.numNodes
    edgesForwards = [ [] for node in range(numNodes) ]
    for node in range(numNodes):
        nextNodes = [ nextNode for label, nextNode in net.next(node, forwards = True) ]
        labels = genLabels(numLabels = len(nextNodes))
        edgesForwards[node] = zip(labels, nextNodes)
    return wnet.ConcreteNet(startNode = net.startNode, endNode = net.endNode, elems = net.elems, edgesForwards = edgesForwards)
def gen_autoregressive_style_net(genElem, genLabels, sortable = True, maxNodes = 20, maxEdgesPerNode = 3):
    numNodes = randint(2, maxNodes + 1)
    elems = [None] + [ genElem() for node in range(1, numNodes - 1) ] + [None]

    edgesForwards = dict()
    for node in range(0, numNodes - 1):
        edgesForwards[node] = []
        elem = elems[node]
        for edge in range(randint(1, maxEdgesPerNode + 1)):
            while True:
                if elem is not None and randBool():
                    nextNode = node
                else:
                    nextNode = randint(1, numNodes)
                if (not sortable) or elem is not None or elems[nextNode] is not None or nextNode > node:
                    break
            edgesForwards[node].append((None, nextNode))
    edgesForwards[numNodes - 1] = []

    net = wnet.ConcreteNet(startNode = 0, endNode = numNodes - 1, elems = elems, edgesForwards = edgesForwards)

    nodeSet = wnet.nodeSetCompute(net, accessibleOnly = True)
    if not nodeSet:
        return gen_autoregressive_style_net(genElem = genElem, genLabels = genLabels, sortable = sortable, maxNodes = maxNodes, maxEdgesPerNode = maxEdgesPerNode)
    else:
        perm = list(nodeSet)
        random.shuffle(perm)
        netPerm = wnet.concretizeNet(net, perm)
        netFinal = add_autoregressive_style_labels(netPerm, genLabels)

        wnet.checkConsistent(netFinal, nodeSet = set(range(len(nodeSet))))
        return netFinal
def gen_simple_autoregressive_style_net(label, acSubLabels, durSubLabels):
    def genElem():
        return None if randBool() else random.choice([ (label, subLabel) for subLabel in acSubLabels ])
    def genLabels(numLabels):
        if numLabels == 0:
            return []
        elif numLabels == 1:
            return [None]
        else:
            # numLabels added below to ensure all nodes with the same phonetic
            #   context have the same number of edges leaving them
            context = random.choice([ (label, (subLabel, numLabels)) for subLabel in durSubLabels ])
            perm = list(range(numLabels))
            random.shuffle(perm)
            return [ (context, adv) for adv in perm ]
    net = gen_autoregressive_style_net(genElem, genLabels, sortable = True)

    # collect phonetic contexts and outputs actually present in net
    nodeSet = wnet.nodeSetCompute(net)
    phInputsAc = set([ net.elem(node) for node in nodeSet ])
    phInputsAc.remove(None)
    phInputToNumClassesDur = dict()
    for node in nodeSet:
        labels = [ label for label, nextNode in net.next(node, forwards = True) ]
        if len(labels) >= 2 or len(labels) == 1 and labels[0] is not None:
            phInputs, phOutputs = zip(*labels)
            numClasses = len(phOutputs)
            assert set(phOutputs) == set(range(numClasses))
            phInput = phInputs[0]
            for phInputAgain in phInputs[1:]:
                assert phInputAgain == phInput
            if phInput in phInputToNumClassesDur:
                assert phInputToNumClassesDur[phInput] == numClasses
            else:
                phInputToNumClassesDur[phInput] = numClasses

    return net, phInputsAc, phInputToNumClassesDur

def gen_constant_AutoregressiveNetDist(depth = 2):
    """Generates an AutoregressiveNetDist which is independent of input."""
    numSubLabels = randint(1, 5)
    while True:
        net, phInputsAc, phInputToNumClassesDur = gen_simple_autoregressive_style_net('g', acSubLabels = range(numSubLabels), durSubLabels = range(numSubLabels))
        nodeSet = wnet.nodeSetCompute(net, accessibleOnly = True)
        numEmitting = len([ node for node in nodeSet if net.elem(node) is not None ])
        if numEmitting >= 2 or numEmitting == 1 and randint(0, 4) == 0 or numEmitting == 0 and randint(0, 10) == 0:
            break
    def makeFullDepth(acInput):
        if len(acInput) < depth:
            acInput = np.concatenate((np.zeros((depth - len(acInput),)), acInput))
        return np.asarray(acInput)
    durDist = d.createDiscreteDist(phInputToNumClassesDur.keys(), lambda phInput:
        d.MappedInputDist(makeFullDepth,
            d.MappedInputDist(xf.AddBias(),
                gen_classifier(numClasses = phInputToNumClassesDur[phInput], dimIn = depth + 1)[0]
            )
        )
    )
    acDist = d.createDiscreteDist(list(phInputsAc), lambda phInput:
        d.MappedInputDist(makeFullDepth,
            gen_stable_autoregressive_dist(depth)[0]
        )
    )
    def netFor(input):
        return net
    dist = d.AutoregressiveNetDist(depth, netFor, durDist, acDist).withTag(randTag())

    def getInputGen():
        while True:
            yield ''
    return dist, getInputGen()

def gen_inSeq_AutoregressiveNetDist(depth = 2):
    """Generates a left-to-right AutoregressiveNetDist where input is a label sequence."""
    labels = string.lowercase[:randint(1, 10)]
    numSubLabels = randint(1, 5)
    subLabels = list(range(numSubLabels))
    def netFor(labelSeq):
        net = wnet.FlatMappedNet(
            # (FIXME : any reason to memoize this?)
            lambda label: wnet.probLeftToRightNet(
                [ (label, subLabel) for subLabel in subLabels ],
                [ [ ((label, subLabel), adv) for adv in [0, 1] ] for subLabel in subLabels ]
            ),
            wnet.SequenceNet(labelSeq, None)
        )
        return net
    def makeFullDepth(acInput):
        if len(acInput) < depth:
            acInput = np.concatenate((np.zeros((depth - len(acInput),)), acInput))
        return np.asarray(acInput)
    labelledSubLabels = [ (label, subLabel) for label in labels for subLabel in subLabels ]
    durDist = d.createDiscreteDist(labelledSubLabels, lambda (label, subLabel):
        d.MappedInputDist(makeFullDepth,
            d.MappedInputDist(xf.AddBias(),
                gen_classifier(numClasses = 2, dimIn = depth + 1)[0]
            )
        )
    )
    acDist = d.createDiscreteDist(labelledSubLabels, lambda (label, subLabel):
        d.MappedInputDist(makeFullDepth,
            gen_stable_autoregressive_dist(depth)[0]
        )
    )
    dist = d.AutoregressiveNetDist(depth, netFor, durDist, acDist).withTag(randTag())

    def getInputGen():
        while True:
            labelSeq = [ random.choice(labels) for i in range(randint(0, 10)) ]
            yield labelSeq
    return dist, getInputGen()

def iidLogProb(dist, training):
    logProb = 0.0
    for input, output, occ in training:
        logProb += dist.logProb(input, output) * occ
    return logProb

def trainedAcc(dist, training):
    acc = d.defaultCreateAcc(dist)
    for input, output, occ in training:
        acc.add(input, output, occ)
    return acc

def trainedAccG(dist, training, ps = d.defaultParamSpec):
    acc = ps.createAccG(dist)
    for input, output, occ in training:
        acc.add(input, output, occ)
    return acc

def randomizeParams(dist, ps = d.defaultParamSpec):
    return ps.parseAll(dist, randn(*np.shape(ps.params(dist))))

def reparse(dist, ps):
    params = ps.params(dist)
    assert len(np.shape(params)) == 1
    distParsed = ps.parseAll(dist, params)
    paramsParsed = ps.params(distParsed)
    assert_allclose(paramsParsed, params)
    assert dist.tag == distParsed.tag
    return distParsed

def check_logProbDerivInput(dist, input, output, eps):
    inputDelta = randn(*np.shape(input)) * eps
    numericDelta = dist.logProb(input + inputDelta, output) - dist.logProb(input, output)
    analyticDelta = np.dot(inputDelta, dist.logProbDerivInput(input, output))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)

def check_logProbDerivOutput(dist, input, output, eps):
    outputDelta = randn(*np.shape(output)) * eps
    numericDelta = dist.logProb(input, output + outputDelta) - dist.logProb(input, output)
    analyticDelta = np.dot(outputDelta, dist.logProbDerivOutput(input, output))
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4)

def check_addAcc(dist, trainingAll, ps):
    accAll = trainedAccG(dist, trainingAll, ps = ps)
    logLikeAll = accAll.logLike()
    derivParamsAll = ps.derivParams(accAll)

    trainingParts = chunkList(trainingAll, numChunks = randint(1, 5))
    accs = [ trainedAccG(dist, trainingPart, ps = ps) for trainingPart in trainingParts ]
    accFull = accs[0]
    for acc in accs[1:]:
        d.addAcc(accFull, acc)
    logLikeFull = accFull.logLike()
    derivParamsFull = ps.derivParams(accFull)

    assert_allclose(logLikeFull, logLikeAll)
    assert_allclose(derivParamsFull, derivParamsAll)

def check_logLike(dist, training, iid, hasEM):
    assert iid == True
    logLikeFromDist = iidLogProb(dist, training)
    if hasEM:
        logLikeEM = trainedAcc(dist, training).logLike()
        assert_allclose(logLikeEM, logLikeFromDist)
    logLikeG = trainedAccG(dist, training).logLike()
    assert_allclose(logLikeG, logLikeFromDist)

def check_derivParams(dist, training, ps, eps):
    params = ps.params(dist)
    acc = trainedAccG(dist, training, ps = ps)
    logLike = acc.logLike()
    derivParams = ps.derivParams(acc)
    paramsDelta = randn(*np.shape(params)) * eps
    distNew = ps.parseAll(dist, params + paramsDelta)
    logLikeNew = trainedAccG(distNew, training, ps = ps).logLike()
    assert_allclose(ps.params(distNew), params + paramsDelta)

    numericDelta = logLikeNew - logLike
    analyticDelta = np.dot(derivParams, paramsDelta)
    assert_allclose(numericDelta, analyticDelta, rtol = 1e-4, atol = 1e-10)

def getTrainEM(initEstDist, verbosity = 0):
    def doTrainEM(training):
        def accumulate(acc):
            for input, output, occ in training:
                acc.add(input, output, occ)
        dist, logLike, occ = trn.trainEM(initEstDist, accumulate, deltaThresh = 1e-9, verbosity = verbosity)
        assert initEstDist.tag is not None
        assert dist.tag == initEstDist.tag
        return dist, logLike, occ
    return doTrainEM

def getTrainCG(initEstDist, ps = d.defaultParamSpec, length = -500, verbosity = 0):
    def doTrainCG(training):
        def accumulate(acc):
            for input, output, occ in training:
                acc.add(input, output, occ)
        dist, logLike, occ = trn.trainCG(initEstDist, accumulate, ps = ps, length = length, verbosity = verbosity)
        assert initEstDist.tag is not None
        assert dist.tag == initEstDist.tag
        return dist, logLike, occ
    return doTrainCG

def getTrainFromAcc(createAcc):
    def doTrainFromAcc(training):
        acc = createAcc()
        for input, output, occ in training:
            acc.add(input, output, occ)
        dist, logLike, occ = d.defaultEstimate(acc)
        assert acc.tag is not None
        assert dist.tag == acc.tag
        return dist, logLike, occ
    return doTrainFromAcc

def check_est(trueDist, train, inputGen, hasParams, iid = True, unitOcc = False, ps = d.defaultParamSpec, logLikeThresh = 2e-2, tslpThresh = 2e-2, testSetSize = 50, initTrainingSetSize = 100, trainingSetMult = 5, maxTrainingSetSize = 100000):
    """Checks specified training method converges with sufficient data.

    More specifically, checks that, for a sufficient amount of training data,
    training using train produces a distribution that assigns roughly the true
    log probability to an unseen test set of size testSetSize.
    (There are several additional checks, but this is the main one).

    train specifies both the training procedure and the initialization, and
    the initialization used should be appropriate for the training procedure.
    For simple models with effective training procedures (e.g. learning a linear
    Gaussian model using expectation-maximization) initializing with a random
    model (e.g. train = getTrainEM(<some random dist in the same family>) )
    provides a stringent test.
    For more complicated models with weaker training procedures (e.g. learning a
    mixture of linear Gaussians using gradient descent, which suffers from
    multiple local optima) initializing with the true distribution (e.g.
    train = getTrainCG(trueDist) ) at least provides a check that the training
    procedure doesn't move away from the true distribution when given sufficient
    training data.
    """
    assert iid == True

    inputsTest = [ input for input, index in zip(inputGen, range(testSetSize)) ]
    testSet = [ (input, trueDist.synth(input), 1.0 if unitOcc else math.exp(randn())) for input in inputsTest ]
    testOcc = sum([ occ for input, output, occ in testSet ])

    training = []

    def extendTrainingSet(trainingSetSizeDelta):
        inputsNew = [ input for input, index in zip(inputGen, range(trainingSetSizeDelta)) ]
        trainingNew = [ (input, trueDist.synth(input), 1.0 if unitOcc else math.exp(randn())) for input in inputsNew ]
        training.extend(trainingNew)

    converged = False
    while not converged and len(training) < maxTrainingSetSize:
        extendTrainingSet((trainingSetMult - 1) * len(training) + initTrainingSetSize)
        trueLogProb = iidLogProb(trueDist, training)
        estDist, estLogLike, estOcc = train(training)
        # N.B. tests both that training has converged and that logLike from estimate agrees with iid logProb from dist
        # (FIXME : might be nice to separate these two criterion)
        assert_allclose(estLogLike, iidLogProb(estDist, training))
        assert_allclose(estOcc, sum([ occ for input, output, occ in training ]))
        tslpTrue = iidLogProb(trueDist, testSet)
        tslpEst = iidLogProb(estDist, testSet)
        if hasParams:
            newAcc = trainedAccG(estDist, training, ps = ps)
            assert_allclose(newAcc.occ, estOcc)
            derivParams = ps.derivParams(newAcc)
            assert_allclose(derivParams / estOcc, np.zeros([len(derivParams)]), atol = 1e-4)
        # (FIXME : math.isinf ~20x faster but only exists in python >= 2.6)
        if np.isinf(estLogLike):
            print 'NOTE: singularity in likelihood function (training set size =', len(training), ', occ '+repr(estOcc)+', estDist =', estDist, ', estLogLike =', estLogLike, ')'
        if abs(trueLogProb - estLogLike) / estOcc < logLikeThresh and abs(tslpTrue - tslpEst) / testOcc < tslpThresh:
            converged = True
    if not converged:
        raise AssertionError('estimated dist did not converge to true dist\n\ttraining set size = '+str(len(training))+'\n\ttrueLogProb = '+str(trueLogProb / estOcc)+' vs estLogLike = '+str(estLogLike / estOcc)+'\n\ttslpTrue = '+str(tslpTrue / testOcc)+' vs tslpEst = '+str(tslpEst / testOcc)+'\n\ttrueDist = '+repr(trueDist)+'\n\testDist = '+repr(estDist))

def getTrainingSet(dist, inputGen, typicalSize, iid, unitOcc):
    trainingSetSize = random.choice([0, 1, 2, typicalSize - 1, typicalSize, typicalSize + 1, 2 * typicalSize - 1, 2 * typicalSize, 2 * typicalSize + 1])
    inputs = [ input for input, index in zip(inputGen, range(trainingSetSize)) ]
    if iid:
        trainingSet = [ (input, dist.synth(input), 1.0 if unitOcc else math.exp(randn())) for input in inputs ]
    else:
        assert unitOcc == True
        # (FIXME : potentially very slow. Could rewrite some of GP stuff to do this better if necessary.)
        updatedDist = dist
        trainingSet = []
        for inputNew in inputs:
            acc = d.defaultCreateAcc(updatedDist)
            for input, output, occ in trainingSet:
                acc.add(input, output, occ)
            updatedDist = d.defaultEstimate(acc)[0]
            trainingSet.append((inputNew, updatedDist.synth(inputNew), 1.0))
    assert len(trainingSet) == trainingSetSize
    return trainingSet

def checkLots(dist, inputGen, hasParams, eps, numPoints, iid = True, unitOcc = False, hasEM = True, ps = d.defaultParamSpec, logProbDerivInputCheck = False, logProbDerivOutputCheck = False, checkAdditional = None):
    # (FIXME : add pickle test)
    # (FIXME : add eval test)
    assert dist.tag is not None
    if hasEM:
        assert d.defaultCreateAcc(dist).tag == dist.tag
    assert ps.createAccG(dist).tag == dist.tag

    training = getTrainingSet(dist, inputGen, typicalSize = numPoints, iid = iid, unitOcc = unitOcc)

    points = []
    for pointIndex in range(numPoints):
        input = inputGen.next()
        output = dist.synth(input)
        points.append((input, output))

    logProbsBefore = [ dist.logProb(input, output) for input, output in points ]
    if hasParams:
        paramsBefore = ps.params(dist)

    distMapped = nodetree.defaultMap(dist)
    assert id(distMapped) != id(dist)
    assert distMapped.tag == dist.tag
    if hasParams:
        distParsed = reparse(dist, ps)
    for input, output in points:
        # (FIXME : math.isinf ~20x faster but only exists in python >= 2.6)
        if not np.isinf(dist.logProb(input, output)):
            if checkAdditional is not None:
                checkAdditional(dist, input, output, eps)
            lp = dist.logProb(input, output)
            assert_allclose(distMapped.logProb(input, output), lp)
            if hasParams:
                assert_allclose(distParsed.logProb(input, output), lp)
            if logProbDerivInputCheck:
                check_logProbDerivInput(dist, input, output, eps)
            if logProbDerivOutputCheck:
                check_logProbDerivOutput(dist, input, output, eps)
        else:
            print 'NOTE: skipping point with logProb =', dist.logProb(input, output), 'for dist =', dist, 'input =', input, 'output =', output

    if hasParams:
        # (FIXME : add addAcc check for Accs which are not AccGs)
        check_addAcc(dist, training, ps)
    if iid:
        check_logLike(dist, training, iid = iid, hasEM = hasEM)
    if hasParams:
        check_derivParams(dist, training, ps, eps = eps)

    logProbsAfter = [ dist.logProb(input, output) for input, output in points ]
    assert_allclose(logProbsAfter, logProbsBefore, msg = 'looks like parsing affected the original distribution, which should never happen')
    if hasParams:
        paramsAfter = ps.params(dist)
        assert_allclose(logProbsAfter, logProbsBefore, rtol = 1e-10, msg = 'looks like parsing affected the original distribution, which should never happen')

    if hasEM:
        # check EM estimation runs at all (if there is a decent amount of data)
        if len(training) >= numPoints - 1:
            estDist, estLogLike, estOcc = getTrainEM(dist)(training)
    if True:
        # check CG estimation runs at all
        estDist, estLogLike, estOcc = getTrainCG(dist, length = -2)(training)

class TestDist(unittest.TestCase):
    def setUp(self):
        self.deepTest = False

    def test_Memo_random_subset(self, its = 10000):
        """Memo class random subsets should be equally likely to include each element"""
        for n in range(0, 5):
            for k in range(n + 1):
                count = np.zeros(n)
                for rep in xrange(its):
                    acc = d.Memo(maxOcc = k)
                    for i in xrange(n):
                        acc.add(i, i)
                    for i in acc.outputs:
                        count[i] += 1
                # (FIXME : thresh hardcoded for 'its' value (and small n, k). Could compute instead.)
                self.assertTrue(la.norm(count / its * n - k) <= 0.05 * n, msg = 'histogram '+repr(count / its)+' for (n, k) = '+repr((n, k)))

    def test_LinearGaussian(self, eps = 1e-8, numDists = 50, numPoints = 100):
        for distIndex in range(numDists):
            bias = random.choice([True, False])
            dimIn = randint(1 if bias else 0, 5)
            dist, inputGen = gen_LinearGaussian(dimIn, bias = bias)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_LinearGaussian(dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)
                createAcc = lambda: d.LinearGaussianAcc(inputLength = dimIn).withTag(randTag())
                check_est(dist, getTrainFromAcc(createAcc), inputGen, hasParams = True)

    def test_StudentDist(self, eps = 1e-8, numDists = 50, numPoints = 100):
        def checkAdditional(dist, input, output, eps):
            assert_allclose(dist.logProb(input, output), math.log(stats.t.pdf(output, dist.df, scale = 1.0 / math.sqrt(dist.precision))))
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_StudentDist(dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, hasEM = False, logProbDerivInputCheck = True, logProbDerivOutputCheck = True, checkAdditional = checkAdditional)
            if self.deepTest:
                initEstDist = gen_StudentDist(dimIn)[0]
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_ConstantClassifier(self, eps = 1e-8, numDists = 50, numPoints = 100):
        for distIndex in range(numDists):
            numClasses = randint(1, 5)
            dist, inputGen = gen_ConstantClassifier(numClasses)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True)
            if self.deepTest:
                initEstDist = gen_ConstantClassifier(numClasses)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)
                createAcc = lambda: d.ConstantClassifierAcc(numClasses = numClasses).withTag(randTag())
                check_est(dist, getTrainFromAcc(createAcc), inputGen, hasParams = True)

    def test_BinaryLogisticClassifier(self, eps = 1e-8, numDists = 50, numPoints = 100):
        for distIndex in range(numDists):
            bias = random.choice([True, False])
            dimIn = randint(1 if bias else 0, 5)
            dist, inputGen = gen_BinaryLogisticClassifier(dimIn, bias = bias)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True)
            if self.deepTest:
                # (useZeroCoeff since it seems to alleviate BinaryLogisticClassifier's convergence issues)
                initEstDist = gen_BinaryLogisticClassifier(dimIn, bias = bias, useZeroCoeff = True)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_estimateInitialMixtureOfTwoExperts(self, eps = 1e-8, numDists = 3):
        if self.deepTest:
            for distIndex in range(numDists):
                dimIn = randint(1, 5)
                dist, inputGen = gen_MixtureOfTwoExperts(dimIn, bias = True)
                def train(training):
                    def accumulate(acc):
                        for input, output, occ in training:
                            acc.add(input, output, occ)
                    acc = d.LinearGaussianAcc(inputLength = dimIn)
                    accumulate(acc)
                    initDist, initLogLike, initOcc = acc.estimateInitialMixtureOfTwoExperts()
                    return trn.trainEM(initDist, accumulate, deltaThresh = 1e-9)
                check_est(dist, train, inputGen, hasParams = True)

    def test_MixtureDist(self, eps = 1e-8, numDists = 10, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_MixtureDist(dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                check_est(dist, getTrainEM(dist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(dist), inputGen, hasParams = True)

    def test_IdentifiableMixtureDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_IdentifiableMixtureDist(dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True)
            if self.deepTest:
                initEstDist = gen_IdentifiableMixtureDist(dimIn, blcUseZeroCoeff = True)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_VectorDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            order = randint(0, 5)
            depth = randint(0, 5)
            dist, inputGen = gen_VectorDist(order, depth)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints)
            if self.deepTest:
                initEstDist = gen_VectorDist(order, depth)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_DiscreteDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            keys = list('abcde')[:randint(1, 5)]
            dimIn = randint(0, 5)
            dist, inputGen = gen_DiscreteDist(keys, dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_DiscreteDist(keys, dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_DecisionTree(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_DecisionTree_with_LinearGaussian_leaves(splitProb = 0.49, dimIn = dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_decisionTreeCluster(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_DecisionTree_with_LinearGaussian_leaves(splitProb = 0.49, dimIn = dimIn)
            def train(training):
                acc = d.AutoGrowingDiscreteAcc(createAcc = lambda: d.LinearGaussianAcc(inputLength = dimIn))
                totalOcc = 0.0
                for input, output, occ in training:
                    acc.add(input, output, occ)
                    totalOcc += occ
                mdlThresh = 0.5 * (dimIn + 1) * math.log(totalOcc + 1.0)
                return acc.decisionTreeCluster(test_dist_questions.getQuestionGroups(), thresh = mdlThresh, minOcc = 0.0, verbosity = 0)
            if True:
                # check decision tree clustering runs at all
                training = [ (input, dist.synth(input), math.exp(randn())) for input, index in zip(inputGen, range(numPoints)) ]
                estDist, estLogLike, estOcc = train(training)
            if self.deepTest:
                check_est(dist, train, inputGen, hasParams = True)

    def test_MappedInputDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            if randint(0, 2) == 0:
                dimOut = dimIn
            else:
                dimOut = randint(0, 5)
            dist, inputGen = gen_MappedInputDist(dimIn, dimOut)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_MappedOutputDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimInput = randint(0, 5)
            dist, inputGen = gen_MappedOutputDist(dimInput)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_TransformedInputDist(self, eps = 1e-8, numDists = 10, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            if randint(0, 2) == 0:
                dimOut = dimIn
            else:
                dimOut = randint(0, 5)
            dist, inputGen = gen_TransformedInputDist(dimIn, dimOut)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_TransformedOutputDist(self, eps = 1e-8, numDists = 10, numPoints = 100):
        for distIndex in range(numDists):
            dimInput = randint(0, 5)
            dist, inputGen = gen_TransformedOutputDist(dimInput)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_nestedTransformDist(self, eps = 1e-8, numDists = 10, numPoints = 100):
        for distIndex in range(numDists):
            numInputs = randint(1, 4)
            dimInputs = [ randint(0, 5) for i in range(numInputs) ]
            dist, inputGen = gen_nestedTransformDist(dimInputs)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_PassThruDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_PassThruDist(dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_PassThruDist(dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_DebugDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            maxOcc = random.choice([0, 1, 10, 100, None])
            dimIn = randint(0, 5)
            dist, inputGen = gen_DebugDist(maxOcc = maxOcc, dimIn = dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, iid = True, unitOcc = True, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_DebugDist(maxOcc = maxOcc, dimIn = dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True, iid = True, unitOcc = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True, iid = True, unitOcc = True)

    # FIXME : fix code to make this test pass!
    def test_shared_DiscreteDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            keys = list('abcde')[:randint(1, 5)]
            dimIn = randint(0, 5)
            dist, inputGen = gen_shared_DiscreteDist(keys, dimIn)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_shared_DiscreteDist(keys, dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    # FIXME : add more tests for shared dists

    # FIXME : add AutoregressiveSequenceDist test

    # FIXME : why is this _so_ slow? (should increase numDists or numPoints if possible)
    def test_AutoregressiveNetDist(self, eps = 1e-8, numDists = 20, numPoints = 5):
        def checkAdditional(dist, input, outSeq, eps):
            # check result of getTimedNet is topologically sorted
            timedNet = dist.getTimedNet(input, outSeq)
            assert wnet.netIsTopSorted(timedNet, wnet.nodeSetCompute(timedNet, accessibleOnly = False), deltaTime = lambda label: 0)
        for distIndex in range(numDists):
            depth = randint(0, 5)
            if randBool():
                dist, inputGen = gen_constant_AutoregressiveNetDist(depth = depth)
            else:
                dist, inputGen = gen_inSeq_AutoregressiveNetDist(depth = depth)
            # (FIXME : add logProbDerivOutputCheck once implemented)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, checkAdditional = checkAdditional)
            if self.deepTest:
                check_est(dist, getTrainEM(dist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(dist), inputGen, hasParams = True)

class DeepTestDist(TestDist):
    def setUp(self):
        self.deepTest = True

# FIXME : this is nowhere near a proper unit test (need to make it more robust, automated, etc)
def testBinaryLogisticClassifier():
    def inputGen(num):
        for i in range(num):
            yield np.append(randn(dim), 1.0)

    dim = 2
    blcTrue = d.BinaryLogisticClassifier(randn(dim + 1))
    num = 10000
    trainData = list((input, blcTrue.synth(input)) for input in inputGen(num))
    def accumulate(acc):
        for input, output in trainData:
            acc.add(input, output)

    blc = d.BinaryLogisticClassifier(np.zeros([dim + 1]))
    blc, trainLogLike, trainOcc = trn.trainEM(blc, accumulate, deltaThresh = 1e-10, minIterations = 10, verbosity = 2)
    print 'training log likelihood =', trainLogLike / trainOcc, '('+str(trainOcc)+' frames)'
    trainLogProb, trainOcc = logProb_frames(blc, trainData)
    print 'train set log prob =', trainLogProb / trainOcc, '('+str(trainOcc)+' frames)'

    print
    print 'DEBUG: (training data set size is', len(trainData), 'of which:'
    print 'DEBUG:    count(0) =', len([ input for input, output in trainData if output == 0 ])
    print 'DEBUG:    count(1) =', len([ input for input, output in trainData if output == 1 ])
    print 'DEBUG: )'
    print
    print 'true coeff =', blcTrue.coeff
    print 'estimated coeff =', blc.coeff
    dist = la.norm(blcTrue.coeff - blc.coeff)
    print '(Euclidean distance =', dist, ')'

    if dist > 0.1:
        sys.stderr.write('WARNING: unusually large discrepancy between estimated and true dist during BinaryLogisticClassifier test\n')

# (N.B. not a unit test. Just draws pictures to help you assess whether results seem reasonable.)
def testBinaryLogisticClassifierFunGraph():
    import pylab

    def location(blc):
        coeff = blc.coeff
        w = coeff[:-1]
        w0 = coeff[-1]
        mag = la.norm(w)
        normal = w / mag
        perpDist = -w0 / mag
        bdyPoint = normal * perpDist
        bdyPointProb = blc.prob(np.append(bdyPoint, 1.0), 0)
        if abs(bdyPointProb - 0.5) > 1e-10:
            raise RuntimeError('value at bdyPoint should be 0.5 but is '+str(bdyPointProb))
        return mag, normal, bdyPoint
    dim = 2
    wTrue = randn(dim + 1)
    blcTrue = d.BinaryLogisticClassifier(wTrue)
    print 'DEBUG: wTrue =', wTrue
    print
    def inputGen(num):
        for i in range(num):
            yield np.append(randn(dim), 1.0)

    num = 3000
    trainData = list((input, blcTrue.synth(input)) for input in inputGen(num))
    def accumulate(acc):
        for input, output in trainData:
            acc.add(input, output)
    print 'DEBUG: (in training data:'
    print 'DEBUG:    count(0) =', len([ input for input, output in trainData if output == 0 ])
    print 'DEBUG:    count(1) =', len([ input for input, output in trainData if output == 1 ])
    print 'DEBUG: )'

    def plotBdy(blc):
        mag, normal, bdyPoint = location(blc)
        dir = np.array([normal[1], -normal[0]])
        if abs(np.dot(dir, normal)) > 1e-10:
            raise RuntimeError('dir and normal are not perpendicular (should never happen)')
        xBdy, yBdy = zip(*[bdyPoint - 5 * dir, bdyPoint + 5 * dir])
        xBdy0, yBdy0 = zip(*[bdyPoint - normal / mag - 5 * dir, bdyPoint - normal / mag + 5 * dir])
        xBdy1, yBdy1 = zip(*[bdyPoint + normal / mag - 5 * dir, bdyPoint + normal / mag + 5 * dir])
        pylab.plot(xBdy, yBdy, 'k-', xBdy0, yBdy0, 'r-', xBdy1, yBdy1, 'b-')

    def plotData():
        x0, y0 = zip(*[ input[:-1] for input, output in trainData if output == 0 ])
        x1, y1 = zip(*[ input[:-1] for input, output in trainData if output == 1 ])
        pylab.plot(x0, y0, 'r+', x1, y1, 'bx')
        pylab.xlim(-3.5, 3.5)
        pylab.ylim(-3.5, 3.5)
        pylab.xlabel('x')
        pylab.ylabel('y')
        pylab.grid(True)

    def afterEst(dist, it):
        plotBdy(dist)

    plotData()
    plotBdy(blcTrue)

    blc = d.BinaryLogisticClassifier(np.zeros([dim + 1]))
    blc, trainLogLike, trainOcc = trn.trainEM(blc, accumulate, deltaThresh = 1e-4, minIterations = 10, maxIterations = 50, afterEst = afterEst, verbosity = 2)
    print 'DEBUG: w estimated final =', blc.coeff
    print 'training log likelihood =', trainLogLike / trainOcc, '('+str(trainOcc)+' frames)'
    trainLogProb, trainOcc = logProb_frames(blc, trainData)
    print 'train set log prob =', trainLogProb / trainOcc, '('+str(trainOcc)+' frames)'

    pylab.show()

# (N.B. not a unit test. Just draws pictures to help you assess whether results seem reasonable.)
def testMixtureOfTwoExpertsInitialization():
    import pylab

    def location(blc):
        coeff = blc.coeff
        w = coeff[:-1]
        w0 = coeff[-1]
        mag = la.norm(w)
        normal = w / mag
        perpDist = -w0 / mag
        bdyPoint = normal * perpDist
        bdyPointProb = blc.prob(np.append(bdyPoint, 1.0), 0)
        if abs(bdyPointProb - 0.5) > 1e-10:
            raise RuntimeError('value at bdyPoint should be 0.5 but is '+str(bdyPointProb))
        return mag, normal, bdyPoint
    dim = 2
    def inputGen(num):
        numClasses = 1
        transform = []
        bias = []
        for classIndex in range(numClasses):
            transform.append(randn(dim, dim) * 0.5)
            bias.append(randn(dim) * 3.0)
        for i in range(num):
            classIndex = randint(0, numClasses)
            yield np.append(np.dot(transform[classIndex], randn(dim)) + bias[classIndex], 1.0)

    num = 10000
    trainData = list((input, randn()) for input in inputGen(num))
    def accumulate(acc):
        for input, output in trainData:
            acc.add(input, output)

    def plotBdy(blc):
        mag, normal, bdyPoint = location(blc)
        dir = np.array([normal[1], -normal[0]])
        if abs(np.dot(dir, normal)) > 1e-10:
            raise RuntimeError('dir and normal are not perpendicular (should never happen)')
        xBdy, yBdy = zip(*[bdyPoint - 5 * dir, bdyPoint + 5 * dir])
        xBdy0, yBdy0 = zip(*[bdyPoint - normal / mag - 5 * dir, bdyPoint - normal / mag + 5 * dir])
        xBdy1, yBdy1 = zip(*[bdyPoint + normal / mag - 5 * dir, bdyPoint + normal / mag + 5 * dir])
        xDir, yDir = zip(*[bdyPoint - 5 * normal, bdyPoint + 5 * normal])
        pylab.plot(xBdy, yBdy, 'k-', xBdy0, yBdy0, 'r-', xBdy1, yBdy1, 'b-', xDir, yDir, 'g-')

    def plotData():
        x, y = zip(*[ input[:-1] for input, output in trainData ])
        pylab.plot(x, y, 'r+')
        pylab.xlabel('x')
        pylab.ylabel('y')
        pylab.grid(True)

    plotData()

    acc = d.LinearGaussianAcc(inputLength = dim + 1)
    accumulate(acc)
    dist, trainLogLike, trainOcc = acc.estimateInitialMixtureOfTwoExperts()
    blc = dist.classDist
    plotBdy(blc)
    print 'DEBUG: w estimated final =', blc.coeff

    pylab.xlim(-10.0, 10.0)
    pylab.ylim(-10.0, 10.0)
    pylab.show()

def suite(deepTest = False):
    if deepTest:
        return unittest.TestLoader().loadTestsFromTestCase(DeepTestDist)
    else:
        return unittest.TestLoader().loadTestsFromTestCase(TestDist)

if __name__ == '__main__':
    unittest.main()
