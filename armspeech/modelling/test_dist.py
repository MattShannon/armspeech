"""Unit tests for distributions, accumulators and model training."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import unittest
import logging
from collections import deque
import math
import random
import numpy as np
from numpy.random import randn, randint
import numpy.linalg as la
from scipy import stats
import cPickle as pickle
import string

from codedep import codeDeps, ForwardRef
from bisque import persist

from armspeech.modelling import nodetree
import armspeech.modelling.dist as d
import armspeech.modelling.train as trn
from armspeech.modelling import summarizer
import armspeech.modelling.transform as xf
from armspeech.modelling import cluster
from armspeech.modelling import wnet
from armspeech.util.mathhelp import logSum
from armspeech.util.iterhelp import chunkList
from armspeech.util.mathhelp import assert_allclose
from armspeech.util.mathhelp import AsArray
from armspeech.util.util import MapElem
from armspeech.modelling import test_transform_questions
from armspeech.modelling import test_transform
import armspeech.numpy_settings

@codeDeps()
def randBool():
    return randint(0, 2) == 0

@codeDeps()
def randTag():
    return 'tag'+str(randint(0, 1000000))

@codeDeps()
def randUttId():
    return 'utt'+str(randint(0, 1000000))

@codeDeps()
def simpleInputGen(dimIn, bias = False):
    while True:
        ret = randn(dimIn)
        if bias:
            ret[-1] = 1.0
        yield ret

@codeDeps()
def simpleVecInputGen(order, dimIn, bias = False):
    while True:
        ret = randn(dimIn, order)
        if bias:
            ret[-1] = 1.0
        yield ret

# (FIXME : add tests to test full range of shapes for transform stuff)
# (FIXME : add tests for Transformed(Input|Output)Learn(Dist|Transform)AccEM (for the Transform ones, have to first add a transform that can be re-estimated using EM))
# (FIXME : deep test for Transformed(Input|Output)Dist doesn't seem to converge to close to true dist in terms of parameters. Multiple local minima? Or just very insensitive to details? For more complicated transforms might the test procedure never converge?)

@codeDeps(d.LinearGaussian, randBool, randTag, simpleInputGen)
def gen_LinearGaussian(dimIn = 3, bias = False):
    coeff = randn(dimIn)
    varianceFloor = 0.0 if randBool() else math.exp(randn()) * 0.01
    variance = math.exp(randn()) + varianceFloor
    dist = d.LinearGaussian(coeff, variance, varianceFloor).withTag(randTag())
    numFloored, numFlooredDenom = dist.flooredSingle()
    return dist, simpleInputGen(dimIn, bias = bias)

@codeDeps(d.LinearGaussianVec, randBool, randTag, simpleVecInputGen)
def gen_LinearGaussianVec(order = 10, dimIn = 3, bias = False):
    coeffVec = randn(order, dimIn)
    varianceFloorVec = np.array([
        (0.0 if randBool() else math.exp(randn()) * 0.01)
        for _ in range(order)
    ])
    varianceVec = np.exp(randn(order)) + varianceFloorVec
    dist = d.LinearGaussianVec(coeffVec, varianceVec,
                               varianceFloorVec).withTag(randTag())
    numFloored, numFlooredDenom = dist.flooredSingle()
    return dist, simpleVecInputGen(order, dimIn, bias = bias)

@codeDeps(d.GaussianVec, randBool, randTag, simpleInputGen)
def gen_GaussianVec(order = 10):
    meanVec = randn(order)
    varianceFloorVec = np.array([
        (0.0 if randBool() else math.exp(randn()) * 0.01)
        for _ in range(order)
    ])
    varianceVec = np.exp(randn(order)) + varianceFloorVec
    dist = d.GaussianVec(meanVec, varianceVec,
                         varianceFloorVec).withTag(randTag())
    numFloored, numFlooredDenom = dist.flooredSingle()
    return dist, simpleInputGen(order)

@codeDeps(d.StudentDist, randTag, simpleInputGen)
def gen_StudentDist(dimIn = 3):
    df = math.exp(randn() + 1.0)
    precision = math.exp(randn())
    dist = d.StudentDist(df, precision).withTag(randTag())
    return dist, simpleInputGen(dimIn)

@codeDeps(d.ConstantClassifier, randBool, randTag, simpleInputGen)
def gen_ConstantClassifier(numClasses = 5):
    if randBool():
        probFloors = np.zeros((numClasses,))
        probs = np.exp(randn(numClasses))
        probs = probs / sum(probs)
    else:
        probFloors = np.exp(randn(numClasses))
        probFloors = probFloors / sum(probFloors) * 0.01
        probExtras = np.exp(randn(numClasses))
        probExtras = probExtras / sum(probExtras) * 0.99
        probs = probExtras + probFloors
    dist = d.ConstantClassifier(probs, probFloors).withTag(randTag())
    numFloored, numFlooredDenom = dist.flooredSingle()
    return dist, simpleInputGen(0)

@codeDeps(d.BinaryLogisticClassifier, randBool, randTag, simpleInputGen)
def gen_BinaryLogisticClassifier(dimIn = 3, bias = False, useZeroCoeff = False):
    coeffFloor = np.ones((dimIn,)) * (float('inf') if randBool() else 5.0)
    if useZeroCoeff:
        coeff = np.zeros((dimIn,))
    else:
        coeff = randn(dimIn)
        coeff = np.minimum(coeff, coeffFloor)
        coeff = np.maximum(coeff, -coeffFloor)
    dist = d.BinaryLogisticClassifier(coeff, coeffFloor).withTag(randTag())
    numFloored, numFlooredDenom = dist.flooredSingle()
    return dist, simpleInputGen(dimIn, bias = bias)

@codeDeps(gen_BinaryLogisticClassifier, gen_ConstantClassifier, randBool)
def gen_classifier(numClasses, dimIn, bias = False):
    """Generates a random classifier with vector input."""
    if numClasses == 2 and randBool():
        return gen_BinaryLogisticClassifier(dimIn = dimIn, bias = bias)
    else:
        return gen_ConstantClassifier(numClasses)

@codeDeps(ForwardRef(lambda: gen_MixtureOfTwoExperts))
def gen_MixtureDist(dimIn):
    return gen_MixtureOfTwoExperts(dimIn = 3)

@codeDeps(d.MixtureDist, gen_BinaryLogisticClassifier, gen_LinearGaussian,
    randBool, randTag
)
def gen_MixtureOfTwoExperts(dimIn = 3, bias = False):
    blc, blcGen = gen_BinaryLogisticClassifier(dimIn, bias = bias)
    dist0 = gen_LinearGaussian(dimIn)[0]
    dist1 = gen_LinearGaussian(dimIn)[0]
    dist = d.MixtureDist(blc, [dist0, dist1],
                         hardMean = randBool()).withTag(randTag())
    return dist, blcGen

@codeDeps(d.FixedValueDist, d.IdentifiableMixtureDist,
    gen_BinaryLogisticClassifier, gen_LinearGaussian, randTag
)
def gen_IdentifiableMixtureDist(dimIn = 3, blcUseZeroCoeff = False):
    blc, blcGen = gen_BinaryLogisticClassifier(dimIn, useZeroCoeff = blcUseZeroCoeff)
    dist0 = d.FixedValueDist(None)
    dist1 = gen_LinearGaussian(dimIn)[0]
    dist = d.IdentifiableMixtureDist(blc, [dist0, dist1]).withTag(randTag())
    return dist, blcGen

@codeDeps(AsArray, d.MappedInputDist, gen_LinearGaussian, randTag,
    summarizer.VectorSeqSummarizer
)
def gen_VectorDist(order = 10, depth = 3):
    depths = dict([ (outIndex, depth) for outIndex in range(order) ])
    vectorSummarizer = summarizer.VectorSeqSummarizer(order, depths)
    dist = vectorSummarizer.createDist(False, lambda outIndex:
        d.MappedInputDist(AsArray(),
            gen_LinearGaussian(depths[outIndex])[0]
        )
    ).withTag(randTag())
    def getInputGen():
        while True:
            yield randn(depth, order)
    return dist, getInputGen()

@codeDeps(d.createDiscreteDist, gen_LinearGaussian, randTag)
def gen_DiscreteDist(keys = ['a', 'b', 'c'], dimIn = 3):
    dist = d.createDiscreteDist(keys, lambda key:
        gen_LinearGaussian(dimIn)[0]
    ).withTag(randTag())
    def getInputGen():
        while True:
            yield random.choice(keys), randn(dimIn)
    return dist, getInputGen()

@codeDeps(d.createDiscreteDist, gen_LinearGaussian, randTag)
def gen_shared_DiscreteDist(keys = ['a', 'b', 'c'], dimIn = 3):
    subDist = gen_LinearGaussian(dimIn)[0]
    dist = d.createDiscreteDist(keys, lambda key:
        subDist
    ).withTag(randTag())
    def getInputGen():
        while True:
            yield random.choice(keys), randn(dimIn)
    return dist, getInputGen()

@codeDeps(MapElem, d.MappedInputDist, d.createDiscreteDist, gen_LinearGaussian,
    randTag, test_transform.gen_DecisionTree,
    test_transform_questions.SimplePhoneset,
    test_transform_questions.getQuestionGroups
)
def gen_DecisionTree_with_LinearGaussian_leaves(splitProb = 0.49, dimIn = 3):
    phoneset = test_transform_questions.SimplePhoneset()
    labels = phoneset.phoneList
    questionGroups = test_transform_questions.getQuestionGroups(phoneset)

    decTree = test_transform.gen_DecisionTree(questionGroups, labels,
                                              splitProb = splitProb)
    dist = d.MappedInputDist(MapElem(0, 2, decTree),
        d.createDiscreteDist(range(decTree.numLeaves), lambda key:
            gen_LinearGaussian(dimIn)[0]
        )
    ).withTag(randTag())

    def getInputGen():
        while True:
            yield random.choice(labels), randn(dimIn)

    return dist, getInputGen()

@codeDeps(d.MappedInputDist, gen_LinearGaussian, randTag, simpleInputGen,
    test_transform.gen_genericTransform
)
def gen_MappedInputDist(dimIn = 3, dimOut = 2):
    transform = test_transform.gen_genericTransform([dimIn], [dimOut])
    subDist = gen_LinearGaussian(dimOut)[0]
    return d.MappedInputDist(transform, subDist).withTag(randTag()), simpleInputGen(dimIn)

@codeDeps(d.MappedOutputDist, gen_LinearGaussian, randTag,
    test_transform.gen_genericOutputTransform
)
def gen_MappedOutputDist(dimInput = 3):
    outputTransform = test_transform.gen_genericOutputTransform([dimInput], [])
    subDist, inputGen = gen_LinearGaussian(dimInput)
    return d.MappedOutputDist(outputTransform, subDist).withTag(randTag()), inputGen

@codeDeps(d.TransformedInputDist, gen_LinearGaussian, randTag, simpleInputGen,
    test_transform.gen_genericTransform
)
def gen_TransformedInputDist(dimIn = 3, dimOut = 2):
    transform = test_transform.gen_genericTransform([dimIn], [dimOut])
    subDist = gen_LinearGaussian(dimOut)[0]
    return d.TransformedInputDist(transform, subDist).withTag(randTag()), simpleInputGen(dimIn)

@codeDeps(d.TransformedOutputDist, gen_LinearGaussian, randTag,
    test_transform.gen_genericOutputTransform
)
def gen_TransformedOutputDist(dimInput = 3):
    outputTransform = test_transform.gen_genericOutputTransform([dimInput], [])
    subDist, inputGen = gen_LinearGaussian(dimInput)
    return d.TransformedOutputDist(outputTransform, subDist).withTag(randTag()), inputGen

@codeDeps(d.MappedInputDist, d.MappedOutputDist, d.TransformedInputDist,
    d.TransformedOutputDist, gen_LinearGaussian, randBool, randTag,
    simpleInputGen, test_transform.gen_genericOutputTransform,
    test_transform.gen_genericTransform
)
def gen_nestedTransformDist(dimInputs = [3, 4, 2]):
    assert len(dimInputs) >= 1
    dimIn = dimInputs[-1]
    dist = gen_LinearGaussian(dimIn)[0]
    if randBool():
        outputTransform = test_transform.gen_genericOutputTransform([dimIn], [])
        if randBool():
            dist = d.MappedOutputDist(outputTransform, dist)
        else:
            dist = d.TransformedOutputDist(outputTransform, dist)
    for dimIn, dimOut in reversed(zip(dimInputs, dimInputs[1:])):
        transform = test_transform.gen_genericTransform([dimIn], [dimOut])
        if randBool():
            dist = d.MappedInputDist(transform, dist)
        else:
            dist = d.TransformedInputDist(transform, dist)
        if randBool():
            outputTransform = test_transform.gen_genericOutputTransform([dimIn], [])
            if randBool():
                dist = d.MappedOutputDist(outputTransform, dist)
            else:
                dist = d.TransformedOutputDist(outputTransform, dist)
    return dist.withTag(randTag()), simpleInputGen(dimInputs[0])

@codeDeps(d.PassThruDist, gen_LinearGaussian, randTag)
def gen_PassThruDist(dimIn = 3):
    subDist, inputGen = gen_LinearGaussian(dimIn)
    return d.PassThruDist(subDist).withTag(randTag()), inputGen

@codeDeps(d.CountFramesDist, gen_VectorDist, randTag)
def gen_CountFramesDist(framesPerObs):
    subDist, inputGen = gen_VectorDist(order = framesPerObs)
    return d.CountFramesDist(subDist).withTag(randTag()), inputGen

@codeDeps(d.DebugDist, gen_LinearGaussian, randTag)
def gen_DebugDist(maxOcc = None, dimIn = 3):
    subDist, inputGen = gen_LinearGaussian(dimIn)
    return d.DebugDist(maxOcc, subDist).withTag(randTag()), inputGen

@codeDeps(d.MappedInputDist, gen_LinearGaussian, xf.AddBias)
def gen_autoregressive_dist(depth = 2):
    dist = d.MappedInputDist(xf.AddBias(),
        gen_LinearGaussian(dimIn = depth + 1)[0]
    )
    return dist, None
@codeDeps()
def autoregressive_1D_is_stable(dist, depth, starts = 5, stepsIntoFuture = 100, bigThresh = 1e6):
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
@codeDeps(autoregressive_1D_is_stable, gen_autoregressive_dist)
def gen_stable_autoregressive_dist(depth = 2):
    while True:
        dist = gen_autoregressive_dist(depth)[0]
        if autoregressive_1D_is_stable(dist, depth):
            break
    return dist, None

@codeDeps(d.AutoregressiveSequenceDist, d.createDiscreteDist,
    gen_stable_autoregressive_dist, randTag, randUttId, xf.IdentityTransform
)
def gen_AutoregressiveSequenceDist(depth = 2):
    labels = string.lowercase[:randint(1, 10)]
    acDist = d.createDiscreteDist(labels, lambda label:
        gen_stable_autoregressive_dist(depth)[0]
    )
    dist = d.AutoregressiveSequenceDist(depth, xf.IdentityTransform(), [ 0.0 for i in range(depth) ], acDist).withTag(randTag())

    def getInputGen():
        while True:
            labelSeq = [ random.choice(labels) for i in range(randint(0, 4)) ]
            inSeq = [ label for label in labelSeq for i in range(randint(1, 4)) ]
            yield randUttId(), inSeq
    inputGen = getInputGen()
    return dist, getInputGen()

@codeDeps(wnet.ConcreteNet)
def add_autoregressive_style_labels(concreteNet, genLabels):
    net = concreteNet
    numNodes = net.numNodes
    edgesForwards = [ [] for node in range(numNodes) ]
    for node in range(numNodes):
        nextNodes = [ nextNode for label, nextNode in net.next(node, forwards = True) ]
        labels = genLabels(numLabels = len(nextNodes))
        edgesForwards[node] = zip(labels, nextNodes)
    return wnet.ConcreteNet(startNode = net.startNode, endNode = net.endNode, elems = net.elems, edgesForwards = edgesForwards)
@codeDeps(add_autoregressive_style_labels, randBool, wnet.ConcreteNet,
    wnet.checkConsistent, wnet.concretizeNet, wnet.nodeSetCompute
)
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
@codeDeps(gen_autoregressive_style_net, randBool, wnet.nodeSetCompute)
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

@codeDeps(d.AutoregressiveNetDist, d.MappedInputDist, d.SimplePruneSpec,
    d.createDiscreteDist, gen_classifier, gen_simple_autoregressive_style_net,
    gen_stable_autoregressive_dist, randBool, randTag, randUttId,
    wnet.nodeSetCompute, xf.AddBias, xf.ConstantTransform
)
def gen_constant_AutoregressiveNetDist(depth = 2):
    """Generates an AutoregressiveNetDist which is independent of input."""
    numSubLabels = randint(1, 5)
    while True:
        net, phInputsAc, phInputToNumClassesDur = gen_simple_autoregressive_style_net('g', acSubLabels = range(numSubLabels), durSubLabels = range(numSubLabels))
        nodeSet = wnet.nodeSetCompute(net, accessibleOnly = True)
        numEmitting = len([ node for node in nodeSet if net.elem(node) is not None ])
        if numEmitting >= 2 or numEmitting == 1 and randint(0, 4) == 0 or numEmitting == 0 and randint(0, 10) == 0:
            break
    durDist = d.createDiscreteDist(phInputToNumClassesDur.keys(), lambda phInput:
        d.MappedInputDist(xf.AddBias(),
            gen_classifier(numClasses = phInputToNumClassesDur[phInput], dimIn = depth + 1)[0]
        )
    )
    acDist = d.createDiscreteDist(list(phInputsAc), lambda phInput:
        gen_stable_autoregressive_dist(depth)[0]
    )
    pruneSpec = None if randBool() else d.SimplePruneSpec(betaThresh = (None if randBool() else 1000.0), logOccThresh = (None if randBool() else 1000.0))
    dist = d.AutoregressiveNetDist(depth, xf.ConstantTransform(net), [ 0.0 for i in range(depth) ], durDist, acDist, pruneSpec).withTag(randTag())

    def getInputGen():
        while True:
            yield randUttId(), ''
    return dist, getInputGen()

@codeDeps(d.AutoregressiveNetDist, d.MappedInputDist, d.SimpleLeftToRightNetFor,
    d.SimplePruneSpec, d.createDiscreteDist, gen_classifier,
    gen_stable_autoregressive_dist, randBool, randTag, randUttId, xf.AddBias
)
def gen_inSeq_AutoregressiveNetDist(depth = 2):
    """Generates a left-to-right AutoregressiveNetDist where input is a label sequence."""
    labels = string.lowercase[:randint(1, 10)]
    numSubLabels = randint(1, 5)
    subLabels = list(range(numSubLabels))
    labelledSubLabels = [ (label, subLabel) for label in labels for subLabel in subLabels ]
    durDist = d.createDiscreteDist(labelledSubLabels, lambda (label, subLabel):
        d.MappedInputDist(xf.AddBias(),
            gen_classifier(numClasses = 2, dimIn = depth + 1)[0]
        )
    )
    acDist = d.createDiscreteDist(labelledSubLabels, lambda (label, subLabel):
        gen_stable_autoregressive_dist(depth)[0]
    )
    pruneSpec = None if randBool() else d.SimplePruneSpec(betaThresh = (None if randBool() else 1000.0), logOccThresh = (None if randBool() else 1000.0))
    dist = d.AutoregressiveNetDist(depth, d.SimpleLeftToRightNetFor(subLabels), [ 0.0 for i in range(depth) ], durDist, acDist, pruneSpec).withTag(randTag())

    def getInputGen():
        while True:
            labelSeq = [ random.choice(labels) for i in range(randint(0, 4)) ]
            yield randUttId(), labelSeq
    return dist, getInputGen()

@codeDeps(d.SynthSeqTooLongError)
def restrictTypicalOutputLength(genDist, maxLength = 20, numPoints = 100):
    bad = True
    while bad:
        dist, inputGen = genDist()
        bad = False
        for i in range(numPoints):
            input = inputGen.next()
            try:
                output = dist.synth(input, maxLength = maxLength)
            except d.SynthSeqTooLongError, e:
                bad = True
                break
    return dist, inputGen

@codeDeps(nodetree.nodeList)
def dagInfoExtract(parentNode):
    """Extracts basic structure information about the DAG for a parentNode."""
    return [ type(node) for node in nodetree.nodeList(parentNode) ]

@codeDeps()
def iidLogProb(dist, training):
    logProb = 0.0
    for input, output, occ in training:
        logProb += dist.logProb(input, output) * occ
    return logProb

@codeDeps(d.getDefaultCreateAcc)
def trainedAcc(dist, training):
    acc = d.getDefaultCreateAcc()(dist)
    for input, output, occ in training:
        acc.add(input, output, occ)
    return acc

@codeDeps(d.getDefaultParamSpec)
def trainedAccG(dist, training, ps = d.getDefaultParamSpec()):
    acc = ps.createAccG(dist)
    for input, output, occ in training:
        acc.add(input, output, occ)
    return acc

@codeDeps(d.getDefaultParamSpec)
def randomizeParams(dist, ps = d.getDefaultParamSpec()):
    return ps.parseAll(dist, randn(*np.shape(ps.params(dist))))

@codeDeps(assert_allclose, dagInfoExtract)
def reparse(dist, ps):
    params = ps.params(dist)
    assert len(np.shape(params)) == 1
    distParsed = ps.parseAll(dist, params)
    paramsParsed = ps.params(distParsed)
    assert_allclose(paramsParsed, params)
    assert dist.tag == distParsed.tag
    assert dagInfoExtract(distParsed) == dagInfoExtract(dist)
    return distParsed

@codeDeps(assert_allclose)
def check_logProbDerivInput(dist, input, output, eps):
    inputDirection = randn(*np.shape(input))
    numericDeriv = (
        dist.logProb(input + inputDirection * eps, output) -
        dist.logProb(input - inputDirection * eps, output)
    ) / (eps * 2.0)
    analyticDeriv = np.sum(inputDirection * dist.logProbDerivInput(input, output))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)

@codeDeps(assert_allclose)
def check_logProbDerivInput_hasDiscrete(dist, (disc, input), output, eps):
    inputDirection = randn(*np.shape(input))
    numericDeriv = (
        dist.logProb((disc, input + inputDirection * eps), output) -
        dist.logProb((disc, input - inputDirection * eps), output)
    ) / (eps * 2.0)
    analyticDeriv = np.sum(inputDirection * dist.logProbDerivInput((disc, input), output))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)

@codeDeps(assert_allclose)
def check_logProbDerivOutput(dist, input, output, eps):
    outputDirection = randn(*np.shape(output))
    numericDeriv = (
        dist.logProb(input, output + outputDirection * eps) -
        dist.logProb(input, output - outputDirection * eps)
    ) / (eps * 2.0)
    analyticDeriv = np.sum(outputDirection * dist.logProbDerivOutput(input, output))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)

@codeDeps(assert_allclose)
def check_logProbDerivOutput_hasDiscrete(dist, input, (disc, output), eps):
    outputDirection = randn(*np.shape(output))
    numericDeriv = (
        dist.logProb(input, (disc, output + outputDirection * eps)) -
        dist.logProb(input, (disc, output - outputDirection * eps))
    ) / (eps * 2.0)
    analyticDeriv = np.sum(outputDirection * dist.logProbDerivOutput(input, (disc, output)))
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-6, rtol = 1e-4)

@codeDeps(assert_allclose, chunkList, d.addAcc, trainedAccG)
def check_addAcc(dist, trainingAll, ps):
    accAll = trainedAccG(dist, trainingAll, ps = ps)
    occAll = accAll.occ
    countAll = accAll.count()
    logLikeAll = accAll.logLike()
    derivParamsAll = ps.derivParams(accAll)

    trainingParts = chunkList(trainingAll, numChunks = randint(1, 5))
    accs = [ trainedAccG(dist, trainingPart, ps = ps) for trainingPart in trainingParts ]
    accFull = accs[0]
    for acc in accs[1:]:
        d.addAcc(accFull, acc)
    occFull = accFull.occ
    countFull = accFull.count()
    logLikeFull = accFull.logLike()
    derivParamsFull = ps.derivParams(accFull)

    assert_allclose(occFull, occAll)
    assert_allclose(countFull, countAll)
    assert_allclose(logLikeFull, logLikeAll, atol = 1e-10)
    assert_allclose(derivParamsFull, derivParamsAll, atol = 1e-10)

@codeDeps(assert_allclose, iidLogProb, trainedAcc, trainedAccG)
def check_occ_and_logLike(dist, training, iid, hasEM):
    assert iid == True
    totOcc = sum([ occ for input, output, occ in training ])
    logLikeFromDist = iidLogProb(dist, training)
    if hasEM:
        acc = trainedAcc(dist, training)
        assert_allclose(acc.occ, totOcc)
        assert_allclose(acc.logLike(), logLikeFromDist, atol = 1e-10)
    acc = trainedAccG(dist, training)
    assert_allclose(acc.occ, totOcc)
    assert_allclose(acc.logLike(), logLikeFromDist, atol = 1e-10)

@codeDeps(assert_allclose, trainedAccG)
def check_derivParams(dist, training, ps, eps):
    params = ps.params(dist)
    acc = trainedAccG(dist, training, ps = ps)
    logLike = acc.logLike()
    derivParams = ps.derivParams(acc)
    paramsDirection = randn(*np.shape(params))
    distNew = ps.parseAll(dist, params + paramsDirection * eps)
    logLikeNew = trainedAccG(distNew, training, ps = ps).logLike()
    assert_allclose(ps.params(distNew), params + paramsDirection * eps)
    distNew2 = ps.parseAll(dist, params - paramsDirection * eps)
    logLikeNew2 = trainedAccG(distNew2, training, ps = ps).logLike()
    assert_allclose(ps.params(distNew2), params - paramsDirection * eps)

    numericDeriv = (logLikeNew - logLikeNew2) / (eps * 2.0)
    analyticDeriv = np.dot(derivParams, paramsDirection)
    assert_allclose(numericDeriv, analyticDeriv, atol = 1e-4, rtol = 1e-4)

@codeDeps(dagInfoExtract, trn.trainEM)
def getTrainEM(initEstDist, maxIterations = None, verbosity = 0):
    def doTrainEM(training):
        def accumulate(acc):
            for input, output, occ in training:
                acc.add(input, output, occ)
        dist = trn.trainEM(initEstDist, accumulate, deltaThresh = 1e-9, maxIterations = maxIterations, verbosity = verbosity)
        assert initEstDist.tag is not None
        assert dist.tag == initEstDist.tag
        assert dagInfoExtract(dist) == dagInfoExtract(initEstDist)
        return dist
    return doTrainEM

@codeDeps(d.getDefaultParamSpec, dagInfoExtract, trn.trainCG)
def getTrainCG(initEstDist, ps = d.getDefaultParamSpec(), length = -500, verbosity = 0):
    def doTrainCG(training):
        def accumulate(acc):
            for input, output, occ in training:
                acc.add(input, output, occ)
        dist = trn.trainCG(initEstDist, accumulate, ps = ps, length = length, verbosity = verbosity)
        assert initEstDist.tag is not None
        assert dist.tag == initEstDist.tag
        assert dagInfoExtract(dist) == dagInfoExtract(initEstDist)
        return dist
    return doTrainCG

@codeDeps(d.getDefaultEstimate)
def getTrainFromAcc(createAcc):
    def doTrainFromAcc(training):
        acc = createAcc()
        for input, output, occ in training:
            acc.add(input, output, occ)
        dist = d.getDefaultEstimate()(acc)
        assert acc.tag is not None
        assert dist.tag == acc.tag
        return dist
    return doTrainFromAcc

@codeDeps(assert_allclose, d.getDefaultParamSpec, iidLogProb, trainedAccG)
def check_est(trueDist, train, inputGen, hasParams, iid = True, unitOcc = False, ps = d.getDefaultParamSpec(), logLikeThresh = 2e-2, tslpThresh = 2e-2, testSetSize = 50, initTrainingSetSize = 100, trainingSetMult = 5, maxTrainingSetSize = 100000):
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
        totOcc = sum([ occ for input, output, occ in training ])
        estDist = train(training)
        logLikeTrue = iidLogProb(trueDist, training)
        logLikeEst = iidLogProb(estDist, training)
        tslpTrue = iidLogProb(trueDist, testSet)
        tslpEst = iidLogProb(estDist, testSet)
        if hasParams:
            newAcc = trainedAccG(estDist, training, ps = ps)
            derivParams = ps.derivParams(newAcc)
            assert_allclose(derivParams / totOcc, np.zeros([len(derivParams)]), atol = 1e-4)
        if math.isinf(logLikeEst):
            print 'NOTE: singularity in likelihood function (training set size =', len(training), ', occ '+repr(totOcc)+', estDist =', estDist, ', logLikeEst =', logLikeEst / totOcc, ')'
        if abs(logLikeTrue - logLikeEst) / totOcc < logLikeThresh and abs(tslpTrue - tslpEst) / testOcc < tslpThresh:
            converged = True
    if not converged:
        raise AssertionError('estimated dist did not converge to true dist\n\ttraining set size = '+str(len(training))+'\n\tlogLikeTrue = '+str(logLikeTrue / totOcc)+' vs logLikeEst = '+str(logLikeEst / totOcc)+'\n\ttslpTrue = '+str(tslpTrue / testOcc)+' vs tslpEst = '+str(tslpEst / testOcc)+'\n\ttrueDist = '+repr(trueDist)+'\n\testDist = '+repr(estDist))

@codeDeps(d.getDefaultCreateAcc, d.getDefaultEstimate)
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
            acc = d.getDefaultCreateAcc()(updatedDist)
            for input, output, occ in trainingSet:
                acc.add(input, output, occ)
            updatedDist = d.getDefaultEstimate()(acc)
            trainingSet.append((inputNew, updatedDist.synth(inputNew), 1.0))
    assert len(trainingSet) == trainingSetSize
    return trainingSet

@codeDeps(assert_allclose, check_addAcc, check_derivParams,
    check_logProbDerivInput, check_logProbDerivInput_hasDiscrete,
    check_logProbDerivOutput, check_logProbDerivOutput_hasDiscrete,
    check_occ_and_logLike, d.eval_local, d.getDefaultCreateAcc,
    d.getDefaultParamSpec, d.isolateDist, dagInfoExtract, getTrainCG,
    getTrainEM, getTrainingSet, persist.roundTrip, reparse, trainedAcc,
    trainedAccG
)
def checkLots(dist, inputGen, hasParams, eps, numPoints, iid = True, unitOcc = False, hasEM = True, evalShouldWork = True, ps = d.getDefaultParamSpec(), logProbDerivInputCheck = False, logProbDerivInput_hasDiscrete_check = False, logProbDerivOutputCheck = False, logProbDerivOutput_hasDiscrete_checkFor = lambda output: False, checkAdditional = None, checkAccAdditional = None):
    assert dist.tag is not None
    if hasEM:
        assert d.getDefaultCreateAcc()(dist).tag == dist.tag
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

    if True:
        distMapped = d.isolateDist(dist)
        assert id(distMapped) != id(dist)
        assert distMapped.tag == dist.tag
        assert dagInfoExtract(distMapped) == dagInfoExtract(dist)
        if hasParams:
            assert_allclose(ps.params(distMapped), ps.params(dist))
    if True:
        distFromPickle = persist.roundTrip(dist)
        assert id(distFromPickle) != id(dist)
        assert distFromPickle.tag == dist.tag
        assert dagInfoExtract(distFromPickle) == dagInfoExtract(dist)
        # checks that the operation of roundTripping from pickle to pickle is
        #   idempotent (this seems to usually be true, so we might as well
        #   verify it). (As it happens this property is required by secHash).
        assert pickle.dumps(persist.roundTrip(distFromPickle), protocol = 2) == pickle.dumps(distFromPickle, protocol = 2)
        if hasParams:
            assert_allclose(ps.params(distFromPickle), ps.params(dist))
    if True:
        distEvaled = d.eval_local(repr(dist))
        assert distEvaled.tag == dist.tag
        assert repr(dist) == repr(distEvaled)
        if evalShouldWork:
            assert dagInfoExtract(distEvaled) == dagInfoExtract(dist)
            if hasParams:
                assert_allclose(ps.params(distEvaled), ps.params(dist))
    if hasParams:
        distParsed = reparse(dist, ps)
    for input, output in points:
        if not math.isinf(dist.logProb(input, output)):
            if checkAdditional is not None:
                checkAdditional(dist, input, output, eps)
            lp = dist.logProb(input, output)
            if True:
                assert_allclose(distMapped.logProb(input, output), lp)
            if True:
                assert_allclose(distFromPickle.logProb(input, output), lp)
            if True:
                assert_allclose(distEvaled.logProb(input, output), lp)
            if hasParams:
                assert_allclose(distParsed.logProb(input, output), lp)
            if logProbDerivInputCheck:
                check_logProbDerivInput(dist, input, output, eps)
            if logProbDerivInput_hasDiscrete_check:
                check_logProbDerivInput_hasDiscrete(dist, input, output, eps)
            if logProbDerivOutputCheck:
                check_logProbDerivOutput(dist, input, output, eps)
            if logProbDerivOutput_hasDiscrete_checkFor(output):
                check_logProbDerivOutput_hasDiscrete(dist, input, output, eps)
        else:
            print 'NOTE: skipping point with logProb =', dist.logProb(input, output), 'for dist =', dist, 'input =', input, 'output =', output

    if hasParams:
        # (FIXME : add addAcc check for Accs which are not AccGs)
        check_addAcc(dist, training, ps)
    if iid:
        check_occ_and_logLike(dist, training, iid = iid, hasEM = hasEM)
    if hasParams:
        check_derivParams(dist, training, ps, eps = eps)
    if checkAccAdditional is not None:
        if hasEM:
            checkAccAdditional(trainedAcc(dist, training), training)
        checkAccAdditional(trainedAccG(dist, training), training)

    logProbsAfter = [ dist.logProb(input, output) for input, output in points ]
    assert_allclose(logProbsAfter, logProbsBefore, atol = 1e-10, msg = 'looks like parsing affected the original distribution, which should never happen')
    if hasParams:
        paramsAfter = ps.params(dist)
        assert_allclose(logProbsAfter, logProbsBefore, atol = 1e-10, msg = 'looks like parsing affected the original distribution, which should never happen')

    if hasEM:
        # check EM estimation runs at all (if there is a decent amount of data)
        if len(training) >= numPoints - 1:
            getTrainEM(dist, maxIterations = 1)(training)
    if True:
        # check CG estimation runs at all (if there is a decent amount of data)
        if len(training) >= numPoints - 1:
            getTrainCG(dist, length = -2)(training)

@codeDeps(assert_allclose, checkLots, check_est, cluster.ClusteringSpec,
    cluster.MdlUtilitySpec, cluster.decisionTreeCluster,
    cluster.decisionTreeClusterDepthBased, d.AutoGrowingDiscreteAcc,
    d.ConstantClassifierAcc, d.LinearGaussianAcc, d.Memo,
    d.estimateInitialMixtureOfTwoExperts, gen_AutoregressiveSequenceDist,
    gen_BinaryLogisticClassifier, gen_ConstantClassifier, gen_CountFramesDist,
    gen_DebugDist, gen_DecisionTree_with_LinearGaussian_leaves,
    gen_DiscreteDist, gen_GaussianVec, gen_IdentifiableMixtureDist,
    gen_LinearGaussian, gen_LinearGaussianVec, gen_MappedInputDist,
    gen_MappedOutputDist, gen_MixtureDist, gen_MixtureOfTwoExperts,
    gen_PassThruDist, gen_StudentDist, gen_TransformedInputDist,
    gen_TransformedOutputDist, gen_VectorDist,
    gen_constant_AutoregressiveNetDist, gen_inSeq_AutoregressiveNetDist,
    gen_nestedTransformDist, gen_shared_DiscreteDist, getTrainCG, getTrainEM,
    getTrainFromAcc, randBool, randTag, randomizeParams,
    restrictTypicalOutputLength, test_transform_questions.SimplePhoneset,
    test_transform_questions.getQuestionGroups, trn.trainEM,
    wnet.netIsTopSorted, wnet.nodeSetCompute
)
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
                        count[i] += 1.0
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
                createAcc = lambda: d.LinearGaussianAcc(inputLength = dimIn, varianceFloor = 0.0).withTag(randTag())
                check_est(dist, getTrainFromAcc(createAcc), inputGen, hasParams = True)

    def test_LinearGaussianVec(self, eps = 1e-8, numDists = 30, numPoints = 100):
        for distIndex in range(numDists):
            bias = random.choice([True, False])
            order = randint(0, 10)
            dimIn = randint(1 if bias else 0, 5)
            dist, inputGen = gen_LinearGaussianVec(order, dimIn, bias = bias)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_LinearGaussianVec(order, dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_GaussianVec(self, eps = 1e-8, numDists = 30, numPoints = 100):
        for distIndex in range(numDists):
            order = randint(0, 10)
            dist, inputGen = gen_GaussianVec(order)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_GaussianVec(order)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

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
                createAcc = lambda: d.ConstantClassifierAcc(numClasses = numClasses, probFloors = np.zeros((numClasses,))).withTag(randTag())
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

    def test_estimateInitialMixtureOfTwoExperts(self, eps = 1e-8, numDists = 3, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(1, 5)
            dist, inputGen = gen_MixtureOfTwoExperts(dimIn, bias = True)
            def train(training, maxIterations = None):
                def accumulate(acc):
                    for input, output, occ in training:
                        acc.add(input, output, occ)
                acc = d.LinearGaussianAcc(inputLength = dimIn, varianceFloor = 0.0)
                accumulate(acc)
                initDist = d.estimateInitialMixtureOfTwoExperts(acc)
                return trn.trainEM(initDist, accumulate, deltaThresh = 1e-9, maxIterations = maxIterations)
            if True:
                # check estimateInitialMixtureOfTwoExperts runs at all
                training = [ (input, dist.synth(input), math.exp(randn())) for input, index in zip(inputGen, range(numPoints)) ]
                estDist = train(training, maxIterations = 1)
            if self.deepTest:
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
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInputCheck = True, logProbDerivOutput_hasDiscrete_checkFor = lambda (comp, acOutput): acOutput is not None)
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
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivInput_hasDiscrete_check = True, logProbDerivOutputCheck = True)
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

    # (FIXME : add a separate, independent unit test for decisionTreeCluster
    #   (put in test_cluster.py))
    def test_AutoGrowingDiscreteAcc_and_decisionTreeCluster(self, eps = 1e-8, numDists = 20, numPoints = 100):
        phoneset = test_transform_questions.SimplePhoneset()
        questionGroups = test_transform_questions.getQuestionGroups(phoneset)
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            dist, inputGen = gen_DecisionTree_with_LinearGaussian_leaves(splitProb = 0.49, dimIn = dimIn)
            useDepthBasedTraining = randBool()
            def train(training):
                acc = d.AutoGrowingDiscreteAcc(createAcc = lambda: d.LinearGaussianAcc(inputLength = dimIn, varianceFloor = 0.0))
                for input, output, occ in training:
                    acc.add(input, output, occ)
                utilitySpec = cluster.MdlUtilitySpec(1.0)
                clusteringSpec = cluster.ClusteringSpec(
                    utilitySpec, questionGroups, minCount = 0.1, verbosity = 0
                )
                if useDepthBasedTraining:
                    return cluster.decisionTreeClusterDepthBased(
                        clusteringSpec, acc.accDict.keys(),
                        [acc.accDict.items()], acc.createAcc
                    )
                else:
                    return cluster.decisionTreeCluster(
                        clusteringSpec, acc.accDict.keys(),
                        lambda label: acc.accDict[label], acc.createAcc
                    )
            if True:
                # check decision tree clustering runs at all
                training = [ (input, dist.synth(input), math.exp(randn())) for input, index in zip(inputGen, range(numPoints)) ]
                estDist = train(training)
            if self.deepTest:
                check_est(dist, train, inputGen, hasParams = True)

    def test_MappedInputDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            dimIn = randint(0, 5)
            if randBool():
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
            if randBool():
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

    def test_CountFramesDist(self, eps = 1e-8, numDists = 10, numPoints = 50):
        def checkAccAdditional(acc, training):
            assert_allclose(acc.frames, sum([ len(output) * occ for input, output, occ in training ]))
        for distIndex in range(numDists):
            framesPerObs = randint(0, 5)
            dist, inputGen = gen_CountFramesDist(framesPerObs)
            # (FIXME : add logProbDerivInput and logProbDerivOutput eventually)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, checkAccAdditional = checkAccAdditional)
            if self.deepTest:
                initEstDist = gen_CountFramesDist(framesPerObs)[0]
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

    def test_shared_DiscreteDist(self, eps = 1e-8, numDists = 20, numPoints = 100):
        for distIndex in range(numDists):
            keys = list('abcde')[:randint(1, 5)]
            dimIn = randint(0, 5)
            dist, inputGen = gen_shared_DiscreteDist(keys, dimIn)
            checkLots(dist, inputGen, evalShouldWork = False, hasParams = True, eps = eps, numPoints = numPoints, logProbDerivOutputCheck = True)
            if self.deepTest:
                initEstDist = gen_shared_DiscreteDist(keys, dimIn)[0]
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    def test_AutoregressiveSequenceDist(self, eps = 1e-8, numDists = 10, numPoints = 100):
        def checkAccAdditional(acc, training):
            assert_allclose(acc.frames, sum([ len(output) * occ for input, output, occ in training ]))
        for distIndex in range(numDists):
            depth = randint(0, 5)
            dist, inputGen = gen_AutoregressiveSequenceDist(depth)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, checkAccAdditional = checkAccAdditional)
            if self.deepTest:
                initEstDist = randomizeParams(dist)
                check_est(dist, getTrainEM(initEstDist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(initEstDist), inputGen, hasParams = True)

    # (FIXME : check this is not unnecessarily slow for any reason)
    def test_AutoregressiveNetDist(self, eps = 1e-8, numDists = 5, numPoints = 100):
        def checkAdditional(dist, (uttId, input), outSeq, eps):
            # check result of getTimedNet is topologically sorted
            timedNet, labelToWeight = dist.getTimedNet(input, outSeq, preComputeLabelToWeight = randBool())
            assert wnet.netIsTopSorted(timedNet, wnet.nodeSetCompute(timedNet, accessibleOnly = False), deltaTime = lambda label: 0)
        def checkAccAdditional(acc, training):
            assert_allclose(acc.frames, sum([ len(output) * occ for input, output, occ in training ]))
        for distIndex in range(numDists):
            depth = randint(0, 5)
            # below restriction to dists with a moderate typical output length is to prevent tests taking too long
            if randBool():
                dist, inputGen = restrictTypicalOutputLength(genDist = lambda: gen_constant_AutoregressiveNetDist(depth = depth), numPoints = numPoints)
            else:
                dist, inputGen = restrictTypicalOutputLength(genDist = lambda: gen_inSeq_AutoregressiveNetDist(depth = depth), numPoints = numPoints)
            checkLots(dist, inputGen, hasParams = True, eps = eps, numPoints = numPoints, checkAdditional = checkAdditional, checkAccAdditional = checkAccAdditional)
            if self.deepTest:
                check_est(dist, getTrainEM(dist), inputGen, hasParams = True)
                check_est(dist, getTrainCG(dist), inputGen, hasParams = True)

@codeDeps()
def logProb_occ(dist, trainData):
    lp = 0.0
    occ = 0
    for input, output in trainData:
        lp += dist.logProb(input, output)
        occ += 1
    return lp, occ

# FIXME : this is nowhere near a proper unit test (need to make it more robust, automated, etc)
@codeDeps(gen_BinaryLogisticClassifier, logProb_occ, trn.trainEM)
def testBinaryLogisticClassifier():
    def inputGen(num):
        for i in range(num):
            yield np.append(randn(dim), 1.0)

    dim = 2
    blcTrue = gen_BinaryLogisticClassifier(dimIn = dim + 1)[0]
    num = 10000
    trainData = [ (input, blcTrue.synth(input)) for input in inputGen(num) ]
    def accumulate(acc):
        for input, output in trainData:
            acc.add(input, output)

    blc = gen_BinaryLogisticClassifier(dimIn = dim + 1, useZeroCoeff = True)[0]
    blc = trn.trainEM(blc, accumulate, deltaThresh = 1e-10, minIterations = 10, verbosity = 2)
    trainLogProb, trainOcc = logProb_occ(blc, trainData)
    print 'train set log prob = %s (%s frames)' % (trainLogProb / trainOcc, trainOcc)

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
        logging.warning('unusually large discrepancy between estimated and true dist during BinaryLogisticClassifier test')

# (N.B. not a unit test. Just draws pictures to help you assess whether results seem reasonable.)
@codeDeps(gen_BinaryLogisticClassifier, logProb_occ, trn.trainEM)
def testBinaryLogisticClassifierFunGraph():
    import matplotlib.pyplot as plt

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
    blcTrue = gen_BinaryLogisticClassifier(dimIn = dim + 1)[0]
    wTrue = blcTrue.coeff
    print 'DEBUG: wTrue =', wTrue
    print
    def inputGen(num):
        for i in range(num):
            yield np.append(randn(dim), 1.0)

    num = 3000
    trainData = [ (input, blcTrue.synth(input)) for input in inputGen(num) ]
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
        plt.plot(xBdy, yBdy, 'k-', xBdy0, yBdy0, 'r-', xBdy1, yBdy1, 'b-')

    def plotData():
        x0, y0 = zip(*[ input[:-1] for input, output in trainData if output == 0 ])
        x1, y1 = zip(*[ input[:-1] for input, output in trainData if output == 1 ])
        plt.plot(x0, y0, 'r+', x1, y1, 'bx')
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

    def afterEst(dist, it):
        plotBdy(dist)

    plotData()
    plotBdy(blcTrue)

    blc = gen_BinaryLogisticClassifier(dimIn = dim + 1, useZeroCoeff = True)[0]
    blc = trn.trainEM(blc, accumulate, deltaThresh = 1e-4, minIterations = 10, maxIterations = 50, afterEst = afterEst, verbosity = 2)
    print 'DEBUG: w estimated final =', blc.coeff
    trainLogProb, trainOcc = logProb_occ(blc, trainData)
    print 'train set log prob = %s (%s frames)' % (trainLogProb / trainOcc, trainOcc)

    plt.show()

# (N.B. not a unit test. Just draws pictures to help you assess whether results seem reasonable.)
@codeDeps(d.LinearGaussianAcc, d.estimateInitialMixtureOfTwoExperts)
def testMixtureOfTwoExpertsInitialization():
    import matplotlib.pyplot as plt

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
    trainData = [ (input, randn()) for input in inputGen(num) ]
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
        plt.plot(xBdy, yBdy, 'k-', xBdy0, yBdy0, 'r-', xBdy1, yBdy1, 'b-', xDir, yDir, 'g-')

    def plotData():
        x, y = zip(*[ input[:-1] for input, output in trainData ])
        plt.plot(x, y, 'r+')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

    plotData()

    acc = d.LinearGaussianAcc(inputLength = dim + 1, varianceFloor = 0.0)
    accumulate(acc)
    dist = d.estimateInitialMixtureOfTwoExperts(acc)
    blc = dist.classDist
    plotBdy(blc)
    print 'DEBUG: w estimated final =', blc.coeff

    plt.xlim(-10.0, 10.0)
    plt.ylim(-10.0, 10.0)
    plt.show()

@codeDeps(TestDist)
def suite(deepTest = False):
    # below definition nested here so that unittest search only finds shallow
    #   version of tests by default
    class DeepTestDist(TestDist):
        def setUp(self):
            self.deepTest = True
    if deepTest:
        return unittest.TestLoader().loadTestsFromTestCase(DeepTestDist)
    else:
        return unittest.TestLoader().loadTestsFromTestCase(TestDist)

if __name__ == '__main__':
    unittest.main()
