"""Clustering algorithms."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


import dist as d
import transform as xf
from armspeech.util.util import MapElem
from armspeech.util.mathhelp import ThreshMax
from armspeech.util.mathhelp import assert_allclose
from armspeech.util.timing import timed
from codedep import codeDeps

import logging
import math
from collections import defaultdict
import itertools
import heapq

@codeDeps()
def partitionLabels(labels, fullQuestion):
    labelValuer, question = fullQuestion
    labelsForAnswer = [ [] for _ in question.codomain() ]
    for label in labels:
        answer = question(labelValuer(label))
        labelsForAnswer[answer].append(label)
    return labelsForAnswer

@codeDeps()
def removeTrivialQuestions(labels, questionGroups):
    if not labels:
        return []
    else:
        questionGroupsOut = []
        for labelValuer, questions in questionGroups:
            questionsOut = []
            for question in questions:
                answerFirst = question(labelValuer(labels[0]))
                trivial = True
                for label in labels:
                    if question(labelValuer(label)) != answerFirst:
                        trivial = False
                        break
                if not trivial:
                    questionsOut.append(question)
            if questionsOut:
                questionGroupsOut.append((labelValuer, questionsOut))
        return questionGroupsOut

@codeDeps(d.addAcc)
class SecondLevelAccSummer(object):
    def __init__(self, createAcc):
        self.createAcc = createAcc

    def forQuestion(self, valueToAcc, question):
        accForAnswer = [ self.createAcc() for _ in question.codomain() ]
        for labelValue, acc in valueToAcc.iteritems():
            answer = question(labelValue)
            d.addAcc(accForAnswer[answer], acc)

        return accForAnswer

    def forQuestionGroups(self, qgToValueToAcc, questionGroups,
                          minCount = 0.0):
        """Computes the acc for each labelValuer, question and answer.

        The returned value is a list with zero or one elements for each
        questionGroup; if one element then this element is a list with zero or
        one elements for each question; if one element then this element is a
        list of accs, one for each possible answer.

        We say that a question violates the minimum count constraint if any of
        its answers has count less than minCount.
        Questions which violate this constraint, and question groups for which
        all the questions violate this constraint, are not represented in the
        returned value.
        """
        accsForQuestionGroups = []
        for (
            valueToAcc, (labelValuer, questions)
        ) in zip(qgToValueToAcc, questionGroups):
            accsForQuestions = []
            for question in questions:
                accForAnswer = self.forQuestion(valueToAcc, question)
                if all([ acc.count() >= minCount for acc in accForAnswer ]):
                    accsForQuestions.append((question, accForAnswer))
            if accsForQuestions:
                accsForQuestionGroups.append((labelValuer, accsForQuestions))

        return accsForQuestionGroups

@codeDeps(d.addAcc)
class NodeBasedFirstLevelAccSummer(object):
    """A first-level acc summer that is useful for node-based clustering."""
    def __init__(self, accForLabel, createAcc):
        self.accForLabel = accForLabel
        self.createAcc = createAcc

    def all(self, labels):
        accForLabel = self.accForLabel
        accTot = self.createAcc()
        for label in labels:
            d.addAcc(accTot, accForLabel(label))
        return accTot

    def forQuestion(self, labels, fullQuestion):
        labelValuer, question = fullQuestion
        accForLabel = self.accForLabel

        accForAnswer = [ self.createAcc() for _ in question.codomain() ]
        for label in labels:
            acc = accForLabel(label)
            answer = question(labelValuer(label))
            d.addAcc(accForAnswer[answer], acc)

        return accForAnswer

    def getQgToValueToAcc(self, labels, questionGroups):
        accForLabel = self.accForLabel
        numQuestionGroups = len(questionGroups)

        labelValuers = [ labelValuer
                         for labelValuer, questions in questionGroups ]
        qgToValueToAcc = [ defaultdict(self.createAcc)
                           for _ in range(numQuestionGroups) ]

        for label in labels:
            acc = accForLabel(label)
            for qgIndex in range(numQuestionGroups):
                labelValuer = labelValuers[qgIndex]
                valueToAcc = qgToValueToAcc[qgIndex]
                d.addAcc(valueToAcc[labelValuer(label)], acc)

        return qgToValueToAcc

@codeDeps(d.addAcc)
class DepthBasedFirstLevelAccSummer(object):
    """A first-level acc summer that is useful for depth-based clustering."""
    def __init__(self, labelledAccChunks, createAcc):
        self.labelledAccChunks = labelledAccChunks
        self.createAcc = createAcc

    def all(self):
        accTot = self.createAcc()
        for labelledAccs in self.labelledAccChunks:
            for _, acc in labelledAccs:
                d.addAcc(accTot, acc)
        return accTot

    def getLeafToQgToValueToAcc(self, numLeaves, labelToLeafIndex, questionGroups):
        leafToQgToValueToAcc = [
            [
                defaultdict(self.createAcc)
                for _ in questionGroups
            ]
            for _ in range(numLeaves)
        ]
        for labelledAccs in self.labelledAccChunks:
            for label, acc in labelledAccs:
                try:
                    leafIndex = labelToLeafIndex[label]
                except KeyError:
                    pass
                else:
                    for qgIndex, (labelValuer, _) in enumerate(questionGroups):
                        labelValue = labelValuer(label)
                        d.addAcc(
                            leafToQgToValueToAcc[leafIndex][qgIndex][labelValue],
                            acc
                        )

        return leafToQgToValueToAcc

@codeDeps()
class ProtoLeaf(object):
    def __init__(self, dist, aux, auxRat, count):
        self.dist = dist
        self.aux = aux
        self.auxRat = auxRat
        self.count = count

@codeDeps(ProtoLeaf, d.EstimationError)
class LeafEstimator(object):
    def __init__(self, estimateTotAux, catchEstimationErrors = False):
        self.estimateTotAux = estimateTotAux
        self.catchEstimationErrors = catchEstimationErrors

    def est(self, acc):
        dist, (aux, auxRat) = self.estimateTotAux(acc)
        count = acc.count()
        return ProtoLeaf(dist, aux, auxRat, count)

    def estForAnswer(self, accForAnswer):
        return [ self.est(acc) for acc in accForAnswer ]

    def estOrNone(self, acc):
        if self.catchEstimationErrors:
            try:
                return self.est(acc)
            except d.EstimationError:
                return None
        else:
            return self.est(acc)

    def estForAnswerOrNone(self, accForAnswer):
        if self.catchEstimationErrors:
            try:
                return self.estForAnswer(accForAnswer)
            except d.EstimationError:
                return None
        else:
            return self.estForAnswer(accForAnswer)

@codeDeps(assert_allclose)
class SplitInfo(object):
    """Collected information for a (potential or actual) split."""
    def __init__(self, protoNoSplit, fullQuestion, protoForAnswer):
        self.protoNoSplit = protoNoSplit
        self.fullQuestion = fullQuestion
        self.protoForAnswer = protoForAnswer

        assert len(self.protoForAnswer) >= 1
        if self.fullQuestion is None:
            assert self.protoForAnswer == [self.protoNoSplit]
        assert_allclose(
            sum([ proto.count for proto in self.protoForAnswer ]),
            self.protoNoSplit.count
        )

    def deltaNumLeaves(self):
        return len(self.protoForAnswer) - 1

    def delta(self, perLeafPenalty = 0.0):
        """Returns the delta for this split.

        The delta is used to choose which question to use to split a given node
        and to decide whether to split at all.
        """
        return (
            sum([ proto.aux - perLeafPenalty
                  for proto in self.protoForAnswer ]) -
            (self.protoNoSplit.aux - perLeafPenalty)
        )

@codeDeps()
class SplitValuer(object):
    def __init__(self, perLeafPenalty):
        self.perLeafPenalty = perLeafPenalty

        assert self.perLeafPenalty >= 0.0

    def __call__(self, splitInfo):
        return splitInfo.delta(self.perLeafPenalty)

@codeDeps(SplitValuer)
class FixedUtilitySpec(object):
    def __init__(self, perLeafPenalty):
        self.perLeafPenalty = perLeafPenalty

    def __call__(self, distRoot, countRoot, verbosity = 1):
        return SplitValuer(self.perLeafPenalty)

@codeDeps(SplitValuer, d.getDefaultParamSpec)
class MdlUtilitySpec(object):
    def __init__(self, mdlFactor, paramSpec = d.getDefaultParamSpec()):
        self.mdlFactor = mdlFactor
        self.paramSpec = paramSpec

    def __call__(self, distRoot, countRoot, verbosity = 1):
        numParamsPerLeaf = len(self.paramSpec.params(distRoot))
        perLeafPenalty = (
            0.5 * self.mdlFactor * numParamsPerLeaf * math.log(countRoot + 1.0)
        )
        if verbosity >= 1:
            print ('cluster: setting perLeafPenalty using MDL: mdlFactor = %s'
                   ' and numParamsPerLeaf = %s and count = %s' %
                   (self.mdlFactor, numParamsPerLeaf, countRoot))
        return SplitValuer(perLeafPenalty)

@codeDeps(SplitInfo)
def getPossSplits(protoNoSplit, accsForQuestionGroups, leafEstimator):
    splitInfos = []
    for labelValuer, accsForQuestions in accsForQuestionGroups:
        for question, accForAnswer in accsForQuestions:
            protoForAnswer = leafEstimator.estForAnswerOrNone(accForAnswer)
            if protoForAnswer is not None:
                fullQuestion = labelValuer, question
                splitInfos.append(
                    SplitInfo(protoNoSplit, fullQuestion, protoForAnswer)
                )

    return splitInfos

@codeDeps()
def getPrunedQuestionGroups(accsForQuestionGroups):
    return [
        (labelValuer, [ question for question, _ in accsForQuestions ])
        for labelValuer, accsForQuestions in accsForQuestionGroups
    ]

@codeDeps(SplitInfo, ThreshMax)
def getBestAction(protoNoSplit, splitInfos, splitValuer, goodThresh = 0.1):
    threshMax = ThreshMax(goodThresh, key = splitValuer)
    threshMaxZero = ThreshMax(0.0, key = splitValuer)

    noSplitInfo = SplitInfo(protoNoSplit, None, [protoNoSplit])
    goodSplitInfos = threshMax(splitInfos + [noSplitInfo])
    bestSplitInfos = threshMaxZero(goodSplitInfos)
    bestSplitInfo = bestSplitInfos[0]
    bestSplitInfo.bests = bestSplitInfos
    bestSplitInfo.goods = goodSplitInfos
    return bestSplitInfo

@codeDeps(MapElem, d.DiscreteDist, d.MappedInputDist, d.sumValuedRats,
    xf.DecisionTree
)
def constructTree(splitInfoDict):
    leafProtos = []
    def construct(answerSeq):
        splitInfo = splitInfoDict[answerSeq]
        if splitInfo.fullQuestion is None:
            leafProto, = splitInfo.protoForAnswer
            leafProtos.append(leafProto)
            return len(leafProtos) - 1
        else:
            _, question = splitInfo.fullQuestion
            treeForAnswer = [ construct(answerSeq + (answer,))
                              for answer in question.codomain() ]
            return (splitInfo.fullQuestion, treeForAnswer)
    tree = construct(())

    leafDists = [ leafProto.dist for leafProto in leafProtos ]
    auxValuedRat = d.sumValuedRats([ (leafProto.aux, leafProto.auxRat)
                                     for leafProto in leafProtos ])
    decTree = xf.DecisionTree(tree, numLeaves = len(leafProtos))
    dist = d.MappedInputDist(MapElem(0, 2, decTree),
        d.DiscreteDist(range(decTree.numLeaves),
            dict(enumerate(leafDists))
        )
    )
    return dist, auxValuedRat

@codeDeps(getBestAction, getPossSplits, getPrunedQuestionGroups,
    partitionLabels, timed
)
class NodeBasedClusterer(object):
    """Supports decision tree clustering in a node-by-node fashion.

    More specifically this class contains methods that use a certain form of
    state which is useful for node-based clustering.
    """
    def __init__(self, accSummer1, accSummer2, minCount, leafEstimator,
                 splitValuer, goodThresh, verbosity):
        self.accSummer1 = accSummer1
        self.accSummer2 = accSummer2
        self.minCount = minCount
        self.leafEstimator = leafEstimator
        self.splitValuer = splitValuer
        self.goodThresh = goodThresh
        self.verbosity = verbosity

    def computeBestSplitAndStateAdj(self, state):
        labels, questionGroups, answerSeq, protoNoSplit = state

        qgToValueToAcc = self.accSummer1.getQgToValueToAcc(
            labels, questionGroups
        )
        accsForQuestionGroups = self.accSummer2.forQuestionGroups(
            qgToValueToAcc, questionGroups, minCount = self.minCount
        )

        questionGroupsOut = getPrunedQuestionGroups(accsForQuestionGroups)

        splitInfos = getPossSplits(protoNoSplit, accsForQuestionGroups,
                                   self.leafEstimator)
        bestSplitInfo = getBestAction(protoNoSplit, splitInfos,
                                      self.splitValuer,
                                      goodThresh = self.goodThresh)

        stateAdj = labels, questionGroupsOut, answerSeq, protoNoSplit
        return bestSplitInfo, stateAdj

    def getNextStates(self, state, splitInfo):
        labels, questionGroups, answerSeq, protoNoSplit = state

        if self.verbosity >= 2:
            indent = '    '+''.join([ ('|  ' if answer != 0 else '   ')
                                      for answer in answerSeq ])
        if self.verbosity >= 2:
            print ('cluster:%s(bests = %s, goods = %s)' %
                   (indent, len(splitInfo.bests), len(splitInfo.goods)))
        if splitInfo.fullQuestion is None:
            if self.verbosity >= 2:
                print 'cluster:'+indent+'leaf'
            return []
        else:
            labelValuer, question = splitInfo.fullQuestion
            if self.verbosity >= 2:
                print ('cluster:%squestion ( %s %s ) ( delta = %s )' %
                       (indent, labelValuer.shortRepr(), question.shortRepr(),
                        splitInfo.delta()))
            labelsForAnswer = partitionLabels(labels, (labelValuer, question))

            return [
                (labels, questionGroups, answerSeq + (answer,), proto)
                # (FIXME : using reversed below is a bit odd: it means tree is
                #   explored with later answers (such as 1 / "yes") first.
                #   Did it this way to provide backwards compatibility for
                #   printing out pretty pictures of the tree so far, but may
                #   ultimately want to get rid of both pictures and reversed.)
                for answer, labels, proto in reversed(zip(
                    question.codomain(),
                    labelsForAnswer,
                    splitInfo.protoForAnswer
                ))
            ]

    def printNodeInfo(self, state):
        labels, questionGroups, answerSeq, protoNoSplit = state

        indent = '    '+''.join([ ('|  ' if answer != 0 else '   ')
                                  for answer in answerSeq[:-1] ])
        if not answerSeq:
            extra = ''
        elif answerSeq[-1] != 0:
            extra = '|->'
        else:
            extra = '\->'
        print ('cluster:%s%s(%s)-node ( count = %s ,'
               ' remaining labels/QGs/Qs = %s/%s/%s )' %
               (indent, extra, answerSeq[-1] if answerSeq else '',
                protoNoSplit.count, len(labels), len(questionGroups),
                sum([ len(questions)
                      for _, questions in questionGroups ])))

    def subTreeSplitInfoIter(self, stateInit):
        agenda = [stateInit]
        while agenda:
            state = agenda.pop()
            labels, questionGroups, answerSeq, protoNoSplit = state
            if self.verbosity >= 2:
                self.printNodeInfo(state)
            if self.verbosity >= 3:
                indent = '    '+''.join([ ('|  ' if answer != 0 else '   ')
                                          for answer in answerSeq ])
                computeBestSplit = timed(
                    self.computeBestSplitAndStateAdj,
                    msg = 'cluster:%schoose and perform split took' % indent
                )
            else:
                computeBestSplit = self.computeBestSplitAndStateAdj
            splitInfo, stateAdj = computeBestSplit(state)
            nextStates = self.getNextStates(stateAdj, splitInfo)
            agenda.extend(reversed(nextStates))
            yield answerSeq, splitInfo

    def subTreeSplitInfoIterInGreedyOrder(self, stateInit):
        agenda = []

        def agendaPush(state):
            splitInfo, stateAdj = self.computeBestSplitAndStateAdj(state)
            if splitInfo.fullQuestion is not None:
                heapq.heappush(
                    agenda,
                    (-self.splitValuer(splitInfo), splitInfo, stateAdj)
                )

        agendaPush(stateInit)
        while agenda:
            value, splitInfo, state = heapq.heappop(agenda)
            labels, questionGroups, answerSeq, protoNoSplit = state
            if self.verbosity >= 2:
                self.printNodeInfo(state)
            nextStates = self.getNextStates(state, splitInfo)
            for nextState in nextStates:
                agendaPush(nextState)
            yield answerSeq, splitInfo

@codeDeps(getBestAction, getPossSplits, timed)
class DepthBasedClusterer(object):
    """Supports decision tree clustering in a layer-by-layer fashion.

    More specifically this class contains methods that use a certain form of
    state which is useful for depth-based clustering.
    """
    def __init__(self, accSummer1, accSummer2, minCount, leafEstimator,
                 splitValuer, goodThresh, verbosity):
        self.accSummer1 = accSummer1
        self.accSummer2 = accSummer2
        self.minCount = minCount
        self.leafEstimator = leafEstimator
        self.splitValuer = splitValuer
        self.goodThresh = goodThresh
        self.verbosity = verbosity

    def getAccsForQuestionGroupsForLeaf(self, leafToQgToValueToAcc,
                                        questionGroups):
        accsForQuestionGroupsForLeaf = []
        for qgToValueToAcc in leafToQgToValueToAcc:
            accsForQuestionGroups = self.accSummer2.forQuestionGroups(
                qgToValueToAcc, questionGroups, minCount = self.minCount
            )
            accsForQuestionGroupsForLeaf.append(accsForQuestionGroups)

        return accsForQuestionGroupsForLeaf

    def checkStateConsistent(self, state):
        labelToLeafIndex, leafProtos, leafAnswerSeqs = state
        numLeaves = len(leafProtos)
        assert len(leafAnswerSeqs) == numLeaves
        assert set(labelToLeafIndex.values()) == set(range(numLeaves))

    def getInitialState(self, labels, protoRoot):
        labelToLeafIndex = dict()
        for label in labels:
            labelToLeafIndex[label] = 0

        leafProtos = [protoRoot]
        leafAnswerSeqs = [()]

        state = labelToLeafIndex, leafProtos, leafAnswerSeqs
        return state

    def getNextState(self, state, bestSplitInfoForLeaf):
        labelToLeafIndex, leafProtos, leafAnswerSeqs = state

        leafProtosNew = []
        leafAnswerSeqsNew = []
        leafIndexMaps = []
        for leafIndex, splitInfo in enumerate(bestSplitInfoForLeaf):
            if splitInfo.fullQuestion is None:
                leafIndexMaps.append(None)
            else:
                _, question = splitInfo.fullQuestion
                protoForAnswer = splitInfo.protoForAnswer
                answerSeq = leafAnswerSeqs[leafIndex]
                leafIndexMap = []
                for answer, proto in zip(question.codomain(), protoForAnswer):
                    leafIndexNew = len(leafProtosNew)
                    leafProtosNew.append(proto)
                    leafAnswerSeqsNew.append(answerSeq + (answer,))
                    leafIndexMap.append(leafIndexNew)
                leafIndexMaps.append(leafIndexMap)
        assert len(leafIndexMaps) == len(leafProtos)

        labelToLeafIndexNew = dict()
        for label in labelToLeafIndex:
            leafIndex = labelToLeafIndex[label]
            leafIndexMap = leafIndexMaps[leafIndex]
            if leafIndexMap is not None:
                splitInfo = bestSplitInfoForLeaf[leafIndex]
                labelValuer, question = splitInfo.fullQuestion
                answer = question(labelValuer(label))
                leafIndexNew = leafIndexMap[answer]
                labelToLeafIndexNew[label] = leafIndexNew

        stateNew = labelToLeafIndexNew, leafProtosNew, leafAnswerSeqsNew
        return stateNew

    def addLayer(self, state, questionGroups):
        """Computes a single additional layer of the decision tree."""
        labelToLeafIndex, leafProtos, leafAnswerSeqs = state
        numLeaves = len(leafProtos)

        leafToQgToValueToAcc = self.accSummer1.getLeafToQgToValueToAcc(
            numLeaves, labelToLeafIndex, questionGroups
        )

        accsForQuestionGroupsForLeaf = self.getAccsForQuestionGroupsForLeaf(
            leafToQgToValueToAcc, questionGroups
        )

        bestSplitInfoForLeaf = []
        for leafIndex in range(numLeaves):
            accsForQuestionGroups = accsForQuestionGroupsForLeaf[leafIndex]
            protoNoSplit = leafProtos[leafIndex]
            splitInfos = getPossSplits(protoNoSplit, accsForQuestionGroups,
                                       self.leafEstimator)
            bestSplitInfo = getBestAction(
                protoNoSplit, splitInfos, self.splitValuer,
                goodThresh = self.goodThresh
            )

            bestSplitInfoForLeaf.append(bestSplitInfo)

        splitInfoDict = dict()
        for leafIndex, splitInfo in enumerate(bestSplitInfoForLeaf):
            answerSeq = leafAnswerSeqs[leafIndex]
            splitInfoDict[answerSeq] = splitInfo

        stateNew = self.getNextState(state, bestSplitInfoForLeaf)

        return stateNew, splitInfoDict

    def addLayers(self, stateInit, questionGroups):
        """Computes all remaining layers of the decision tree."""
        splitInfoDict = dict()
        state = stateInit
        while True:
            addLayer = self.addLayer
            if self.verbosity >= 3:
                addLayer = timed(addLayer)
            state, splitInfoDictMore = addLayer(state, questionGroups)
            splitInfoDict.update(splitInfoDictMore)
            if self.verbosity >= 2:
                print 'cluster: added %s nodes' % len(splitInfoDictMore)
            if not splitInfoDictMore:
                break

        return splitInfoDict

@codeDeps(d.getDefaultEstimateTotAuxNoRevert)
class ClusteringSpec(object):
    def __init__(self, utilitySpec, questionGroups, minCount,
                 estimateTotAux = d.getDefaultEstimateTotAuxNoRevert(),
                 catchEstimationErrors = False,
                 goodThresh = 0.1,
                 verbosity = 2):
        self.utilitySpec = utilitySpec
        self.questionGroups = questionGroups
        self.minCount = minCount
        self.estimateTotAux = estimateTotAux
        self.catchEstimationErrors = catchEstimationErrors
        self.goodThresh = goodThresh
        self.verbosity = verbosity

@codeDeps(LeafEstimator, NodeBasedClusterer, NodeBasedFirstLevelAccSummer,
    SecondLevelAccSummer, constructTree, d.Rat, removeTrivialQuestions, timed
)
def decisionTreeCluster(clusteringSpec, labels, accForLabel, createAcc):
    verbosity = clusteringSpec.verbosity
    accSummer1 = NodeBasedFirstLevelAccSummer(accForLabel, createAcc)
    accSummer2 = SecondLevelAccSummer(createAcc)
    minCount = clusteringSpec.minCount
    leafEstimator = LeafEstimator(
        clusteringSpec.estimateTotAux,
        catchEstimationErrors = clusteringSpec.catchEstimationErrors
    )
    def getProtoRoot():
        return leafEstimator.est(accSummer1.all(labels))
    if verbosity >= 3:
        getProtoRoot = timed(getProtoRoot)
    protoRoot = getProtoRoot()
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = NodeBasedClusterer(accSummer1, accSummer2, minCount,
                                   leafEstimator, splitValuer,
                                   clusteringSpec.goodThresh,
                                   verbosity = verbosity)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with perLeafPenalty = %s and'
               ' minCount = %s' %
               (splitValuer.perLeafPenalty, minCount))

    questionGroups = removeTrivialQuestions(labels,
                                            clusteringSpec.questionGroups)
    splitInfoDict = dict(
        clusterer.subTreeSplitInfoIter((labels, questionGroups, (), protoRoot))
    )
    dist, (aux, auxRat) = constructTree(splitInfoDict)

    if verbosity >= 1:
        countRoot = protoRoot.count
        # (FIXME : leaf computation relies on specific form of dist)
        print 'cluster: %s leaves' % len(dist.dist.distDict)
        print ('cluster: aux root = %s (%s) -> aux tree = %s (%s) (%s count)' %
               (protoRoot.aux / countRoot, d.Rat.toString(protoRoot.auxRat),
                aux / countRoot, d.Rat.toString(auxRat),
                countRoot))
    return dist

@codeDeps(DepthBasedClusterer, DepthBasedFirstLevelAccSummer, LeafEstimator,
    SecondLevelAccSummer, constructTree, d.Rat, removeTrivialQuestions, timed
)
def decisionTreeClusterDepthBased(clusteringSpec, labels, labelledAccChunks,
                                  createAcc):
    verbosity = clusteringSpec.verbosity
    accSummer1 = DepthBasedFirstLevelAccSummer(labelledAccChunks, createAcc)
    accSummer2 = SecondLevelAccSummer(createAcc)
    minCount = clusteringSpec.minCount
    leafEstimator = LeafEstimator(
        clusteringSpec.estimateTotAux,
        catchEstimationErrors = clusteringSpec.catchEstimationErrors
    )
    def getProtoRoot():
        return leafEstimator.est(accSummer1.all())
    if verbosity >= 3:
        getProtoRoot = timed(getProtoRoot)
    protoRoot = getProtoRoot()
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = DepthBasedClusterer(accSummer1, accSummer2, minCount,
                                    leafEstimator, splitValuer,
                                    clusteringSpec.goodThresh,
                                    verbosity = verbosity)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with perLeafPenalty = %s and'
               ' minCount = %s' %
               (splitValuer.perLeafPenalty, minCount))

    questionGroups = removeTrivialQuestions(labels,
                                            clusteringSpec.questionGroups)
    stateInit = clusterer.getInitialState(labels, protoRoot)
    splitInfoDict = clusterer.addLayers(stateInit, questionGroups)
    dist, (aux, auxRat) = constructTree(splitInfoDict)

    if verbosity >= 1:
        countRoot = protoRoot.count
        # (FIXME : leaf computation relies on specific form of dist)
        print 'cluster: %s leaves' % len(dist.dist.distDict)
        print ('cluster: aux root = %s (%s) -> aux tree = %s (%s) (%s count)' %
               (protoRoot.aux / countRoot, d.Rat.toString(protoRoot.auxRat),
                aux / countRoot, d.Rat.toString(auxRat),
                countRoot))
    return dist

@codeDeps(d.addAcc, d.getDefaultCreateAcc, partitionLabels)
def getDeltaIter(labelsRoot, accForLabel, distRoot, splitIter,
                 createAcc = d.getDefaultCreateAcc()):
    def getLogProb(dist, labels):
        acc = createAcc(dist)
        for label in labels:
            d.addAcc(acc, accForLabel(label))
        return acc.logLike()

    logProbDict = dict()
    logProbDict[()] = getLogProb(distRoot, labelsRoot)
    labelsDict = dict()
    labelsDict[()] = labelsRoot

    for answerSeq, fullQuestion, distForAnswer in splitIter:
        labelsForAnswer = partitionLabels(labelsDict[answerSeq], fullQuestion)

        _, question = fullQuestion
        delta = -logProbDict[answerSeq]
        for answer, labels, dist in zip(question.codomain(), labelsForAnswer,
                                        distForAnswer):
            logProb = getLogProb(dist, labels)
            delta += logProb
            logProbDict[answerSeq + (answer,)] = logProb
            labelsDict[answerSeq + (answer,)] = labels

        yield delta

@codeDeps(LeafEstimator, NodeBasedClusterer, NodeBasedFirstLevelAccSummer,
    SecondLevelAccSummer, getDeltaIter, removeTrivialQuestions
)
def decisionTreeClusterInGreedyOrderWithTest(clusteringSpec,
                                             labels, labelsTest,
                                             accForLabel, accForLabelTest,
                                             createAcc):
    verbosity = clusteringSpec.verbosity
    accSummer1 = NodeBasedFirstLevelAccSummer(accForLabel, createAcc)
    accSummer2 = SecondLevelAccSummer(createAcc)
    minCount = clusteringSpec.minCount
    leafEstimator = LeafEstimator(
        clusteringSpec.estimateTotAux,
        catchEstimationErrors = clusteringSpec.catchEstimationErrors
    )
    protoRoot = leafEstimator.est(accSummer1.all(labels))
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = NodeBasedClusterer(accSummer1, accSummer2, minCount,
                                   leafEstimator, splitValuer,
                                   clusteringSpec.goodThresh,
                                   verbosity = verbosity)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with perLeafPenalty = %s and'
               ' minCount = %s' %
               (splitValuer.perLeafPenalty, minCount))

    questionGroups = removeTrivialQuestions(labels,
                                            clusteringSpec.questionGroups)

    # (have to be a bit careful about iterators getting used up; not ideal)
    splitInfoIter = clusterer.subTreeSplitInfoIterInGreedyOrder(
        (labels, questionGroups, (), protoRoot)
    )
    splitInfoIter, splitInfoIter2, splitInfoIter3 = itertools.tee(
        splitInfoIter, 3
    )
    splitIter = ( (answerSeq, splitInfo.fullQuestion,
                   [ proto.dist for proto in splitInfo.protoForAnswer ])
                  for answerSeq, splitInfo in splitInfoIter2 )
    deltaNumLeavesIter = ( splitInfo.deltaNumLeaves()
                           for answerSeq, splitInfo in splitInfoIter3 )
    deltaIterTrain = ( splitInfo.delta()
                       for answerSeq, splitInfo in splitInfoIter )
    deltaIterTest = getDeltaIter(labelsTest, accForLabelTest, protoRoot.dist,
                                 splitIter)
    return itertools.izip(deltaNumLeavesIter, deltaIterTrain, deltaIterTest)
