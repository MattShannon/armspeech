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
class AccSummer(object):
    def __init__(self, accForLabel, createAcc):
        self.accForLabel = accForLabel
        self.createAcc = createAcc

    def sumAccs(self, labels):
        accForLabel = self.accForLabel
        accTot = self.createAcc()
        for label in labels:
            d.addAcc(accTot, accForLabel(label))
        return accTot

    def sumAccsForQuestion(self, labels, fullQuestion):
        labelValuer, question = fullQuestion
        accForLabel = self.accForLabel

        accForAnswer = [ self.createAcc() for _ in question.codomain() ]
        for label in labels:
            acc = accForLabel(label)
            answer = question(labelValuer(label))
            d.addAcc(accForAnswer[answer], acc)

        return accForAnswer

    def sumAccsFirstLevel(self, labels, questionGroups):
        accForLabel = self.accForLabel
        numQuestionGroups = len(questionGroups)

        labelValuers = [ labelValuer
                         for labelValuer, questions in questionGroups ]
        labelValueToAccs = [ defaultdict(self.createAcc)
                             for _ in range(numQuestionGroups) ]

        for label in labels:
            acc = accForLabel(label)
            for qgIndex in range(numQuestionGroups):
                labelValuer = labelValuers[qgIndex]
                labelValueToAcc = labelValueToAccs[qgIndex]
                d.addAcc(labelValueToAcc[labelValuer(label)], acc)

        # list contains one labelValueToAcc for each questionGroup
        return labelValueToAccs

    def sumAccsSecondLevel(self, labelValueToAcc, question):
        accForAnswer = [ self.createAcc() for _ in question.codomain() ]
        for labelValue, acc in labelValueToAcc.iteritems():
            answer = question(labelValue)
            d.addAcc(accForAnswer[answer], acc)

        return accForAnswer

    def sumAccsForQuestionGroups(self, labels, questionGroups, minCount = 0.0):
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
        labelValueToAccs = self.sumAccsFirstLevel(labels, questionGroups)

        accsForQuestionGroups = []
        for (
            labelValueToAcc, (labelValuer, questions)
        ) in zip(labelValueToAccs, questionGroups):
            accsForQuestions = []
            for question in questions:
                accForAnswer = self.sumAccsSecondLevel(labelValueToAcc,
                                                       question)
                if all([ acc.count() >= minCount for acc in accForAnswer ]):
                    accsForQuestions.append((question, accForAnswer))
            if accsForQuestions:
                accsForQuestionGroups.append((labelValuer, accsForQuestions))

        return accsForQuestionGroups

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

        assert self.protoNoSplit is not None
        assert len(self.protoForAnswer) >= 1
        assert all([ proto is not None for proto in self.protoForAnswer ])
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

@codeDeps(MapElem, SplitInfo, d.DiscreteDist, d.MappedInputDist,
    d.sumValuedRats, partitionLabels, timed, xf.DecisionTree
)
class DecisionTreeClusterer(object):
    def __init__(self, accSummer, minCount, leafEstimator, splitValuer,
                 nearBestThresh, verbosity):
        self.accSummer = accSummer
        self.minCount = minCount
        self.leafEstimator = leafEstimator
        self.splitValuer = splitValuer
        self.nearBestThresh = nearBestThresh
        self.verbosity = verbosity

        self.threshMax = ThreshMax(self.nearBestThresh, key = self.splitValuer)
        self.threshMaxZero = ThreshMax(0.0, key = self.splitValuer)

    def getPossSplits(self, protoNoSplit, accsForQuestionGroups):
        splitInfos = []
        for labelValuer, accsForQuestions in accsForQuestionGroups:
            for question, accForAnswer in accsForQuestions:
                protoForAnswer = self.leafEstimator.estForAnswerOrNone(
                    accForAnswer
                )
                if protoForAnswer is not None:
                    fullQuestion = labelValuer, question
                    splitInfos.append(
                        SplitInfo(protoNoSplit, fullQuestion, protoForAnswer)
                    )

        return splitInfos

    def getPrunedQuestionGroups(self, accsForQuestionGroups):
        return [
            (labelValuer, [ question for question, _ in accsForQuestions ])
            for labelValuer, accsForQuestions in accsForQuestionGroups
        ]

    def computeBestSplitAndStateAdj(self, state):
        labels, questionGroups, answerSeq, protoNoSplit = state

        accsForQuestionGroups = self.accSummer.sumAccsForQuestionGroups(
            labels, questionGroups, minCount = self.minCount
        )

        questionGroupsOut = self.getPrunedQuestionGroups(accsForQuestionGroups)

        splitInfos = self.getPossSplits(protoNoSplit, accsForQuestionGroups)
        noSplitInfo = SplitInfo(protoNoSplit, None, [protoNoSplit])
        nearBestSplitInfos = self.threshMax(splitInfos + [noSplitInfo])
        bestSplitInfos = self.threshMaxZero(nearBestSplitInfos)
        bestSplitInfo = bestSplitInfos[0]
        bestSplitInfo.bests = bestSplitInfos
        bestSplitInfo.nearBests = nearBestSplitInfos

        stateAdj = labels, questionGroupsOut, answerSeq, protoNoSplit
        return bestSplitInfo, stateAdj

    def getNextStates(self, state, splitInfo):
        labels, questionGroups, answerSeq, protoNoSplit = state

        if self.verbosity >= 2:
            indent = '    '+''.join([ ('|  ' if answer != 0 else '   ')
                                      for answer in answerSeq ])
        if self.verbosity >= 2:
            print ('cluster:%s(bests = %s, nearBests = %s)' %
                   (indent, len(splitInfo.bests), len(splitInfo.nearBests)))
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
            if splitInfo.fullQuestion is not None:
                splitInfo, stateAdj = self.computeBestSplitAndStateAdj(state)
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

    def constructTree(self, splitInfoDict):
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

@codeDeps(d.getDefaultEstimateTotAuxNoRevert)
class ClusteringSpec(object):
    def __init__(self, utilitySpec, questionGroups, minCount,
                 estimateTotAux = d.getDefaultEstimateTotAuxNoRevert(),
                 catchEstimationErrors = False,
                 nearBestThresh = 0.1,
                 verbosity = 2):
        self.utilitySpec = utilitySpec
        self.questionGroups = questionGroups
        self.minCount = minCount
        self.estimateTotAux = estimateTotAux
        self.catchEstimationErrors = catchEstimationErrors
        self.nearBestThresh = nearBestThresh
        self.verbosity = verbosity

@codeDeps(AccSummer, DecisionTreeClusterer, LeafEstimator, d.Rat,
    removeTrivialQuestions, timed
)
def decisionTreeCluster(clusteringSpec, labels, accForLabel, createAcc):
    verbosity = clusteringSpec.verbosity
    accSummer = AccSummer(accForLabel, createAcc)
    minCount = clusteringSpec.minCount
    leafEstimator = LeafEstimator(
        clusteringSpec.estimateTotAux,
        catchEstimationErrors = clusteringSpec.catchEstimationErrors
    )
    def getProtoRoot():
        return leafEstimator.est(accSummer.sumAccs(labels))
    if verbosity >= 3:
        getProtoRoot = timed(getProtoRoot)
    protoRoot = getProtoRoot()
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = DecisionTreeClusterer(accSummer, minCount, leafEstimator,
                                      splitValuer,
                                      clusteringSpec.nearBestThresh,
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
    dist, (aux, auxRat) = clusterer.constructTree(splitInfoDict)

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

@codeDeps(AccSummer, DecisionTreeClusterer, LeafEstimator, getDeltaIter,
    removeTrivialQuestions
)
def decisionTreeClusterInGreedyOrderWithTest(clusteringSpec,
                                             labels, labelsTest,
                                             accForLabel, accForLabelTest,
                                             createAcc):
    verbosity = clusteringSpec.verbosity
    accSummer = AccSummer(accForLabel, createAcc)
    minCount = clusteringSpec.minCount
    leafEstimator = LeafEstimator(
        clusteringSpec.estimateTotAux,
        catchEstimationErrors = clusteringSpec.catchEstimationErrors
    )
    protoRoot = leafEstimator.est(accSummer.sumAccs(labels))
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = DecisionTreeClusterer(accSummer, minCount, leafEstimator,
                                      splitValuer,
                                      clusteringSpec.nearBestThresh,
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
