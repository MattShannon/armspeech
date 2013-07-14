"""Clustering algorithms."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


import dist as d
import transform as xf
from armspeech.util.util import MapElem
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

    def sumAccsForQuestions(self, labels, questionGroups):
        """Returns an iterator with yes and no accs for each question."""
        labelValueToAccs = self.sumAccsFirstLevel(labels, questionGroups)

        for (
            labelValueToAcc, (labelValuer, questions)
        ) in zip(labelValueToAccs, questionGroups):
            for question in questions:
                accForAnswer = self.sumAccsSecondLevel(labelValueToAcc,
                                                       question)
                fullQuestion = labelValuer, question
                yield fullQuestion, accForAnswer

    def sumAccsForQuestionGroups(self, labels, questionGroups):
        labelValueToAccs = self.sumAccsFirstLevel(labels, questionGroups)

        accsForQuestionGroups = []
        for (
            labelValueToAcc, (labelValuer, questions)
        ) in zip(labelValueToAccs, questionGroups):
            accsForQuestions = []
            for question in questions:
                accForAnswer = self.sumAccsSecondLevel(labelValueToAcc,
                                                       question)
                accsForQuestions.append((question, accForAnswer))
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
    def __init__(self, estimateTotAux):
        self.estimateTotAux = estimateTotAux

    def est(self, acc):
        dist, (aux, auxRat) = self.estimateTotAux(acc)
        count = acc.count()
        return ProtoLeaf(dist, aux, auxRat, count)

    def estOrNone(self, acc):
        try:
            return self.est(acc)
        except d.EstimationError:
            return None

@codeDeps(assert_allclose)
class SplitInfo(object):
    """Collected information for a (potential or actual) split."""
    def __init__(self, protoNoSplit, fullQuestion, protoForAnswer):
        self.protoNoSplit = protoNoSplit
        self.fullQuestion = fullQuestion
        self.protoForAnswer = protoForAnswer

        assert self.protoNoSplit is not None
        assert len(self.protoForAnswer) >= 1
        if self.fullQuestion is None:
            assert self.protoForAnswer == [self.protoNoSplit]
        if any([ proto is not None for proto in self.protoForAnswer ]):
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
        if any([ proto is None for proto in self.protoForAnswer ]):
            return float('-inf')
        else:
            return (
                sum([ proto.aux - perLeafPenalty
                      for proto in self.protoForAnswer ]) -
                (self.protoNoSplit.aux - perLeafPenalty)
            )

@codeDeps()
class SplitValuer(object):
    def __init__(self, perLeafPenalty, minCount):
        self.perLeafPenalty = perLeafPenalty
        self.minCount = minCount

        assert self.perLeafPenalty >= 0.0
        assert self.minCount > 0.0

    def __call__(self, splitInfo):
        if all([ proto is not None and proto.count >= self.minCount
                 for proto in splitInfo.protoForAnswer ]):
            return splitInfo.delta(self.perLeafPenalty)
        else:
            return float('-inf')

@codeDeps(SplitValuer)
class FixedUtilitySpec(object):
    def __init__(self, perLeafPenalty, minCount):
        self.perLeafPenalty = perLeafPenalty
        self.minCount = minCount

    def __call__(self, distRoot, countRoot, verbosity = 1):
        return SplitValuer(self.perLeafPenalty, self.minCount)

@codeDeps(SplitValuer, d.getDefaultParamSpec)
class MdlUtilitySpec(object):
    def __init__(self, mdlFactor, minCount,
                 paramSpec = d.getDefaultParamSpec()):
        self.mdlFactor = mdlFactor
        self.minCount = minCount
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
        return SplitValuer(perLeafPenalty, self.minCount)

@codeDeps(MapElem, SplitInfo, d.DiscreteDist, d.MappedInputDist,
    d.sumValuedRats, partitionLabels, timed, xf.DecisionTree
)
class DecisionTreeClusterer(object):
    def __init__(self, accSummer, leafEstimator, splitValuer, verbosity):
        self.accSummer = accSummer
        self.leafEstimator = leafEstimator
        self.splitValuer = splitValuer
        self.verbosity = verbosity

    def getPossSplitIter(self, state, questionGroups):
        """Returns an iterator with one SplitInfo for each allowed question.

        A clustering state is a proto-leaf together with info about its
        position in the tree and enough info about the initial parts of the
        tree to allow clustering of the sub-tree with the proto-leaf as root.
        (This initial tree info consists of the labels remaining and the
        non-trivial question groups remaining).

        For a given state and list of questionGroups, this function returns an
        iterator over splits, one for each allowed question.
        """
        labels, questionGroupsRemaining, answerSeq, protoNoSplit = state

        for (
            fullQuestion, accForAnswer
        ) in self.accSummer.sumAccsForQuestions(labels, questionGroups):
            _, question = fullQuestion
            protoForAnswer = [ self.leafEstimator.estOrNone(acc)
                               for acc in accForAnswer ]
            splitInfo = SplitInfo(protoNoSplit, fullQuestion, protoForAnswer)
            if self.splitValuer(splitInfo) > float('-inf'):
                yield splitInfo

    def getPossSplitsWithPrunedQuestionGroups(self, state, questionGroups):
        """Returns a list of possible splits (and an updated questionGroups).

        This method is like getPossSplitIter but returns a list rather than an
        iterator, and returns an updated questionGroups with certain questions
        that can never be selected in future (that is, lower in the tree)
        removed.
        """
        labels, questionGroupsRemaining, answerSeq, protoNoSplit = state

        minCount = self.splitValuer.minCount

        questionGroupsOut = []
        splitInfos = []
        for (
            labelValuer, accsForQuestions
        ) in self.accSummer.sumAccsForQuestionGroups(labels, questionGroups):
            questionsOut = []
            for question, accForAnswer in accsForQuestions:
                if all([ acc.count() >= minCount for acc in accForAnswer ]):
                    questionsOut.append(question)

                    protoForAnswer = [ self.leafEstimator.estOrNone(acc)
                                       for acc in accForAnswer ]
                    fullQuestion = labelValuer, question
                    splitInfo = SplitInfo(protoNoSplit, fullQuestion,
                                          protoForAnswer)
                    if self.splitValuer(splitInfo) > float('-inf'):
                        splitInfos.append(splitInfo)
            if questionsOut:
                questionGroupsOut.append((labelValuer, questionsOut))

        return splitInfos, questionGroupsOut

    def getNextStates(self, state, splitInfo):
        labels, questionGroups, answerSeq, protoNoSplit = state

        if self.verbosity >= 2:
            indent = '    '+''.join([ ('|  ' if answer != 0 else '   ')
                                      for answer in answerSeq ])
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

    def maxSplitAndNextStates(self, state, *splitInfos):
        labels, questionGroups, answerSeq, protoNoSplit = state

        splitInfos = list(splitInfos)
        splitInfos.append(SplitInfo(protoNoSplit, None, [protoNoSplit]))
        splitInfo = max(splitInfos, key = self.splitValuer)
        nextStates = self.getNextStates(state, splitInfo)
        return splitInfo, nextStates

    def computeBestSplitAndNextStates(self, state):
        labels, questionGroups, answerSeq, protoNoSplit = state

        splitInfos, questionGroupsOut = (
            self.getPossSplitsWithPrunedQuestionGroups(state, questionGroups)
        )
        splitInfos.append(SplitInfo(protoNoSplit, None, [protoNoSplit]))
        splitInfo = max(splitInfos, key = self.splitValuer)
        stateNew = labels, questionGroupsOut, answerSeq, protoNoSplit
        nextStates = self.getNextStates(stateNew, splitInfo)
        return splitInfo, nextStates

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
                    self.computeBestSplitAndNextStates,
                    msg = 'cluster:%schoose and perform split took' % indent
                )
            else:
                computeBestSplit = self.computeBestSplitAndNextStates
            splitInfo, nextStates = computeBestSplit(state)
            agenda.extend(reversed(nextStates))
            yield answerSeq, splitInfo

    def subTreeSplitInfoIterInGreedyOrder(self, stateInit):
        agenda = []

        def agendaPush(state):
            labels, questionGroups, answerSeq, protoNoSplit = state
            splitInfos, questionGroupsOut = (
                self.getPossSplitsWithPrunedQuestionGroups(state,
                                                           questionGroups)
            )
            splitInfos.append(SplitInfo(protoNoSplit, None, [protoNoSplit]))
            splitInfo = max(splitInfos, key = self.splitValuer)
            stateNew = labels, questionGroupsOut, answerSeq, protoNoSplit
            if splitInfo.fullQuestion is not None:
                heapq.heappush(
                    agenda,
                    (-self.splitValuer(splitInfo), splitInfo, stateNew)
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
                leafProtos.append(splitInfo.protoNoSplit)
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
    def __init__(self, utilitySpec, questionGroups,
                 estimateTotAux = d.getDefaultEstimateTotAuxNoRevert(),
                 verbosity = 2):
        self.utilitySpec = utilitySpec
        self.questionGroups = questionGroups
        self.estimateTotAux = estimateTotAux
        self.verbosity = verbosity

@codeDeps(AccSummer, DecisionTreeClusterer, LeafEstimator, d.Rat,
    removeTrivialQuestions, timed
)
def decisionTreeCluster(clusteringSpec, labels, accForLabel, createAcc):
    verbosity = clusteringSpec.verbosity
    accSummer = AccSummer(accForLabel, createAcc)
    leafEstimator = LeafEstimator(clusteringSpec.estimateTotAux)
    def getProtoRoot():
        return leafEstimator.est(accSummer.sumAccs(labels))
    if verbosity >= 3:
        getProtoRoot = timed(getProtoRoot)
    protoRoot = getProtoRoot()
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = DecisionTreeClusterer(accSummer, leafEstimator, splitValuer,
                                      verbosity = verbosity)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with perLeafPenalty = %s and'
               ' minCount = %s' %
               (splitValuer.perLeafPenalty, splitValuer.minCount))

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
    leafEstimator = LeafEstimator(clusteringSpec.estimateTotAux)
    protoRoot = leafEstimator.est(accSummer.sumAccs(labels))
    splitValuer = clusteringSpec.utilitySpec(protoRoot.dist, protoRoot.count,
                                             verbosity = verbosity)
    clusterer = DecisionTreeClusterer(accSummer, leafEstimator, splitValuer,
                                      verbosity = verbosity)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with perLeafPenalty = %s and'
               ' minCount = %s' %
               (splitValuer.perLeafPenalty, splitValuer.minCount))

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
