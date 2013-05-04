"""Clustering algorithms."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


import dist as d
from armspeech.util.mathhelp import assert_allclose
from armspeech.util.timing import timed
from codedep import codeDeps

import logging
import math
from collections import defaultdict

@codeDeps()
class ProtoLeaf(object):
    def __init__(self, dist, aux, auxRat, count):
        self.dist = dist
        self.aux = aux
        self.auxRat = auxRat
        self.count = count

@codeDeps(assert_allclose)
class SplitInfo(object):
    """Collected information for a (potential or actual) split."""
    def __init__(self, protoNoSplit, fullQuestion, protoYes, protoNo):
        self.protoNoSplit = protoNoSplit
        self.fullQuestion = fullQuestion
        self.protoYes = protoYes
        self.protoNo = protoNo

        assert self.protoNoSplit is not None
        if self.fullQuestion is None:
            assert self.protoYes is None and self.protoNo is None
        if self.protoYes is not None and self.protoNo is not None:
            assert_allclose(self.protoYes.count + self.protoNo.count,
                            self.protoNoSplit.count)

    def delta(self):
        """Returns the delta for this split.

        The delta is used to choose which question to use to split a given node
        and to decide whether to split at all.
        """
        if self.protoYes is None or self.protoNo is None:
            return float('-inf')
        else:
            return self.protoYes.aux + self.protoNo.aux - self.protoNoSplit.aux

@codeDeps(SplitInfo)
def maxSplit(protoNoSplit, splitInfos):
    bestSplitInfo = SplitInfo(protoNoSplit, None, None, None)
    for splitInfo in splitInfos:
        if splitInfo.delta() > bestSplitInfo.delta():
            bestSplitInfo = splitInfo
    return bestSplitInfo

@codeDeps()
class Grower(object):
    def allowSplit(self, splitInfo):
        abstract

    def useSplit(self, splitInfo):
        abstract

@codeDeps(ProtoLeaf, SplitInfo, d.DecisionTreeLeaf, d.DecisionTreeNode,
    d.EstimationError, d.addAcc, d.sumValuedRats, maxSplit, timed
)
class DecisionTreeClusterer(object):
    def __init__(self, accForLabel, questionGroups, createAcc, estimateTotAux,
                 grower, verbosity):
        self.accForLabel = accForLabel
        self.questionGroups = questionGroups
        self.createAcc = createAcc
        self.estimateTotAux = estimateTotAux
        self.grower = grower
        self.verbosity = verbosity

    def withGrower(self, grower):
        return DecisionTreeClusterer(
            self.accForLabel, self.questionGroups, self.createAcc,
            self.estimateTotAux, grower, self.verbosity
        )

    def sumAccs(self, labels):
        accForLabel = self.accForLabel
        accTot = self.createAcc()
        for label in labels:
            d.addAcc(accTot, accForLabel(label))
        return accTot

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
        accYes = self.createAcc()
        accNo = self.createAcc()
        for labelValue, acc in labelValueToAcc.iteritems():
            if question(labelValue):
                d.addAcc(accYes, acc)
            else:
                d.addAcc(accNo, acc)

        return accYes, accNo

    def getProtoLeaf(self, acc):
        try:
            dist, (aux, auxRat) = self.estimateTotAux(acc)
        except d.EstimationError:
            return None
        count = acc.count()
        return ProtoLeaf(dist, aux, auxRat, count)

    def splitInfosIter(self, state, questionGroups):
        """Returns an iterator with one SplitInfo for each allowed question.

        A clustering state is a proto-leaf together with info about its
        position in the tree and enough info about the initial parts of the
        tree to allow clustering of the sub-tree with the proto-leaf as root.
        (This initial tree info is just the labels remaining).

        For a given state and list of questionGroups, this function returns an
        iterator over splits, one for each allowed question.
        """
        labels, isYesList, protoNoSplit = state

        labelValueToAccs = self.sumAccsFirstLevel(labels, questionGroups)

        for (
            labelValueToAcc, (labelValuer, questions)
        ) in zip(labelValueToAccs, questionGroups):
            for question in questions:
                accYes, accNo = self.sumAccsSecondLevel(labelValueToAcc,
                                                        question)
                protoYes = self.getProtoLeaf(accYes)
                protoNo = self.getProtoLeaf(accNo)
                fullQuestion = labelValuer, question
                splitInfo = SplitInfo(protoNoSplit, fullQuestion,
                                      protoYes, protoNo)
                if self.grower.allowSplit(splitInfo):
                    yield splitInfo

    def decideSplit(self, state, splitInfos):
        labels, isYesList, protoNoSplit = state

        splitInfo = maxSplit(protoNoSplit, splitInfos)

        if self.verbosity >= 2:
            indent = '    '+''.join([ ('|  ' if isYes else '   ')
                                      for isYes in isYesList ])
        if self.grower.useSplit(splitInfo):
            labelValuer, question = splitInfo.fullQuestion
            if self.verbosity >= 2:
                print ('cluster:%squestion ( %s %s ) ( delta = %s )' %
                       (indent, labelValuer.shortRepr(), question.shortRepr(),
                        splitInfo.delta()))
            labelsYes = []
            labelsNo = []
            for label in labels:
                if question(labelValuer(label)):
                    labelsYes.append(label)
                else:
                    labelsNo.append(label)

            return splitInfo, [
                (labelsYes, isYesList + [True], splitInfo.protoYes),
                (labelsNo, isYesList + [False], splitInfo.protoNo),
            ]
        else:
            if self.verbosity >= 2:
                print 'cluster:'+indent+'leaf'
            return splitInfo, []

    def maxSplitAndDecide(self, state, *splitInfos):
        labels, isYesList, protoNoSplit = state

        splitInfo, nextStates = self.decideSplit(state, splitInfos)
        splitInfoDict = dict()
        splitInfoDict[tuple(isYesList)] = splitInfo
        return splitInfoDict, nextStates

    def computeBestSplitAndDecide(self, state):
        labels, isYesList, protoNoSplit = state

        if self.verbosity >= 3:
            indent = '    '+''.join([ ('|  ' if isYes else '   ')
                                      for isYes in isYesList ])
            decideSplit = timed(
                self.decideSplit,
                msg = 'cluster:%schoose split took' % indent
            )
        else:
            decideSplit = self.decideSplit

        splitInfos = self.splitInfosIter(state, self.questionGroups)
        splitInfo, nextStates = decideSplit(state, splitInfos)
        splitInfoDict = dict()
        splitInfoDict[tuple(isYesList)] = splitInfo
        return splitInfoDict, nextStates

    def printNodeInfo(self, state):
        labels, isYesList, protoNoSplit = state

        indent = '    '+''.join([ ('|  ' if isYes else '   ')
                                  for isYes in isYesList[:-1] ])
        if not isYesList:
            extra = ''
        elif isYesList[-1]:
            extra = '|->'
        else:
            extra = '\->'
        print ('cluster:%s%snode ( count = %s , remaining labels = %s )' %
               (indent, extra, protoNoSplit.count, len(labels)))

    def subTreeSplitInfoDict(self, stateInit):
        splitInfoDict = dict()
        agenda = [stateInit]
        while agenda:
            state = agenda.pop()
            if self.verbosity >= 2:
                self.printNodeInfo(state)
            splitInfoDictOne, nextStates = (
                self.computeBestSplitAndDecide(state)
            )
            assert all([ path not in splitInfoDict
                         for path in splitInfoDictOne ])
            splitInfoDict.update(splitInfoDictOne)
            agenda.extend(reversed(nextStates))
        return splitInfoDict

    def combineSplitInfoDicts(self, splitInfoDicts):
        splitInfoDictTot = dict()
        for splitInfoDict in splitInfoDicts:
            for path, splitInfo in splitInfoDict.iteritems():
                assert path not in splitInfoDictTot
                splitInfoDictTot[path] = splitInfo
        return splitInfoDictTot

    def growTree(self, splitInfoDict):
        def grow(isYesList):
            splitInfo = splitInfoDict[tuple(isYesList)]
            if self.grower.useSplit(splitInfo):
                distYes, auxValuedRatYes = grow(isYesList + [True])
                distNo, auxValuedRatNo = grow(isYesList + [False])
                auxValuedRat = d.sumValuedRats([auxValuedRatYes,
                                                auxValuedRatNo])
                distNew = d.DecisionTreeNode(splitInfo.fullQuestion,
                                             distYes, distNo)
                return distNew, auxValuedRat
            else:
                protoNoSplit = splitInfo.protoNoSplit
                auxValuedRat = protoNoSplit.aux, protoNoSplit.auxRat
                return d.DecisionTreeLeaf(protoNoSplit.dist), auxValuedRat

        return grow([])

@codeDeps(Grower)
class SimpleGrower(Grower):
    def __init__(self, thresh, minCount, maxCount = None):
        self.thresh = thresh
        self.minCount = minCount
        self.maxCount = maxCount

    def allowSplit(self, splitInfo):
        protoYes = splitInfo.protoYes
        protoNo = splitInfo.protoNo
        return (protoYes is not None and
                protoNo is not None and
                protoYes.count >= self.minCount and
                protoNo.count >= self.minCount)

    def useSplit(self, splitInfo):
        protoNoSplit = splitInfo.protoNoSplit
        allowNoSplit = (self.maxCount is None or
                        protoNoSplit.count <= self.maxCount)

        if splitInfo.fullQuestion is not None and (
            not allowNoSplit or splitInfo.delta() > self.thresh
        ):
            return True
        else:
            if not allowNoSplit:
                assert splitInfo.fullQuestion is None
                logging.warning('not splitting decision tree node even though'
                                ' count = %s > maxCount = %s, since no further'
                                ' splitting allowed' %
                                (protoNoSplit.count, self.maxCount))
            return False

@codeDeps(DecisionTreeClusterer, SimpleGrower, d.Rat,
    d.getDefaultEstimateTotAuxNoRevert, d.getDefaultParamSpec
)
def decisionTreeCluster(labels, accForLabel, createAcc, questionGroups,
                        thresh, minCount, maxCount = None, mdlFactor = 1.0,
                        estimateTotAux = d.getDefaultEstimateTotAuxNoRevert(),
                        paramSpec = d.getDefaultParamSpec(),
                        verbosity = 2):
    grower = SimpleGrower(thresh, minCount, maxCount)
    clusterer = DecisionTreeClusterer(accForLabel, questionGroups, createAcc,
                                      estimateTotAux, grower, verbosity)

    protoRoot = clusterer.getProtoLeaf(clusterer.sumAccs(labels))
    countRoot = protoRoot.count
    if thresh is None:
        numParamsPerLeaf = len(paramSpec.params(protoRoot.dist))
        thresh = 0.5 * mdlFactor * numParamsPerLeaf * math.log(countRoot + 1.0)
        if verbosity >= 1:
            print ('cluster: setting thresh using MDL: mdlFactor = %s and'
                   ' numParamsPerLeaf = %s and count = %s' %
                   (mdlFactor, numParamsPerLeaf, countRoot))
        grower = SimpleGrower(thresh, minCount, maxCount)
        clusterer = clusterer.withGrower(grower)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with thresh = %s and'
               ' minCount = %s and maxCount = %s' %
               (thresh, minCount, maxCount))

    splitInfoDict = clusterer.subTreeSplitInfoDict((labels, [], protoRoot))
    dist, (aux, auxRat) = clusterer.growTree(splitInfoDict)

    if verbosity >= 1:
        print 'cluster: %s leaves' % dist.countLeaves()
        print ('cluster: aux root = %s (%s) -> aux tree = %s (%s) (%s count)' %
               (protoRoot.aux / countRoot, d.Rat.toString(protoRoot.auxRat),
                aux / countRoot, d.Rat.toString(auxRat),
                countRoot))
    return dist
