"""Clustering algorithms."""

# Copyright 2011, 2012 Matt Shannon

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

@codeDeps()
class Grower(object):
    def allowSplit(self, splitInfo):
        abstract

    def useSplit(self, splitInfo):
        abstract

@codeDeps(ProtoLeaf, SplitInfo, d.DecisionTreeLeaf, d.DecisionTreeNode,
    d.EstimationError, d.addAcc, d.sumValuedRats, timed
)
class DecisionTreeClusterer(object):
    def __init__(self, accForLabel, questionGroups, createAcc, estimateTotAux,
                 verbosity):
        self.accForLabel = accForLabel
        self.questionGroups = questionGroups
        self.createAcc = createAcc
        self.estimateTotAux = estimateTotAux
        self.verbosity = verbosity

    def getAccFromLabels(self, labels):
        accForLabel = self.accForLabel
        accTot = self.createAcc()
        for label in labels:
            d.addAcc(accTot, accForLabel(label))
        return accTot

    def getProto(self, acc):
        try:
            dist, (aux, auxRat) = self.estimateTotAux(acc)
        except d.EstimationError:
            return None
        count = acc.count()
        return ProtoLeaf(dist, aux, auxRat, count)

    def findBestSplit(self, protoNoSplit, splitInfos):
        bestSplitInfo = SplitInfo(protoNoSplit, None, None, None)
        for splitInfo in splitInfos:
            if splitInfo.delta() > bestSplitInfo.delta():
                bestSplitInfo = splitInfo
        return bestSplitInfo

    def getBestSplit(self, labels, protoNoSplit, grower, questionGroups):
        def getProtosForQuestion(labelValueToAcc, question):
            accYes = self.createAcc()
            accNo = self.createAcc()
            for labelValue, acc in labelValueToAcc.iteritems():
                if question(labelValue):
                    d.addAcc(accYes, acc)
                else:
                    d.addAcc(accNo, acc)

            return self.getProto(accYes), self.getProto(accNo)

        accForLabel = self.accForLabel
        labelValuers = [ labelValuer
                         for labelValuer, questions in questionGroups ]
        labelValueToAccs = [ defaultdict(self.createAcc)
                             for questionGroup in questionGroups ]
        labelValueToAccAndLabelValuers = zip(labelValueToAccs, labelValuers)
        for label in labels:
            acc = accForLabel(label)
            for labelValueToAcc, labelValuer in labelValueToAccAndLabelValuers:
                d.addAcc(labelValueToAcc[labelValuer(label)], acc)

        def getSplitInfos(labelValueToAccs, questionGroups):
            for (
                labelValueToAcc, (labelValuer, questions)
            ) in zip(labelValueToAccs, questionGroups):
                for question in questions:
                    protoYes, protoNo = getProtosForQuestion(labelValueToAcc,
                                                             question)
                    fullQuestion = labelValuer, question
                    splitInfo = SplitInfo(protoNoSplit, fullQuestion,
                                          protoYes, protoNo)
                    if grower.allowSplit(splitInfo):
                        yield splitInfo

        splitInfos = getSplitInfos(labelValueToAccs, questionGroups)
        return self.findBestSplit(protoNoSplit, splitInfos)

    def clusterSub(self, labels, isYesList, protoNoSplit, grower):
        if self.verbosity >= 2:
            indent = '    '+''.join([ ('|  ' if isYes else '   ')
                                      for isYes in isYesList[:-1] ])
            if not isYesList:
                extra1 = ''
                extra2 = ''
            elif isYesList[-1]:
                extra1 = '|->'
                extra2 = '|  '
            else:
                extra1 = '\->'
                extra2 = '   '
            print ('cluster:%s%snode ( count = %s , remaining labels = %s )' %
                   (indent, extra1, protoNoSplit.count, len(labels)))
            indent += extra2

        if self.verbosity >= 3:
            splitInfo = timed(
                self.getBestSplit,
                msg = 'cluster:%schoose split took' % indent
            )(labels, protoNoSplit, grower, self.questionGroups)
        else:
            splitInfo = self.getBestSplit(labels, protoNoSplit, grower,
                                          self.questionGroups)

        if grower.useSplit(splitInfo):
            fullQuestion = splitInfo.fullQuestion
            labelValuer, question = fullQuestion
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
            protoYes = splitInfo.protoYes
            protoNo = splitInfo.protoNo
            distYes, auxValuedRatYes = self.clusterSub(labelsYes,
                                                       isYesList + [True],
                                                       protoYes, grower)
            distNo, auxValuedRatNo = self.clusterSub(labelsNo,
                                                     isYesList + [False],
                                                     protoNo, grower)
            auxValuedRat = d.sumValuedRats([auxValuedRatYes, auxValuedRatNo])
            distNew = d.DecisionTreeNode(fullQuestion, distYes, distNo)
            return distNew, auxValuedRat
        else:
            if self.verbosity >= 2:
                print 'cluster:'+indent+'leaf'
            auxValuedRat = protoNoSplit.aux, protoNoSplit.auxRat
            return d.DecisionTreeLeaf(protoNoSplit.dist), auxValuedRat

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
    clusterer = DecisionTreeClusterer(accForLabel, questionGroups, createAcc,
                                      estimateTotAux, verbosity)

    protoRoot = clusterer.getProto(clusterer.getAccFromLabels(labels))
    countRoot = protoRoot.count
    if thresh is None:
        numParamsPerLeaf = len(paramSpec.params(protoRoot.dist))
        thresh = 0.5 * mdlFactor * numParamsPerLeaf * math.log(countRoot + 1.0)
        if verbosity >= 1:
            print ('cluster: setting thresh using MDL: mdlFactor = %s and'
                   ' numParamsPerLeaf = %s and count = %s' %
                   (mdlFactor, numParamsPerLeaf, countRoot))
    grower = SimpleGrower(thresh, minCount, maxCount)
    if verbosity >= 1:
        print ('cluster: decision tree clustering with thresh = %s and'
               ' minCount = %s and maxCount = %s' %
               (thresh, minCount, maxCount))
    dist, (aux, auxRat) = clusterer.clusterSub(labels, [], protoRoot, grower)
    if verbosity >= 1:
        print 'cluster: %s leaves' % dist.countLeaves()
        print ('cluster: aux root = %s (%s) -> aux tree = %s (%s) (%s count)' %
               (protoRoot.aux / countRoot, d.Rat.toString(protoRoot.auxRat),
                aux / countRoot, d.Rat.toString(auxRat),
                countRoot))
    return dist
