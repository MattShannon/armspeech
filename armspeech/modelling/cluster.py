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

    def score(self):
        """Returns a score for this proto-leaf.

        A score is an abstract quantity that is only used when choosing which
        question to use for a potential split and whether to split at all.
        """
        return self.aux

@codeDeps()
class SplitInfo(object):
    """Collected information for a (potential or actual) split."""
    def __init__(self, fullQuestion, protoYes, protoNo):
        self.fullQuestion = fullQuestion
        self.protoYes = protoYes
        self.protoNo = protoNo

    def score(self):
        """Returns a score for this split.

        A score is an abstract quantity that is only used when choosing which
        question to use for a potential split and whether to split at all.
        """
        return self.protoYes.score() + self.protoNo.score()

@codeDeps()
class Grower(object):
    def allowSplit(self, protoYes, protoNo):
        abstract

    def useSplit(self, protoNoSplit, splitInfo):
        abstract

@codeDeps(ProtoLeaf, SplitInfo, assert_allclose, d.DecisionTreeLeaf,
    d.DecisionTreeNode, d.EstimationError, d.addAcc, d.sumValuedRats, timed
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

    def findBestSplit(self, splitInfos):
        bestSplitInfo = None
        for splitInfo in splitInfos:
            if splitInfo is not None:
                if bestSplitInfo is None or (splitInfo.score() >
                                             bestSplitInfo.score()):
                    bestSplitInfo = splitInfo
        return bestSplitInfo

    def getBestSplit(self, labels, grower, questionGroups):
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
                    if grower.allowSplit(protoYes, protoNo):
                        fullQuestion = labelValuer, question
                        yield SplitInfo(fullQuestion, protoYes, protoNo)

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
            )(labels, grower, self.questionGroups)
        else:
            splitInfo = self.getBestSplit(labels, grower, self.questionGroups)

        if grower.useSplit(protoNoSplit, splitInfo):
            fullQuestion = splitInfo.fullQuestion
            labelValuer, question = fullQuestion
            if self.verbosity >= 2:
                print ('cluster:%squestion ( %s %s ) ( delta = %s )' %
                       (indent, labelValuer.shortRepr(), question.shortRepr(),
                        splitInfo.score() - protoNoSplit.score()))
            labelsYes = []
            labelsNo = []
            for label in labels:
                if question(labelValuer(label)):
                    labelsYes.append(label)
                else:
                    labelsNo.append(label)
            protoYes = splitInfo.protoYes
            protoNo = splitInfo.protoNo
            assert_allclose(protoYes.count + protoNo.count, protoNoSplit.count)
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

    def allowSplit(self, protoYes, protoNo):
        return (protoYes is not None and
                protoNo is not None and
                protoYes.count >= self.minCount and
                protoNo.count >= self.minCount)

    def useSplit(self, protoNoSplit, splitInfo):
        allowNoSplit = (self.maxCount is None or
                        protoNoSplit.count <= self.maxCount)

        if splitInfo is not None and (
            not allowNoSplit or
            splitInfo.score() - protoNoSplit.score() > self.thresh
        ):
            return True
        else:
            if not allowNoSplit:
                assert splitInfo is None
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
