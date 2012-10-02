"""Clustering algorithms."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


import dist as d
from armspeech.util.mathhelp import assert_allclose
from armspeech.util.timing import timed

import logging
import math
from collections import defaultdict

def decisionTreeCluster(labelList, accForLabel, createAcc, questionGroups, thresh, minCount, maxCount = None, mdlFactor = 1.0, verbosity = 2):
    root = createAcc()
    for label in labelList:
        d.addAcc(root, accForLabel(label))
    countRoot = root.count()
    distRoot, (auxRoot, auxRootRat) = d.defaultEstimateTotAux(root)
    if thresh is None:
        numParamsPerLeaf = len(d.defaultParamSpec.params(distRoot))
        thresh = 0.5 * mdlFactor * numParamsPerLeaf * math.log(countRoot + 1.0)
        if verbosity >= 1:
            print 'cluster: setting thresh using MDL: mdlFactor =', mdlFactor, 'and numParamsPerLeaf =', numParamsPerLeaf, 'and count =', countRoot
    if verbosity >= 1:
        print 'cluster: decision tree clustering with thresh =', thresh, 'and minCount =', minCount, 'and maxCount =', maxCount
    dist, aux = decisionTreeSubCluster(labelList, accForLabel, createAcc, questionGroups, thresh, minCount, maxCount, [], distRoot, auxRoot, countRoot, verbosity)
    if verbosity >= 1:
        print 'cluster: %s leaves' % dist.countLeaves()
        print 'cluster: aux root = %s (%s) -> aux tree = %s (%s count)' % (auxRoot / countRoot, d.Rat.toString(auxRootRat), aux / countRoot, countRoot)
    return dist

def decisionTreeSubCluster(labelList, accForLabel, createAcc, questionGroups, thresh, minCount, maxCount, isYesList, distLeaf, auxLeaf, countNode, verbosity):
    if verbosity >= 2:
        indent = '    '+''.join([ ('|  ' if isYes else '   ') for isYes in isYesList[:-1] ])
        if not isYesList:
            extra1 = ''
            extra2 = ''
        elif isYesList[-1]:
            extra1 = '|->'
            extra2 = '|  '
        else:
            extra1 = '\->'
            extra2 = '   '
        print 'cluster:'+indent+extra1+'node ( count =', countNode, ', remaining labels =', len(labelList), ')'
        indent += extra2

    getBestSplit = timed(decisionTreeGetBestSplit, msg = 'cluster:'+indent+'choose split took') if verbosity >= 3 else decisionTreeGetBestSplit
    bestFullQuestion, bestAux, bestEstimatedYes, bestEstimatedNo = getBestSplit(labelList, accForLabel, createAcc, questionGroups, minCount, auxLeaf)

    if bestFullQuestion is not None and (bestAux - auxLeaf > thresh or maxCount is not None and countNode > maxCount):
        bestLabelValuer, bestQuestion = bestFullQuestion
        if verbosity >= 2:
            print 'cluster:'+indent+'question (', bestLabelValuer.shortRepr()+' '+bestQuestion.shortRepr(), ')', '( delta =', bestAux - auxLeaf, ')'
        labelListYes = []
        labelListNo = []
        for label in labelList:
            if bestQuestion(bestLabelValuer(label)):
                labelListYes.append(label)
            else:
                labelListNo.append(label)
        distYesLeaf, auxYesLeaf, countYes = bestEstimatedYes
        distNoLeaf, auxNoLeaf, countNo = bestEstimatedNo
        assert_allclose(countYes + countNo, countNode)
        distYes, auxYes = decisionTreeSubCluster(labelListYes, accForLabel, createAcc, questionGroups, thresh, minCount, maxCount, isYesList + [True], distYesLeaf, auxYesLeaf, countYes, verbosity)
        distNo, auxNo = decisionTreeSubCluster(labelListNo, accForLabel, createAcc, questionGroups, thresh, minCount, maxCount, isYesList + [False], distNoLeaf, auxNoLeaf, countNo, verbosity)
        aux = auxYes + auxNo
        return d.DecisionTreeNode(bestFullQuestion, distYes, distNo), aux
    else:
        if maxCount is not None and countNode > maxCount:
            assert bestFullQuestion is None
            logging.warning('decision tree leaf has count = %s > maxCount = %s, but no further splitting possible' % (countNode, maxCount))
        if verbosity >= 2:
            print 'cluster:'+indent+'leaf'
        return d.DecisionTreeLeaf(distLeaf), auxLeaf

def decisionTreeGetBestSplit(labelList, accForLabel, createAcc, questionGroups, minCount, auxLeaf):
    # (N.B. Could probably get a further speed-up (over and above that
    #   obtained by using question groups) for EqualityQuestion and
    #   ThreshQuestion by doing clever stuff with subtracting accs (i.e.
    #   adding accs with negative occupancies).
    #   However not at all clear this would worth it in terms of speed-up
    #   achieved vs implementation complexity / fragility.
    # )
    bestFullQuestion = None
    bestAux = auxLeaf
    bestEstimatedYes = None
    bestEstimatedNo = None
    labelValueToAccs = [ defaultdict(createAcc) for questionGroup in questionGroups ]
    for label in labelList:
        acc = accForLabel(label)
        for labelValueToAcc, (labelValuer, questions) in zip(labelValueToAccs, questionGroups):
            d.addAcc(labelValueToAcc[labelValuer(label)], acc)
    for labelValueToAcc, (labelValuer, questions) in zip(labelValueToAccs, questionGroups):
        for question in questions:
            yes = createAcc()
            no = createAcc()
            for labelValue, acc in labelValueToAcc.iteritems():
                if question(labelValue):
                    d.addAcc(yes, acc)
                else:
                    d.addAcc(no, acc)
            if yes.count() > minCount and no.count() > minCount:
                try:
                    distYesLeaf, (auxYesLeaf, auxYesLeafRat) = d.defaultEstimateTotAux(yes)
                    distNoLeaf, (auxNoLeaf, auxNoLeafRat) = d.defaultEstimateTotAux(no)
                except d.EstimationError:
                    pass
                else:
                    aux = auxYesLeaf + auxNoLeaf
                    if bestFullQuestion is None or aux > bestAux:
                        bestFullQuestion = labelValuer, question
                        bestAux = aux
                        bestEstimatedYes = distYesLeaf, auxYesLeaf, yes.count()
                        bestEstimatedNo = distNoLeaf, auxNoLeaf, no.count()
    return bestFullQuestion, bestAux, bestEstimatedYes, bestEstimatedNo
