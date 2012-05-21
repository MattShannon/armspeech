"""Clustering algorithms."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


import dist as d
from armspeech.util.mathhelp import assert_allclose
from armspeech.util.timing import timed

import math
from collections import defaultdict

def decisionTreeCluster(labelList, accForLabel, createAcc, questionGroups, thresh, minOcc, maxOcc = None, mdlFactor = 1.0, verbosity = 2):
    root = createAcc()
    for label in labelList:
        d.addAcc(root, accForLabel(label))
    distRoot, logLikeRoot, occRoot = d.defaultEstimate(root)
    if thresh is None:
        numParamsPerLeaf = len(d.defaultParamSpec.params(distRoot))
        totalOcc = root.occ
        thresh = 0.5 * mdlFactor * numParamsPerLeaf * math.log(totalOcc + 1.0)
        if verbosity >= 1:
            print 'cluster: setting thresh using MDL: mdlFactor =', mdlFactor, 'and numParamsPerLeaf =', numParamsPerLeaf, 'and occ =', totalOcc
    if verbosity >= 1:
        print 'cluster: decision tree clustering with thresh =', thresh, 'and minOcc =', minOcc, 'and maxOcc =', maxOcc
    dist, logLike, occ = decisionTreeSubCluster(labelList, accForLabel, createAcc, questionGroups, thresh, minOcc, maxOcc, [], distRoot, logLikeRoot, occRoot, verbosity)
    assert_allclose(occ, occRoot)
    if verbosity >= 1:
        print 'cluster: log likelihood after =', logLike / occ, '('+str(occ)+' frames)', '('+str(dist.countLeaves())+' leaves)'
    return dist, logLike, occ

def decisionTreeSubCluster(labelList, accForLabel, createAcc, questionGroups, thresh, minOcc, maxOcc, isYesList, distLeaf, logLikeLeaf, occNode, verbosity):
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
        print 'cluster:'+indent+extra1+'node ( occ =', occNode, ', remaining labels =', len(labelList), ')'
        indent += extra2

    getBestSplit = timed(decisionTreeGetBestSplit, msg = 'cluster:'+indent+'choose split took') if verbosity >= 3 else decisionTreeGetBestSplit
    bestFullQuestion, bestLogLike, bestEstimatedYes, bestEstimatedNo = getBestSplit(labelList, accForLabel, createAcc, questionGroups, minOcc, logLikeLeaf, occNode)

    if bestFullQuestion is not None and (bestLogLike - logLikeLeaf > thresh or maxOcc is not None and occNode > maxOcc):
        bestLabelValuer, bestQuestion = bestFullQuestion
        if verbosity >= 2:
            print 'cluster:'+indent+'question (', bestLabelValuer.shortRepr()+' '+bestQuestion.shortRepr(), ')', '( delta =', bestLogLike - logLikeLeaf, ')'
        labelListYes = []
        labelListNo = []
        for label in labelList:
            if bestQuestion(bestLabelValuer(label)):
                labelListYes.append(label)
            else:
                labelListNo.append(label)
        distYesLeaf, logLikeYesLeaf, occYes = bestEstimatedYes
        distNoLeaf, logLikeNoLeaf, occNo = bestEstimatedNo
        assert_allclose(occYes + occNo, occNode)
        distYes, logLikeYes, occYes = decisionTreeSubCluster(labelListYes, accForLabel, createAcc, questionGroups, thresh, minOcc, maxOcc, isYesList + [True], distYesLeaf, logLikeYesLeaf, occYes, verbosity)
        distNo, logLikeNo, occNo = decisionTreeSubCluster(labelListNo, accForLabel, createAcc, questionGroups, thresh, minOcc, maxOcc, isYesList + [False], distNoLeaf, logLikeNoLeaf, occNo, verbosity)
        logLike = logLikeYes + logLikeNo
        occ = occYes + occNo
        assert_allclose(occ, occNode)
        return d.DecisionTreeNode(bestFullQuestion, distYes, distNo), logLike, occ
    else:
        if verbosity >= 2:
            print 'cluster:'+indent+'leaf'
        return d.DecisionTreeLeaf(distLeaf), logLikeLeaf, occNode

def decisionTreeGetBestSplit(labelList, accForLabel, createAcc, questionGroups, minOcc, logLikeLeaf, occNode):
    # (N.B. Could probably get a further speed-up (over and above that
    #   obtained by using question groups) for EqualityQuestion and
    #   ThreshQuestion by doing clever stuff with subtracting accs (i.e.
    #   adding accs with negative occupancies).
    #   However not at all clear this would worth it in terms of speed-up
    #   achieved vs implementation complexity / fragility.
    # )
    bestFullQuestion = None
    bestLogLike = logLikeLeaf
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
            assert_allclose(yes.occ + no.occ, occNode)
            if yes.occ > minOcc and no.occ > minOcc:
                try:
                    distYesLeaf, logLikeYesLeaf, occYes = d.defaultEstimate(yes)
                    distNoLeaf, logLikeNoLeaf, occNo = d.defaultEstimate(no)
                except d.EstimationError:
                    pass
                else:
                    assert_allclose(occYes + occNo, occNode)
                    logLike = logLikeYesLeaf + logLikeNoLeaf
                    if bestFullQuestion is None or logLike > bestLogLike:
                        bestFullQuestion = labelValuer, question
                        bestLogLike = logLike
                        bestEstimatedYes = distYesLeaf, logLikeYesLeaf, occYes
                        bestEstimatedNo = distNoLeaf, logLikeNoLeaf, occNo
    return bestFullQuestion, bestLogLike, bestEstimatedYes, bestEstimatedNo
