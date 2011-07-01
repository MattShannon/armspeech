"""Jobs for unit tests for distributed computation.

N.B. this has to be in a separate module and not in test_distribute due to the
  fact pickling stores `__main__` references (rather than references to the
  appropriate module) for objects defined in the currently-running script.
"""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import persist
import distribute
import dist as d
from model import defaultCreateAcc, defaultEstimate

class DebugJob(distribute.Job):
    def __init__(self, repo, name):
        self.repo = repo
        self.name = name
        self.inputs = []
    def run(self):
        print 'DEBUG: running job', self.name

class ZeroJob(distribute.Job):
    def __init__(self, repo, name):
        self.repo = repo
        self.name = name
        self.valueOut = self.newOutput()
        self.inputs = []
    def run(self):
        value = 0

        persist.savePickle(self.valueOut.location, value)

class AddOneJob(distribute.Job):
    def __init__(self, repo, valueIn, name):
        self.repo = repo
        self.name = name
        self.valueIn = valueIn
        self.valueOut = self.newOutput()
        self.inputs = [ valueIn ]
    def run(self):
        value = persist.loadPickle(self.valueIn.location)

        value += 1

        persist.savePickle(self.valueOut.location, value)

class InitAccJob(distribute.Job):
    def __init__(self, repo, corpus, name):
        self.repo = repo
        self.corpus = corpus
        self.name = name
        self.accOut = self.newOutput()
        self.inputs = []
    def run(self):
        acc = d.LinearGaussianAcc(inputLength = 1)
        for input, output in self.corpus:
            acc.add(input, output)

        persist.savePickle(self.accOut.location, acc)

class AccJob(distribute.Job):
    def __init__(self, repo, corpus, distIn, name):
        self.repo = repo
        self.corpus = corpus
        self.name = name
        self.distIn = distIn
        self.accOut = self.newOutput()
        self.inputs = [distIn]
    def run(self):
        dist = persist.loadPickle(self.distIn.location)

        acc = defaultCreateAcc(dist)
        for input, output in self.corpus:
            acc.add(input, output)

        persist.savePickle(self.accOut.location, acc)

class EstimateJob(distribute.Job):
    def __init__(self, repo, accIn, name):
        self.repo = repo
        self.name = name
        self.accIn = accIn
        self.distOut = self.newOutput()
        self.inputs = [accIn]
    def run(self):
        acc = persist.loadPickle(self.accIn.location)

        dist, trainLogLike, trainOcc = defaultEstimate(acc)

        persist.savePickle(self.distOut.location, dist)

def createSplitAccJobs(repo, corpusList, distIn, name):
    return [ AccJob(repo, corpusList[corpusIndex], distIn, name+'.'+str(corpusIndex)) for corpusIndex in range(len(corpusList)) ]

class SplitEstimateJob(distribute.Job):
    def __init__(self, repo, accsIn, name):
        self.repo = repo
        self.name = name
        self.accsIn = accsIn
        self.distOut = self.newOutput()
        self.inputs = accsIn
    def run(self):
        accs = [ persist.loadPickle(accIn.location) for accIn in self.accsIn ]

        accFull = accs[0]
        for acc in accs[1:]:
            d.addAcc(accFull, acc)

        dist, trainLogLike, trainOcc = defaultEstimate(accFull)

        persist.savePickle(self.distOut.location, dist)
