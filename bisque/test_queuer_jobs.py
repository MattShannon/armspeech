"""Jobs for unit tests for distributed computation execution.

N.B. this has to be in a separate module and not in test_distribute due to the
  fact pickling stores `__main__` references (rather than references to the
  appropriate module) for objects defined in the currently-running script.
"""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import distribute
from codedep import codeDeps

@codeDeps()
def getOne():
    return 1

@codeDeps()
def add(x, y):
    return x + y

@codeDeps(distribute.Job, distribute.JobArtifact)
class OneJob(distribute.Job):
    def __init__(self, name = 'oneJob'):
        self.name = name

        self.inputs = []
        self.valueOut = distribute.JobArtifact(parentJob = self)

        self.secHash()

    def run(self, buildRepo):
        valueOut = 1

        self.valueOut.saveValue(buildRepo, valueOut)

@codeDeps(distribute.Job, distribute.JobArtifact)
class AddJob(distribute.Job):
    def __init__(self, valueLeft, valueRight, name = 'addJob'):
        self.valueLeft = valueLeft
        self.valueRight = valueRight
        self.name = name

        self.inputs = [valueLeft, valueRight]
        self.valueOut = distribute.JobArtifact(parentJob = self)

        self.secHash()

    def run(self, buildRepo):
        valueLeft = self.valueLeft.loadValue(buildRepo)
        valueRight = self.valueRight.loadValue(buildRepo)

        valueOut = valueLeft + valueRight

        self.valueOut.saveValue(buildRepo, valueOut)
