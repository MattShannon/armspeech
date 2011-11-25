"""Jobs for unit tests for distributed computation execution.

N.B. this has to be in a separate module and not in test_distribute due to the
  fact pickling stores `__main__` references (rather than references to the
  appropriate module) for objects defined in the currently-running script.
"""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import distribute

class OneJob(distribute.Job):
    def __init__(self, name):
        self.name = name
        self.valueOut = self.newOutput()
        self.inputs = []
    def run(self, buildRepo):
        valueOut = 1

        buildRepo.saveToArt(self.valueOut, valueOut)

class AddJob(distribute.Job):
    def __init__(self, valueLeft, valueRight, name):
        self.name = name
        self.valueLeft = valueLeft
        self.valueRight = valueRight
        self.valueOut = self.newOutput()
        self.inputs = [ valueLeft, valueRight ]
    def run(self, buildRepo):
        valueLeft = buildRepo.loadFromArt(self.valueLeft)
        valueRight = buildRepo.loadFromArt(self.valueRight)

        valueOut = valueLeft + valueRight

        buildRepo.saveToArt(self.valueOut, valueOut)
