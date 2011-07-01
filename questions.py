"""Representation for decision tree questions."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import collections

class Question(object):
    def __call__(self, label):
        return self.isYes(getattr(label, self.labelKey))

class SubsetQuestion(Question):
    def __init__(self, labelKey, subset, name):
        self.labelKey = labelKey
        self.subset = subset
        self.name = name
    def isYes(self, labelValue):
        return labelValue in self.subset

class EqualityQuestion(Question):
    def __init__(self, labelKey, value, name = None):
        self.labelKey = labelKey
        self.value = value
        if name == None:
            name = self.labelKey+' == '+str(self.value)
        self.name = name
    def isYes(self, labelValue):
        return labelValue == self.value

class ThreshQuestion(Question):
    def __init__(self, labelKey, thresh, name = None):
        self.labelKey = labelKey
        self.thresh = thresh
        if name == None:
            name = self.labelKey+' <= '+str(self.thresh)
        self.name = name
    def isYes(self, labelValue):
        return labelValue <= self.thresh

def getSubsetQuestions(labelKey, namedSubsets):
    return [ SubsetQuestion(labelKey, subset, labelKey+'-'+subsetName) for subsetName, subset in namedSubsets ]
def getEqualityQuestions(labelKey, values):
    return [ EqualityQuestion(labelKey, value) for value in values ]
def getThreshQuestions(labelKey, threshes):
    return [ ThreshQuestion(labelKey, thresh) for thresh in threshes ]

def groupQuestions(questions):
    questionsFor = collections.defaultdict(list)
    for question in questions:
        questionsFor[question.labelKey].append(question)
    return questionsFor.items()
