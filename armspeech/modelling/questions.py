"""Representation for decision tree questions.

A full question consists of a label valuer together with a question. The label
valuer is a callable that maps a label to a value (e.g. extracts the left-hand
phone from a full-context label). A question is a callable that maps this
value to an answer, which is a value in some known codomain.
The codomain should be of the form range(n) for some n.
For example for standard decision tree questions the codomain consists of
0 (answering "no" to the question) and 1 (answering "yes" to the question).
"""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from codedep import codeDeps

@codeDeps()
class IdLabelValuer(object):
    def __init__(self, shortRepr = None):
        if shortRepr is None:
            shortRepr = 'label'
        self._shortRepr = shortRepr
    def __repr__(self):
        return 'IdLabelValuer(shortRepr=%r)' % self._shortRepr
    def shortRepr(self):
        return self._shortRepr
    def __call__(self, label):
        return label

@codeDeps()
class AttrLabelValuer(object):
    def __init__(self, labelKey, shortRepr = None):
        if shortRepr is None:
            shortRepr = '%s' % labelKey
        self.labelKey = labelKey
        self._shortRepr = shortRepr
    def __repr__(self):
        return ('AttrLabelValuer(%r, shortRepr=%r)' %
                (self.labelKey, self._shortRepr))
    def shortRepr(self):
        return self._shortRepr
    def __call__(self, label):
        return getattr(label, self.labelKey)

@codeDeps()
class TupleIdLabelValuer(object):
    def __init__(self, index, shortRepr = None):
        if shortRepr is None:
            shortRepr = '[%s]' % index
        self.index = index
        self._shortRepr = shortRepr
    def __repr__(self):
        return ('TupleIdLabelValuer(%r, shortRepr=%r)' %
                (self.index, self._shortRepr))
    def shortRepr(self):
        return self._shortRepr
    def __call__(self, label):
        return label[self.index]

@codeDeps()
class TupleAttrLabelValuer(object):
    def __init__(self, index, labelKey, shortRepr = None):
        if shortRepr is None:
            shortRepr = '[%s].%s' % (index, labelKey)
        self.index = index
        self.labelKey = labelKey
        self._shortRepr = shortRepr
    def __repr__(self):
        return ('TupleAttrLabelValuer(%r, %r, shortRepr=%r)' %
                (self.index, self.labelKey, self._shortRepr))
    def shortRepr(self):
        return self._shortRepr
    def __call__(self, label):
        return getattr(label[self.index], self.labelKey)

@codeDeps()
class Question(object):
    pass

@codeDeps(Question)
class SubsetQuestion(Question):
    def __init__(self, subset, name):
        self.subset = subset
        self.name = name
    def __repr__(self):
        return 'SubsetQuestion('+repr(self.subset)+', '+repr(self.name)+')'
    def shortRepr(self):
        return 'is '+self.name
    def __call__(self, value):
        return int(value in self.subset)
    def codomain(self):
        return range(2)

@codeDeps(Question)
class EqualityQuestion(Question):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return 'EqualityQuestion('+repr(self.value)+')'
    def shortRepr(self):
        return '== '+str(self.value)
    def __call__(self, value):
        return int(value == self.value)
    def codomain(self):
        return range(2)

@codeDeps(Question)
class ThreshQuestion(Question):
    def __init__(self, thresh):
        self.thresh = thresh
    def __repr__(self):
        return 'ThreshQuestion('+repr(self.thresh)+')'
    def shortRepr(self):
        return '<= '+str(self.thresh)
    def __call__(self, value):
        return int(value <= self.thresh)
    def codomain(self):
        return range(2)

@codeDeps(Question)
class CmpQuestion(Question):
    def __init__(self, thresh):
        self.thresh = thresh
    def __repr__(self):
        return 'CmpQuestion(%r)' % self.thresh
    def shortRepr(self):
        return 'vs '+str(self.thresh)
    def __call__(self, value):
        return cmp(value, self.thresh) + 1
    def codomain(self):
        return range(3)

@codeDeps(SubsetQuestion)
def getSubsetQuestions(namedSubsets):
    return [ SubsetQuestion(subset, subsetName) for subsetName, subset in namedSubsets ]
@codeDeps(EqualityQuestion)
def getEqualityQuestions(values):
    return [ EqualityQuestion(value) for value in values ]
@codeDeps(ThreshQuestion)
def getThreshQuestions(threshes):
    return [ ThreshQuestion(thresh) for thresh in threshes ]
@codeDeps(CmpQuestion)
def getCmpQuestions(threshes):
    return [ CmpQuestion(thresh) for thresh in threshes ]
