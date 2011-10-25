"""Example decision tree questions used for testing."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import questions as ques

a = 'a'
b = 'b'
i = 'i'
m = 'm'
sil = 'sil'

phoneList = [
    a,
    b,
    i,
    m,
    sil
]

namedPhoneSubsetList = [
    ['vowel', [a, i]],
    ['consonant', [b, m]],
    ['b', [b]],
    ['m', [m]],
    ['i', [i]],
    ['a', [a]],
    ['silence', [sil]],
]
namedPhoneSubsets = [ (subsetName, frozenset(subsetList)) for subsetName, subsetList in namedPhoneSubsetList ]

def getQuestionGroups():
    return [(ques.IdLabelValuer(), ques.getSubsetQuestions(namedPhoneSubsets))]
