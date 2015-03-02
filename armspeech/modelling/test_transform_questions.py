"""Example decision tree questions used for testing."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from codedep import codeDeps

import armspeech.modelling.questions as ques

@codeDeps()
class SimplePhoneset(object):
    def __init__(self):
        a = 'a'
        b = 'b'
        i = 'i'
        m = 'm'
        sil = 'sil'

        self.phoneList = [
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
        self.namedPhoneSubsets = [ (subsetName, frozenset(subsetList)) for subsetName, subsetList in namedPhoneSubsetList ]

@codeDeps(ques.IdLabelValuer, ques.getSubsetQuestions)
def getQuestionGroups(phoneset):
    return [(ques.IdLabelValuer(), ques.getSubsetQuestions(phoneset.namedPhoneSubsets))]
