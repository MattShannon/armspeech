"""Decision tree questions for baby phoneset."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import phoneset_baby

class PhoneSubsetQuestion(object):
    def __init__(self, phoneSubset, name):
        self.phoneSubset = phoneSubset
        self.name = name
    def __call__(self, phone):
        return phone in self.phoneSubset

def getQuestions():
    return [ PhoneSubsetQuestion(subset, subsetName) for subsetName, subset in phoneset_baby.namedPhoneSubsets ]
