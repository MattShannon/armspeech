"""Decision tree questions from HTS demo."""

# Copyright 2011 Matt Shannon
# The follow copyrights may apply to the list of questions:
#     Copyright 2001-2008 Nagoya Institute of Technology, Department of Computer Science
#     Copyright 2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.modelling.questions as ques
import phoneset_cmu

def getSubsetQG(labelKey, namedSubsets):
    return (ques.AttrLabelValuer(labelKey), ques.getSubsetQuestions(namedSubsets))
def getEqualityQG(labelKey, values):
    return (ques.AttrLabelValuer(labelKey), ques.getEqualityQuestions(values))
def getThreshQG(labelKey, threshes):
    return (ques.AttrLabelValuer(labelKey), ques.getThreshQuestions(threshes))
def getSubsetQGs(labelKeys, namedSubsets):
    questions = ques.getSubsetQuestions(namedSubsets)
    return [ (ques.AttrLabelValuer(labelKey), questions) for labelKey in labelKeys ]

def getMonophoneQuestionGroups():
    return getSubsetQGs(
        ['phone'],
        phoneset_cmu.namedPhoneSubsets
    )
def getTriphoneQuestionGroups():
    return getSubsetQGs(
        ['l_phone', 'phone', 'r_phone'],
        phoneset_cmu.namedPhoneSubsets
    )
def getQuinphoneQuestionGroups():
    return getSubsetQGs(
        ['ll_phone', 'l_phone', 'phone', 'r_phone', 'rr_phone'],
        phoneset_cmu.namedPhoneSubsets
    )
def getFullContextQuestionGroups():
    quinphoneQuestionGroups = getQuinphoneQuestionGroups()
    otherQuestionGroups = [
        getEqualityQG('seg_fw', [None] + range(1, 8)),
        getThreshQG('seg_fw', range(1, 8)),
        getEqualityQG('seg_bw', [None] + range(1, 8)),
        getThreshQG('seg_bw', range(0, 8)),
        # FIXME : add more questions here
        (ques.AttrLabelValuer('c_syl_vowel'),
            [
                ques.EqualityQuestion(None),
                ques.EqualityQuestion('novowel')
            ] + ques.getSubsetQuestions(phoneset_cmu.namedVowelSubsets)
        ),
        # FIXME : add more questions here
    ]
    return quinphoneQuestionGroups + otherQuestionGroups
