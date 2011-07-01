"""Decision tree questions from HTS demo."""

# Copyright 2011 Matt Shannon
# The follow copyrights may apply to the list of questions:
#     Copyright 2001-2008 Nagoya Institute of Technology, Department of Computer Science
#     Copyright 2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import questions as ques
import phoneset_cmu

def getMonophoneQuestions():
    return ques.getSubsetQuestions('phone', phoneset_cmu.namedPhoneSubsets)
def getTriphoneQuestions():
    return (
        ques.getSubsetQuestions('l_phone', phoneset_cmu.namedPhoneSubsets) +
        ques.getSubsetQuestions('phone', phoneset_cmu.namedPhoneSubsets) +
        ques.getSubsetQuestions('r_phone', phoneset_cmu.namedPhoneSubsets)
    )
def getQuinphoneQuestions():
    return (
        ques.getSubsetQuestions('ll_phone', phoneset_cmu.namedPhoneSubsets) +
        ques.getSubsetQuestions('l_phone', phoneset_cmu.namedPhoneSubsets) +
        ques.getSubsetQuestions('phone', phoneset_cmu.namedPhoneSubsets) +
        ques.getSubsetQuestions('r_phone', phoneset_cmu.namedPhoneSubsets) +
        ques.getSubsetQuestions('rr_phone', phoneset_cmu.namedPhoneSubsets)
    )
def getQuestions():
    questionLists = [
        ques.getSubsetQuestions('ll_phone', phoneset_cmu.namedPhoneSubsets),
        ques.getSubsetQuestions('l_phone', phoneset_cmu.namedPhoneSubsets),
        ques.getSubsetQuestions('phone', phoneset_cmu.namedPhoneSubsets),
        ques.getSubsetQuestions('r_phone', phoneset_cmu.namedPhoneSubsets),
        ques.getSubsetQuestions('rr_phone', phoneset_cmu.namedPhoneSubsets),
        ques.getEqualityQuestions('seg_fw', [None] + range(1, 8)),
        ques.getThreshQuestions('seg_fw', range(1, 8)),
        ques.getEqualityQuestions('seg_bw', [None] + range(1, 8)),
        ques.getThreshQuestions('seg_bw', range(0, 8)),
        # FIXME : add more questions here
        [
            ques.EqualityQuestion('c_syl_vowel', None),
            ques.EqualityQuestion('c_syl_vowel', 'novowel')
        ],
        ques.getSubsetQuestions('c_syl_vowel', phoneset_cmu.namedVowelSubsets),
        # FIXME : add more questions here
    ]
    return [ question for questionList in questionLists for question in questionList ]
