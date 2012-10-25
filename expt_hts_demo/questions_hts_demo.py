"""Decision tree questions from HTS demo."""

# Copyright 2011, 2012 Matt Shannon
# The follow copyrights may apply to the list of questions:
#     Copyright 2001-2008 Nagoya Institute of Technology, Department of
#                         Computer Science
#     Copyright 2001-2008 Tokyo Institute of Technology, Interdisciplinary
#                         Graduate School of Science and Engineering

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.modelling.questions as ques
from expt_hts_demo import labels_hts_demo
from codedep import codeDeps

@codeDeps(ques.AttrLabelValuer, ques.getSubsetQuestions)
def getSubsetQG(labelKey, namedSubsets):
    return (ques.AttrLabelValuer(labelKey),
            ques.getSubsetQuestions(namedSubsets))

@codeDeps(ques.AttrLabelValuer, ques.getEqualityQuestions)
def getEqualityQG(labelKey, values):
    return (ques.AttrLabelValuer(labelKey),
            ques.getEqualityQuestions(values))

@codeDeps(ques.AttrLabelValuer, ques.getThreshQuestions)
def getThreshQG(labelKey, threshes):
    return (ques.AttrLabelValuer(labelKey),
            ques.getThreshQuestions(threshes))

@codeDeps(ques.AttrLabelValuer, ques.getEqualityQuestions,
    ques.getThreshQuestions
)
def getEqualityThreshQG(labelKey, sortedValues, threshes = None):
    values = sortedValues
    if threshes is None:
        threshes = sortedValues[1:]
    return (ques.AttrLabelValuer(labelKey),
            (ques.getEqualityQuestions(values) +
             ques.getThreshQuestions(threshes)))

@codeDeps(ques.AttrLabelValuer, ques.getSubsetQuestions)
def getSubsetQGs(labelKeys, namedSubsets):
    questions = ques.getSubsetQuestions(namedSubsets)
    return [ (ques.AttrLabelValuer(labelKey), questions)
             for labelKey in labelKeys ]

@codeDeps(getSubsetQGs)
def getMonophoneQuestionGroups(phoneset):
    return getSubsetQGs(
        ['phone'],
        phoneset.namedPhoneSubsets
    )

@codeDeps(getSubsetQGs)
def getTriphoneQuestionGroups(phoneset):
    return getSubsetQGs(
        ['l_phone', 'phone', 'r_phone'],
        phoneset.namedPhoneSubsets
    )

@codeDeps(getSubsetQGs)
def getQuinphoneQuestionGroups(phoneset):
    return getSubsetQGs(
        ['ll_phone', 'l_phone', 'phone', 'r_phone', 'rr_phone'],
        phoneset.namedPhoneSubsets
    )

@codeDeps(getEqualityQG, getEqualityThreshQG, getQuinphoneQuestionGroups,
    labels_hts_demo.getGposList, labels_hts_demo.getTobiList,
    ques.AttrLabelValuer, ques.EqualityQuestion, ques.getSubsetQuestions
)
def getFullContextQuestionGroups(phoneset):
    quinphoneQuestionGroups = getQuinphoneQuestionGroups(phoneset)
    otherQuestionGroups = [
        getEqualityThreshQG('seg_fw', [None] + range(1, 8)),
        getEqualityThreshQG('seg_bw', [None] + range(1, 8), range(0, 8)),
        getEqualityQG('l_syl_stress', [True]),
        getEqualityQG('l_syl_accent', [True]),
        getEqualityThreshQG('l_syl_num_segs', range(0, 8)),
        getEqualityQG('c_syl_stress', [True, False, None]),
        getEqualityQG('c_syl_accent', [True, False, None]),
        getEqualityThreshQG('c_syl_num_segs', [None] + range(1, 8)),
        getEqualityThreshQG('pos_c_syl_in_c_word_fw', [None] + range(1, 8)),
        getEqualityThreshQG('pos_c_syl_in_c_word_bw', [None] + range(1, 8)),
        getEqualityThreshQG('pos_c_syl_in_c_phrase_fw', [None] + range(1, 21)),
        getEqualityThreshQG('pos_c_syl_in_c_phrase_bw', [None] + range(1, 21)),
        getEqualityThreshQG('num_stressed_syl_before_c_syl_in_c_phrase',
                            [None] + range(1, 13)),
        getEqualityThreshQG('num_stressed_syl_after_c_syl_in_c_phrase',
                            [None] + range(1, 13)),
        getEqualityThreshQG('num_accented_syl_before_c_syl_in_c_phrase',
                            [None] + range(1, 7)),
        getEqualityThreshQG('num_accented_syl_after_c_syl_in_c_phrase',
                            [None] + range(1, 8)),
        getEqualityThreshQG('num_syl_from_prev_stressed_syl',
                            [None] + range(0, 6)),
        getEqualityThreshQG('num_syl_from_next_stressed_syl',
                            [None] + range(0, 6)),
        getEqualityThreshQG('num_syl_from_prev_accented_syl',
                            [None] + range(0, 17)),
        getEqualityThreshQG('num_syl_from_next_accented_syl',
                            [None] + range(0, 17)),
        (ques.AttrLabelValuer('c_syl_vowel'),
            [
                ques.EqualityQuestion(None),
                ques.EqualityQuestion('novowel')
            ] + ques.getSubsetQuestions(phoneset.namedVowelSubsets)
        ),
        getEqualityQG('r_syl_stress', [True]),
        getEqualityQG('r_syl_accent', [True]),
        getEqualityThreshQG('r_syl_num_segs', range(0, 8)),
        getEqualityQG('l_word_gpos', [None] + labels_hts_demo.getGposList()),
        getEqualityThreshQG('l_word_num_syls', range(0, 8)),
        getEqualityQG('c_word_gpos', [None] + labels_hts_demo.getGposList()),
        getEqualityThreshQG('c_word_num_syls', [None] + range(1, 8)),
        getEqualityThreshQG('pos_c_word_in_c_phrase_fw',
                            [None] + range(1, 14)),
        getEqualityThreshQG('pos_c_word_in_c_phrase_bw',
                            [None] + range(0, 14), range(1, 14)),
        getEqualityThreshQG('num_cont_word_before_c_word_in_c_phrase',
                            [None] + range(1, 10)),
        getEqualityThreshQG('num_cont_word_after_c_word_in_c_phrase',
                            [None] + range(0, 9)),
        getEqualityThreshQG('num_words_from_prev_cont_word',
                            [None] + range(0, 6)),
        getEqualityThreshQG('num_words_from_next_cont_word',
                            [None] + range(0, 6)),
        getEqualityQG('r_word_gpos', [None] + labels_hts_demo.getGposList()),
        getEqualityThreshQG('r_word_num_syls', range(0, 8)),
        getEqualityThreshQG('l_phrase_num_syls', range(0, 21)),
        getEqualityThreshQG('l_phrase_num_words', range(0, 14)),
        getEqualityThreshQG('c_phrase_num_syls', [None] + range(0, 21)),
        getEqualityThreshQG('c_phrase_num_words', [None] + range(0, 14)),
        getEqualityThreshQG('pos_c_phrase_in_utterance_fw', range(1, 5)),
        getEqualityThreshQG('pos_c_phrase_in_utterance_bw', range(1, 5)),
        getEqualityQG('c_phrase_ToBI_end_tone',
                      [None] + labels_hts_demo.getTobiList()),
        getEqualityThreshQG('r_phrase_num_syls', range(0, 21)),
        getEqualityThreshQG('r_phrase_num_words', range(0, 16)),
        getEqualityThreshQG('num_syls_in_utterance', range(1, 29)),
        getEqualityThreshQG('num_words_in_utterance', range(1, 14)),
        getEqualityThreshQG('num_phrases_in_utterance', range(1, 5)),
    ]
    return quinphoneQuestionGroups + otherQuestionGroups
