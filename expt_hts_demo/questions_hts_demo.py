"""Decision tree questions from HTS demo."""

# Copyright 2011, 2012 Matt Shannon
# The follow copyrights may apply to the list of questions:
#     Copyright 2001-2008 Nagoya Institute of Technology, Department of Computer Science
#     Copyright 2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.modelling.questions as ques
from expt_hts_demo import labels_hts_demo
from codedep import codeDeps

@codeDeps(ques.AttrLabelValuer, ques.getSubsetQuestions)
def getSubsetQG(labelKey, namedSubsets):
    return (ques.AttrLabelValuer(labelKey), ques.getSubsetQuestions(namedSubsets))
@codeDeps(ques.AttrLabelValuer, ques.getEqualityQuestions)
def getEqualityQG(labelKey, values):
    return (ques.AttrLabelValuer(labelKey), ques.getEqualityQuestions(values))
@codeDeps(ques.AttrLabelValuer, ques.getThreshQuestions)
def getThreshQG(labelKey, threshes):
    return (ques.AttrLabelValuer(labelKey), ques.getThreshQuestions(threshes))
@codeDeps(ques.AttrLabelValuer, ques.getSubsetQuestions)
def getSubsetQGs(labelKeys, namedSubsets):
    questions = ques.getSubsetQuestions(namedSubsets)
    return [ (ques.AttrLabelValuer(labelKey), questions) for labelKey in labelKeys ]

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
@codeDeps(getEqualityQG, getQuinphoneQuestionGroups, getThreshQG,
    labels_hts_demo.getGposList, labels_hts_demo.getTobiList,
    ques.AttrLabelValuer, ques.EqualityQuestion, ques.getSubsetQuestions
)
def getFullContextQuestionGroups(phoneset):
    quinphoneQuestionGroups = getQuinphoneQuestionGroups(phoneset)
    otherQuestionGroups = [
        getEqualityQG('seg_fw', [None] + range(1, 8)),
        getThreshQG('seg_fw', range(1, 8)),
        getEqualityQG('seg_bw', [None] + range(1, 8)),
        getThreshQG('seg_bw', range(0, 8)),
        getEqualityQG('l_syl_stress', [True]),
        getEqualityQG('l_syl_accent', [True]),
        getEqualityQG('l_syl_num_segs', range(0, 8)),
        getThreshQG('l_syl_num_segs', range(1, 8)),
        getEqualityQG('c_syl_stress', [True, False, None]),
        getEqualityQG('c_syl_accent', [True, False, None]),
        getEqualityQG('c_syl_num_segs', [None] + range(1, 8)),
        getThreshQG('c_syl_num_segs', range(1, 8)),
        getEqualityQG('pos_c_syl_in_c_word_fw', [None] + range(1, 8)),
        getThreshQG('pos_c_syl_in_c_word_fw', range(1, 8)),
        getEqualityQG('pos_c_syl_in_c_word_bw', [None] + range(1, 8)),
        getThreshQG('pos_c_syl_in_c_word_bw', range(1, 8)),
        getEqualityQG('pos_c_syl_in_c_phrase_fw', [None] + range(1, 21)),
        getThreshQG('pos_c_syl_in_c_phrase_fw', range(1, 21)),
        getEqualityQG('pos_c_syl_in_c_phrase_bw', [None] + range(1, 21)),
        getThreshQG('pos_c_syl_in_c_phrase_bw', range(1, 21)),
        getEqualityQG('num_stressed_syl_before_c_syl_in_c_phrase',
                      [None] + range(1, 13)),
        getThreshQG('num_stressed_syl_before_c_syl_in_c_phrase', range(1, 13)),
        getEqualityQG('num_stressed_syl_after_c_syl_in_c_phrase',
                      [None] + range(1, 13)),
        getThreshQG('num_stressed_syl_after_c_syl_in_c_phrase', range(1, 13)),
        getEqualityQG('num_accented_syl_before_c_syl_in_c_phrase',
                      [None] + range(1, 7)),
        getThreshQG('num_accented_syl_before_c_syl_in_c_phrase', range(1, 7)),
        getEqualityQG('num_accented_syl_after_c_syl_in_c_phrase',
                      [None] + range(1, 8)),
        getThreshQG('num_accented_syl_after_c_syl_in_c_phrase', range(1, 8)),
        getEqualityQG('num_syl_from_prev_stressed_syl', [None] + range(0, 6)),
        getThreshQG('num_syl_from_prev_stressed_syl', range(0, 6)),
        getEqualityQG('num_syl_from_next_stressed_syl', [None] + range(0, 6)),
        getThreshQG('num_syl_from_next_stressed_syl', range(0, 6)),
        getEqualityQG('num_syl_from_prev_accented_syl', [None] + range(0, 17)),
        getThreshQG('num_syl_from_prev_accented_syl', range(0, 17)),
        getEqualityQG('num_syl_from_next_accented_syl', [None] + range(0, 17)),
        getThreshQG('num_syl_from_next_accented_syl', range(0, 17)),
        (ques.AttrLabelValuer('c_syl_vowel'),
            [
                ques.EqualityQuestion(None),
                ques.EqualityQuestion('novowel')
            ] + ques.getSubsetQuestions(phoneset.namedVowelSubsets)
        ),
        getEqualityQG('r_syl_stress', [True]),
        getEqualityQG('r_syl_accent', [True]),
        getEqualityQG('r_syl_num_segs', range(0, 8)),
        getThreshQG('r_syl_num_segs', range(1, 8)),
        getEqualityQG('l_word_gpos', [None] + labels_hts_demo.getGposList()),
        getEqualityQG('l_word_num_syls', range(0, 8)),
        getThreshQG('l_word_num_syls', range(1, 8)),
        getEqualityQG('c_word_gpos', [None] + labels_hts_demo.getGposList()),
        getEqualityQG('c_word_num_syls', [None] + range(1, 8)),
        getThreshQG('c_word_num_syls', range(1, 8)),
        getEqualityQG('pos_c_word_in_c_phrase_fw', [None] + range(1, 14)),
        getThreshQG('pos_c_word_in_c_phrase_fw', range(1, 14)),
        getEqualityQG('pos_c_word_in_c_phrase_bw', [None] + range(0, 14)),
        getThreshQG('pos_c_word_in_c_phrase_bw', range(1, 14)),
        getEqualityQG('num_cont_word_before_c_word_in_c_phrase',
                      [None] + range(1, 10)),
        getThreshQG('num_cont_word_before_c_word_in_c_phrase', range(1, 10)),
        getEqualityQG('num_cont_word_after_c_word_in_c_phrase',
                      [None] + range(0, 9)),
        getThreshQG('num_cont_word_after_c_word_in_c_phrase', range(0, 9)),
        getEqualityQG('num_words_from_prev_cont_word', [None] + range(0, 6)),
        getThreshQG('num_words_from_prev_cont_word', range(0, 6)),
        getEqualityQG('num_words_from_next_cont_word', [None] + range(0, 6)),
        getThreshQG('num_words_from_next_cont_word', range(0, 6)),
        getEqualityQG('r_word_gpos', [None] + labels_hts_demo.getGposList()),
        getEqualityQG('r_word_num_syls', range(0, 8)),
        getThreshQG('r_word_num_syls', range(1, 8)),
        getEqualityQG('l_phrase_num_syls', range(0, 21)),
        getThreshQG('l_phrase_num_syls', range(1, 21)),
        getEqualityQG('l_phrase_num_words', range(0, 14)),
        getThreshQG('l_phrase_num_words', range(1, 14)),
        getEqualityQG('c_phrase_num_syls', [None] + range(0, 21)),
        getThreshQG('c_phrase_num_syls', range(0, 21)),
        getEqualityQG('c_phrase_num_words', [None] + range(0, 14)),
        getThreshQG('c_phrase_num_words', range(0, 14)),
        getEqualityQG('pos_c_phrase_in_utterance_fw', range(1, 5)),
        getThreshQG('pos_c_phrase_in_utterance_fw', range(2, 5)),
        getEqualityQG('pos_c_phrase_in_utterance_bw', range(1, 5)),
        getThreshQG('pos_c_phrase_in_utterance_bw', range(2, 5)),
        getEqualityQG('c_phrase_ToBI_end_tone',
                      [None] + labels_hts_demo.getTobiList()),
        getEqualityQG('r_phrase_num_syls', range(0, 21)),
        getThreshQG('r_phrase_num_syls', range(1, 21)),
        getEqualityQG('r_phrase_num_words', range(0, 16)),
        getThreshQG('r_phrase_num_words', range(1, 16)),
        getEqualityQG('num_syls_in_utterance', range(1, 29)),
        getThreshQG('num_syls_in_utterance', range(2, 29)),
        getEqualityQG('num_words_in_utterance', range(1, 14)),
        getThreshQG('num_words_in_utterance', range(2, 14)),
        getEqualityQG('num_phrases_in_utterance', range(1, 5)),
        getThreshQG('num_phrases_in_utterance', range(2, 5)),
    ]
    return quinphoneQuestionGroups + otherQuestionGroups
