"""Representation and I/O for HTS demo-style labels."""

# Copyright 2011, 2012 Matt Shannon
# The following copyrights may apply to the label format:
#     Copyright 2001-2008 Nagoya Institute of Technology, Department of Computer Science
#     Copyright 2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.speech.labels as lab

import collections

phonePat = r'[a-z]+'
phoneOrNonePat = r'(x|[a-z]+)'
numPat = r'[0-9]+'
numOrNonePat = r'(x|[0-9]+)'
boolPat = r'[01]'
boolOrNonePat = r'[x01]'
gposOrNonePat = r'(x|aux|cc|content|det|in|md|pps|punc|to|wp)'
gposOrZeroPat = r'(0|aux|cc|content|det|in|md|pps|punc|to|wp)'
tobiOrZeroPat = r'(0|L-L%|L-H%|H-H%|NONE)'

def orNoneDecode(s):
    return None if s == 'x' else s
def orZeroDecode(s):
    return None if s == '0' else s
def numDecode(s):
    return int(s)
def numOrNoneDecode(s):
    return None if s == 'x' else int(s)
def boolDecode(s):
    return s == '1'
def boolOrNoneDecode(s):
    return None if s == 'x' else s == '1'

labelFormat = [
    ('ll_phone', phoneOrNonePat, orNoneDecode, '^'),
    ('l_phone', phoneOrNonePat, orNoneDecode, '-'),
    ('phone', phonePat, None, '+'),
    ('r_phone', phoneOrNonePat, orNoneDecode, '='),
    ('rr_phone', phoneOrNonePat, orNoneDecode, '@'),
    ('seg_fw', numOrNonePat, numOrNoneDecode, '_'),
    ('seg_bw', numOrNonePat, numOrNoneDecode, '/A:'),
    ('l_syl_stress', boolPat, boolDecode, '_'),
    ('l_syl_accent', boolPat, boolDecode, '_'),
    ('l_syl_num_segs', numPat, numDecode, '/B:'),
    ('c_syl_stress', boolOrNonePat, boolOrNoneDecode, '-'),
    ('c_syl_accent', boolOrNonePat, boolOrNoneDecode, '-'),
    ('c_syl_num_segs', numOrNonePat, numOrNoneDecode, '@'),
    ('pos_c_syl_in_c_word_fw', numOrNonePat, numOrNoneDecode, '-'),
    ('pos_c_syl_in_c_word_bw', numOrNonePat, numOrNoneDecode, '&'),
    ('pos_c_syl_in_c_phrase_fw', numOrNonePat, numOrNoneDecode, '-'),
    ('pos_c_syl_in_c_phrase_bw', numOrNonePat, numOrNoneDecode, '#'),
    ('num_stressed_syl_before_c_syl_in_c_phrase', numOrNonePat, numOrNoneDecode, '-'),
    ('num_stressed_syl_after_c_syl_in_c_phrase', numOrNonePat, numOrNoneDecode, '$'),
    ('num_accented_syl_before_c_syl_in_c_phrase', numOrNonePat, numOrNoneDecode, '-'),
    ('num_accented_syl_after_c_syl_in_c_phrase', numOrNonePat, numOrNoneDecode, '!'),
    ('num_syl_from_prev_stressed_syl', numOrNonePat, numOrNoneDecode, '-'),
    ('num_syl_from_next_stressed_syl', numOrNonePat, numOrNoneDecode, ';'),
    ('num_syl_from_prev_accented_syl', numOrNonePat, numOrNoneDecode, '-'),
    ('num_syl_from_next_accented_syl', numOrNonePat, numOrNoneDecode, '|'),
    ('c_syl_vowel', phoneOrNonePat, orNoneDecode, '/C:'),
    ('r_syl_stress', boolPat, boolDecode, '+'),
    ('r_syl_accent', boolPat, boolDecode, '+'),
    ('r_syl_num_segs', numPat, numDecode, '/D:'),
    ('l_word_gpos', gposOrZeroPat, orZeroDecode, '_'),
    ('l_word_num_syls', numPat, numDecode, '/E:'),
    ('c_word_gpos', gposOrNonePat, orNoneDecode, '+'),
    ('c_word_num_syls', numOrNonePat, numOrNoneDecode, '@'),
    ('pos_c_word_in_c_phrase_fw', numOrNonePat, numOrNoneDecode, '+'),
    ('pos_c_word_in_c_phrase_bw', numOrNonePat, numOrNoneDecode, '&'),
    ('num_cont_word_before_c_word_in_c_phrase', numOrNonePat, numOrNoneDecode, '+'),
    ('num_cont_word_after_c_word_in_c_phrase', numOrNonePat, numOrNoneDecode, '#'),
    ('num_words_from_prev_cont_word', numOrNonePat, numOrNoneDecode, '+'),
    ('num_words_from_next_cont_word', numOrNonePat, numOrNoneDecode, '/F:'),
    ('r_word_gpos', gposOrZeroPat, orZeroDecode, '_'),
    ('r_word_num_syls', numPat, numDecode, '/G:'),
    ('l_phrase_num_syls', numPat, numDecode, '_'),
    ('l_phrase_num_words', numPat, numDecode, '/H:'),
    ('c_phrase_num_syls', numOrNonePat, numOrNoneDecode, '='),
    ('c_phrase_num_words', numOrNonePat, numOrNoneDecode, '@'),
    ('pos_c_phrase_in_utterance_fw', numPat, numDecode, '='),
    ('pos_c_phrase_in_utterance_bw', numPat, numDecode, '|'),
    ('c_phrase_ToBI_end_tone', tobiOrZeroPat, orZeroDecode, '/I:'),
    ('r_phrase_num_syls', numPat, numDecode, '='),
    ('r_phrase_num_words', numPat, numDecode, '/J:'),
    ('num_syls_in_utterance', numPat, numDecode, '+'),
    ('num_words_in_utterance', numPat, numDecode, '-'),
    ('num_phrases_in_utterance', numPat, numDecode, '')
]

labelKeys = [ labelKey for labelKey, pat, decode, sep in labelFormat ]
Label = collections.namedtuple('Label', labelKeys)
parseLabel = lab.getParseLabel(labelFormat, Label)
