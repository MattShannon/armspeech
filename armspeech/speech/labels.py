"""Representation and I/O for labels and alignments."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import re

from codedep import codeDeps

@codeDeps()
def getParseLabel(labelFormat, createLabel):
    labelReStrings = []
    decodeDict = dict()
    for labelKey, pat, decode, sep in labelFormat:
        labelReStrings.append(r'(?P<'+labelKey+r'>'+pat+r')'+re.escape(sep))
        if decode is not None:
            decodeDict[labelKey] = decode

    labelRe = re.compile(r''.join(labelReStrings)+r'$')

    def parseLabel(labelString):
        match = labelRe.match(labelString)
        if not match:
            raise RuntimeError('label '+labelString+' not of required format')
        matchDict = match.groupdict()
        for labelKey in decodeDict:
            matchDict[labelKey] = decodeDict[labelKey](matchDict[labelKey])
        label = createLabel(**matchDict)
        return label

    return parseLabel
