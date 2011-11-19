"""Representation and I/O for labels and alignments."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import re
from armspeech.util import collectionshelp

def readHtkLabFile(labFile, framePeriod, decode = lambda labelString: labelString):
    """Reads HTK-style label file."""
    divisor = framePeriod * 1.0e7
    for line in open(labFile):
        startTicks, endTicks, labelString = line.strip().split(None, 2)
        startTime = int(int(startTicks) / divisor + 0.5)
        endTime = int(int(endTicks) / divisor + 0.5)
        label = decode(labelString)
        yield startTime, endTime, label

# (FIXME : this should probably be moved into modelling subpackage)
def checkAlignment(alignment, allowZeroDur = True):
    """Checks alignment is in order, contiguous and non-overlapping."""
    if alignment:
        for (startTimePrev, endTimePrev, labelPrev), (startTime, endTime, label) in zip(alignment, alignment[1:]):
            if startTime < endTimePrev:
                raise RuntimeError('alignment has overlaps')
            elif startTime > endTimePrev:
                raise RuntimeError('alignment is not contiguous')
    for startTime, endTime, label in alignment:
        if endTime < startTime:
            raise RuntimeError('alignment has segment of negative duration')
        if endTime == startTime and not allowZeroDur:
            raise RuntimeError('alignment has zero duration segment')

def getLabelClass(className, labelFormat):
    labelKeys = []
    labelReStrings = []
    decodeDict = dict()
    for labelKey, pat, decode, sep in labelFormat:
        labelKeys.append(labelKey)
        labelReStrings.append(r'(?P<'+labelKey+r'>'+pat+r')'+re.escape(sep))
        if decode != None:
            decodeDict[labelKey] = decode

    labelRe = re.compile(r''.join(labelReStrings)+r'$')
    Label = collectionshelp.namedtuple(className, labelKeys)

    def parseLabel(labelString):
        match = labelRe.match(labelString)
        if not match:
            raise RuntimeError('label '+labelString+' not of required format')
        matchDict = match.groupdict()
        for labelKey in decodeDict:
            matchDict[labelKey] = decodeDict[labelKey](matchDict[labelKey])
        label = Label(**matchDict)
        return label

    return Label, parseLabel
