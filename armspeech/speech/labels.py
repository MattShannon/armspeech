"""Representation and I/O for labels and alignments."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import re
import collections

def readHtkLabFile(labFile, framePeriod, decode = lambda labelString: labelString):
    """Reads HTK-style label file."""
    divisor = framePeriod * 1e7
    alignment = []
    for line in open(labFile):
        startTicks, endTicks, labelString = line.strip().split()
        startTime = int(int(startTicks) / divisor + 0.5)
        endTime = int(int(endTicks) / divisor + 0.5)
        label = decode(labelString)
        alignment.append((startTime, endTime, label, None))
    return alignment

# (FIXME : this should probably be moved into modelling subpackage)
def checkAlignment(alignment, startTimeReq = None, endTimeReq = None, allowZeroDur = True):
    """Checks an alignment for various possible inconsistencies.

    Checks alignment is in order, contiguous and non-overlapping. Recursively
    checks any sub-alignments, and checks that start and end times of
    sub-alignments agree with their parents.
    """
    if alignment:
        for (startTimePrev, endTimePrev, labelPrev, subAlignmentPrev), (startTime, endTime, label, subAlignment) in zip(alignment, alignment[1:]):
            if startTime < endTimePrev:
                raise RuntimeError('alignment has overlaps')
            elif startTime > endTimePrev:
                raise RuntimeError('alignment is not contiguous')
    if startTimeReq is not None:
        startTime, endTime, label, subAlignment = alignment[0]
        if startTime != startTimeReq:
            raise RuntimeError('alignment start time is incorrect ('+str(startTimeReq)+' desired, '+str(startTime)+' actual)')
    if endTimeReq is not None:
        startTime, endTime, label, subAlignment = alignment[-1]
        if endTime != endTimeReq:
            raise RuntimeError('alignment end time is incorrect ('+str(endTimeReq)+' desired, '+str(endTime)+' actual)')
    for startTime, endTime, label, subAlignment in alignment:
        if endTime < startTime:
            raise RuntimeError('alignment has segment of negative duration')
        if endTime == startTime and not allowZeroDur:
            raise RuntimeError('alignment has zero duration segment')
        if subAlignment is not None:
            checkAlignment(subAlignment, startTimeReq = startTime, endTimeReq = endTime, allowZeroDur = allowZeroDur)

def getLabelClass(className, labelFormat):
    labelKeys = []
    labelReStrings = []
    decodeDict = dict()
    for labelKey, pat, decode, sep in labelFormat:
        labelKeys.append(labelKey)
        labelReStrings.append(r'(?P<'+labelKey+r'>'+pat+r')'+re.escape(sep))
        if decode is not None:
            decodeDict[labelKey] = decode

    labelRe = re.compile(r''.join(labelReStrings)+r'$')
    Label = collections.namedtuple(className, labelKeys)

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
