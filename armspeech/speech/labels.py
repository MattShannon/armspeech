"""Representation and I/O for labels and alignments."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import re

from codedep import codeDeps

@codeDeps()
def readHtkLabFile(labFile, framePeriod, decode = lambda labelString: labelString):
    """Reads HTK-style label file."""
    divisor = framePeriod * 1e7
    alignment = []
    for line in open(labFile):
        startTicks, endTicks, labelString = line.strip().split()
        startTime = int(int(startTicks) * 1.0 / divisor + 0.5)
        endTime = int(int(endTicks) * 1.0 / divisor + 0.5)
        label = decode(labelString)
        alignment.append((startTime, endTime, label, None))
    return alignment

@codeDeps(readHtkLabFile)
def readTwoLevelHtkLabFile(labFile, framePeriod, decodeHigh = lambda labelString: labelString, decodeLow = lambda labelString: labelString, fallbackToOneLevel = False):
    """Reads two-level HTK-style label file."""
    divisor = framePeriod * 1e7
    groupedLabels = []
    for line in open(labFile):
        lineParts = line.strip().split()
        if len(lineParts) == 4:
            startTicks, endTicks, lowString, highString = lineParts
            groupedLabels.append((decodeHigh(highString), []))
        else:
            if len(lineParts) != 3 or not groupedLabels:
                if fallbackToOneLevel:
                    return readHtkLabFile(labFile, framePeriod, decode = decodeHigh)
                else:
                    raise RuntimeError(str(labFile)+' does not appear to be a valid two-level label file')
            startTicks, endTicks, lowString, = lineParts
        startTime = int(int(startTicks) * 1.0 / divisor + 0.5)
        endTime = int(int(endTicks) * 1.0 / divisor + 0.5)
        groupedLabels[-1][1].append((startTime, endTime, decodeLow(lowString), None))
    alignment = [ (subAlignment[0][0], subAlignment[-1][1], label, subAlignment) for label, subAlignment in groupedLabels ]
    return alignment

@codeDeps()
def writeHtkLabFile(alignment, labFile, framePeriod,
                    encode = lambda label: label):
    """Writes HTK-style label file."""
    divisor = framePeriod * 1e7
    with open(labFile, 'w') as f:
        for startTime, endTime, label, subAlignment in alignment:
            assert subAlignment is None
            startTicks = int(startTime * divisor + 0.5)
            endTicks = int(endTime * divisor + 0.5)
            f.write('%s %s %s\n' % (startTicks, endTicks, encode(label)))

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
