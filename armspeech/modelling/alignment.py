"""Representation and I/O for labels and alignments."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from __future__ import division

import sys

from codedep import codeDeps

from armspeech.util.util import identityFn

@codeDeps()
def checkAlignment(alignment, startTimeReq = None, endTimeReq = None,
                   allowZeroDur = True):
    """Checks an alignment for various possible inconsistencies.

    Checks alignment is in order, contiguous and non-overlapping. Recursively
    checks any sub-alignments, and checks that start and end times of
    sub-alignments agree with their parents.
    """
    if alignment:
        for (
            (startTimePrev, endTimePrev, labelPrev, subAlignmentPrev),
            (startTime, endTime, label, subAlignment)
        ) in zip(alignment, alignment[1:]):
            if startTime < endTimePrev:
                raise RuntimeError('alignment has overlaps')
            elif startTime > endTimePrev:
                raise RuntimeError('alignment is not contiguous')
    if startTimeReq is not None:
        startTime, endTime, label, subAlignment = alignment[0]
        if startTime != startTimeReq:
            raise RuntimeError('alignment start time is incorrect (%s desired,'
                               ' %s actual)' % (startTimeReq, startTime))
    if endTimeReq is not None:
        startTime, endTime, label, subAlignment = alignment[-1]
        if endTime != endTimeReq:
            raise RuntimeError('alignment end time is incorrect (%s desired,'
                               ' %s actual)' % (endTimeReq, endTime))
    for startTime, endTime, label, subAlignment in alignment:
        if endTime < startTime:
            raise RuntimeError('alignment has segment of negative duration')
        if endTime == startTime and not allowZeroDur:
            raise RuntimeError('alignment has zero duration segment')
        if subAlignment is not None:
            checkAlignment(subAlignment, startTimeReq = startTime,
                           endTimeReq = endTime, allowZeroDur = allowZeroDur)

@codeDeps()
def alignmentTo1(alignment):
    """Converts an alignment to one-level by ignoring lower level."""
    return [ (startTime, endTime, label, None)
             for startTime, endTime, label, subAlignment in alignment ]

@codeDeps()
def alignmentTo2(alignment):
    """Converts an alignment to two-level with one sub-label."""
    return [ (startTime, endTime, label, [(startTime, endTime, 0, None)])
             for startTime, endTime, label, subAlignment in alignment ]

@codeDeps()
def uniformSegmentAlignment(alignment, subLabelsOut):
    """Converts a one-level alignment to two-level by uniform segmentation."""
    alignmentOut = []
    numSubLabels = len(subLabelsOut)
    for labelStartTime, labelEndTime, label, subAlignment in alignment:
        assert subAlignment is None
        durMult = (labelEndTime - labelStartTime) * 1.0 / numSubLabels
        subAlignmentOut = []
        for subLabelIndex, subLabel in enumerate(subLabelsOut):
            startTime = int(durMult * subLabelIndex + 0.5) + labelStartTime
            endTime = int(durMult * (subLabelIndex + 1) + 0.5) + labelStartTime
            subAlignmentOut.append((startTime, endTime, subLabel, None))
        alignmentOut.append(
            (labelStartTime, labelEndTime, label, subAlignmentOut)
        )
    return alignmentOut

@codeDeps(alignmentTo1, alignmentTo2, uniformSegmentAlignment)
class StandardizeAlignment(object):
    def __init__(self, subLabelsBefore, subLabelsAfter):
        if subLabelsBefore is None:
            self.numSubLabelsBefore = None
        else:
            assert subLabelsBefore == list(range(len(subLabelsBefore)))
            self.numSubLabelsBefore = len(subLabelsBefore)
        if subLabelsAfter is None:
            self.numSubLabelsAfter = None
        else:
            assert subLabelsAfter == list(range(len(subLabelsAfter)))
            self.numSubLabelsAfter = len(subLabelsAfter)

        if not (self.numSubLabelsBefore == self.numSubLabelsAfter or
                self.numSubLabelsAfter in (None, 1)):
            sys.stderr.write('WARNING: cannot convert alignment with %s'
                             ' sub-labels to alignment with %s sub-labels'
                             ' exactly -- using uniform segmentation\n' %
                             (self.numSubLabelsBefore, self.numSubLabelsAfter))

    def __call__(self, alignment):
        """Returns a standardized, state-level alignment."""
        if self.numSubLabelsBefore == self.numSubLabelsAfter:
            return alignment
        elif self.numSubLabelsAfter in (None, 1):
            if self.numSubLabelsBefore is not None:
                alignment = alignmentTo1(alignment)
            if self.numSubLabelsAfter == 1:
                alignment = alignmentTo2(alignment)
            return alignment
        else:
            if self.numSubLabelsBefore is not None:
                alignment = alignmentTo1(alignment)
            return uniformSegmentAlignment(alignment,
                                           list(range(self.numSubLabelsAfter)))

@codeDeps(identityFn)
class AlignmentToPhoneticSeq(object):
    def __init__(self, mapAlignment = identityFn, mapLabel = identityFn):
        self.mapAlignment = mapAlignment
        self.mapLabel = mapLabel

    def toPhoneticIter(self, alignment):
        mapLabel = self.mapLabel
        for labelStartTime, labelEndTime, label, subAlignment in self.mapAlignment(alignment):
            labelOut = mapLabel(label)
            if subAlignment is None:
                # 1-level alignment
                for time in range(labelStartTime, labelEndTime):
                    yield labelOut
            else:
                # 2-level alignment
                for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                    assert subSubAlignment is None
                    for time in range(startTime, endTime):
                        yield labelOut, subLabel

    def __call__(self, alignment):
        return list(self.toPhoneticIter(alignment))

@codeDeps(identityFn)
class AlignmentToPhoneticSeqWithTiming(object):
    def __init__(self, mapAlignment = identityFn, mapLabel = identityFn,
                 mapTiming = identityFn):
        self.mapAlignment = mapAlignment
        self.mapLabel = mapLabel
        self.mapTiming = mapTiming

    def toPhoneticIter(self, alignment):
        mapLabel = self.mapLabel
        mapTiming = self.mapTiming
        for labelStartTime, labelEndTime, label, subAlignment in self.mapAlignment(alignment):
            labelOut = mapLabel(label)
            for startTime, endTime, subLabel, subSubAlignment in subAlignment:
                assert subSubAlignment is None
                for time in range(startTime, endTime):
                    framesBefore = time - startTime
                    framesAfter = endTime - time - 1
                    yield (labelOut, subLabel), mapTiming((framesBefore, framesAfter))

    def __call__(self, alignment):
        return list(self.toPhoneticIter(alignment))
