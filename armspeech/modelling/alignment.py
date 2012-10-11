"""Representation and I/O for labels and alignments."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

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
