"""Helper functions for sequence types."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from __future__ import division

from collections import deque

from codedep import codeDeps

@codeDeps()
def contextualizeIter(contextLength, iter, fillFrames = []):
    assert len(fillFrames) <= contextLength
    context = deque(fillFrames)
    for curr in iter:
        yield list(context), curr
        context.append(curr)
        if len(context) > contextLength:
            context.popleft()

@codeDeps()
def getChunk(xs, chunkIndex, numChunks):
    assert numChunks >= 1
    n = len(xs)
    chunkStart = (n * chunkIndex) // numChunks
    chunkEnd = (n * (chunkIndex + 1)) // numChunks
    return xs[chunkStart:chunkEnd]

@codeDeps(getChunk)
def chunkList(xs, numChunks):
    """Splits list into roughly evenly-sized chunks."""
    assert numChunks >= 1
    return [ getChunk(xs, chunkIndex, numChunks) for chunkIndex in range(numChunks) ]
